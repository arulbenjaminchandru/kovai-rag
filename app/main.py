"""The Kovai Finserv RAG API.

Three endpoints for humans (/health, /ready, /metrics) and one for customers
(/ask). Everything a customer asks passes through guardrails on the way in and
on the way out; see app/guardrails.py.
"""

import collections
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import guardrails, rag
from app.config import settings
from app.schemas import AskRequest, AskResponse, HealthResponse, ReadyResponse

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


# In-process metrics. Deliberately not Prometheus: on a 512 MB free instance,
# a Counter and a bounded deque are enough to answer "is it working and how
# slow is it", and they cost nothing.
STATS = collections.Counter()

# A deque with maxlen rather than a plain list. /ask is a sync endpoint, so
# FastAPI runs it in a threadpool and several requests can append concurrently;
# a bounded deque is atomic on append and cannot grow without limit.
LATENCY_WINDOW = 200
LATENCIES = collections.deque(maxlen=LATENCY_WINDOW)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up before the first customer arrives.

    The first query to a cold process is slow: ChromaDB has to load the ONNX
    embedding model into memory, which takes seconds. Doing that here means the
    first real customer does not pay for it.

    A warmup failure must NOT prevent startup. If the index is missing we still
    want the process up and answering /ready with the bad news, rather than
    crash-looping where nobody can see why.
    """
    logger.info(
        "starting version=%s answer_model=%s guard_model=%s",
        settings.app_version,
        settings.answer_model,
        settings.guard_model,
    )
    try:
        count = rag.get_collection().count()
        logger.info("index loaded chunks=%d", count)

        # Forces the embedding model to load now rather than on the first
        # customer question.
        rag.retrieve("warmup", k=1)

        rag.get_claude()
        logger.info("warmup complete")
    except Exception:
        logger.exception("warmup failed; /ready will report the details")

    yield

    logger.info("shutting down")


app = FastAPI(
    title="Kovai Finserv Policy Assistant",
    version=settings.app_version,
    description="Answers customer questions from Kovai Finserv's loan policy documents.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Wide open for now because the Streamlit UI is deployed separately and its
# hostname is not fixed yet. RESTRICT THIS to the real Streamlit domain before
# going live — a public API with allow_origins=["*"] lets any page on the
# internet spend our Anthropic budget.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a request id to every request and echo it back.

    This is how you trace one customer through the logs. They quote the
    x-request-id from their browser, you grep it, and you get every line this
    process logged for that one question — without ever having stored the
    question text.
    """
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Log the traceback, tell the customer nothing.

    A stack trace names our files, our libraries and our versions. It goes in
    the log, never in the response body.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    STATS["errors"] += 1
    logger.exception("unhandled exception request_id=%s", request_id)

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Something went wrong on our side. Please try again, or "
            "contact care@kovaifinserv.example.",
            "request_id": request_id,
        },
    )


def _percentile(values, pct):
    """Nearest-rank percentile. Good enough for a 200-sample window."""
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(len(ordered) * pct / 100), len(ordered) - 1)
    return ordered[index]


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness only. Checks NOTHING external, on purpose.

    A health check that touches ChromaDB or Anthropic will fail when they are
    slow, and the orchestrator will kill a process that was perfectly capable
    of serving traffic. This endpoint answers exactly one question: is the
    process running? Use /ready to ask whether it can actually do its job.
    """
    return HealthResponse(status="ok", version=settings.app_version)


@app.get("/ready", response_model=ReadyResponse)
def ready():
    """Readiness. This one is allowed to fail, and should.

    It touches the index, so it fails when the index is missing or corrupt —
    which is precisely the signal a load balancer needs to stop sending
    customers here.
    """
    try:
        count = rag.get_collection().count()
    except Exception as exc:
        logger.exception("readiness check failed")
        raise HTTPException(
            status_code=503,
            detail=f"Index unavailable: {type(exc).__name__}. Has scripts/ingest.py been run?",
        )

    if count == 0:
        raise HTTPException(
            status_code=503,
            detail="Index is empty. Run scripts/ingest.py.",
        )

    return ReadyResponse(status="ready", indexed_chunks=count, model=settings.answer_model)


@app.get("/metrics")
def metrics():
    """Counters and latency percentiles for this process.

    Per-process and in-memory, so it resets on restart and does not aggregate
    across instances. That is the honest scope of it.
    """
    latencies = list(LATENCIES)
    return {
        "counters": dict(STATS),
        "latency_ms": {
            "count": len(latencies),
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
        },
    }


def _finish(request_id, started, reason, answer, q_len, sources=None, blocked=False):
    """Stamp latency, record metrics, log exactly one line, build the response.

    Every return path in /ask goes through here so that the metrics and the log
    can never disagree with what the customer actually received.

    The log line carries lengths and decisions, NEVER the question or answer
    text. Both can contain customer PII — a phone number, a PAN, an account
    number — and anything at INFO ends up in a log aggregator we do not control
    and cannot easily purge. Lengths are enough to debug with.
    """
    latency_ms = int((time.perf_counter() - started) * 1000)
    LATENCIES.append(latency_ms)
    STATS[reason] += 1
    STATS["requests"] += 1

    logger.info(
        "ask request_id=%s reason=%s blocked=%s q_len=%d a_len=%d ms=%d",
        request_id,
        reason,
        blocked,
        q_len,
        len(answer),
        latency_ms,
    )

    return AskResponse(
        answer=answer,
        sources=sources or [],
        blocked=blocked,
        reason=reason,
        request_id=request_id,
        latency_ms=latency_ms,
        model=settings.answer_model,
    )


@app.post("/ask", response_model=AskResponse)
@limiter.limit(settings.rate_limit)
def ask(request: Request, body: AskRequest):
    """Answer one customer question.

    Defined with `def`, NOT `async def`, and that is deliberate. Everything
    inside is blocking: the Anthropic SDK is synchronous and ChromaDB does
    synchronous local I/O. FastAPI runs a sync endpoint in a threadpool, so a
    slow question occupies one worker thread and the event loop stays free to
    accept other requests.

    Declaring this `async def` would run the blocking calls directly on the
    event loop and freeze the entire server for the duration of every Anthropic
    round trip. That is the most common FastAPI performance bug there is. If
    this ever needs to become `async def`, the SDK calls must move to
    AsyncAnthropic first.

    `request: Request` is the first parameter because slowapi needs it to find
    the client address for rate limiting.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    started = time.perf_counter()
    q_len = len(body.question)

    # 1. Guard the way in. Blocks injections and off-topic questions, and
    # redacts PII before the question reaches any model.
    decision = guardrails.guard_input(body.question)
    if not decision.allowed:
        return _finish(
            request_id,
            started,
            decision.reason,
            decision.message,
            q_len,
            blocked=True,
        )

    # 2. Retrieve and answer. A question with no relevant policy section never
    # reaches the model at all.
    result = rag.answer_question(decision.question)
    if not result.retrieved:
        return _finish(
            request_id,
            started,
            "NO_CONTEXT",
            result.answer,
            q_len,
            blocked=True,
        )

    # 3. Guard the way out. Checks our own answer against the sections it was
    # supposed to come from.
    checked = guardrails.guard_output(result.answer, result.contexts, rag.REFUSAL)

    return _finish(
        request_id,
        started,
        checked.reason,
        checked.answer,
        q_len,
        sources=result.sources,
        blocked=not checked.grounded,
    )

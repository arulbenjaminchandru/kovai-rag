"""Offline Ragas evaluation of the Kovai Finserv policy assistant.

Runs the 16 golden questions through the REAL pipeline in app/rag.py, then has
Claude judge four metrics per question and writes a markdown report.

Two variants:

  hardened  the pipeline exactly as it ships
  naive     the same pipeline with the guardrails switched off, to reproduce
            the mistakes the original bot made

The point of running both is that a single score means nothing on its own. A
faithfulness of 0.91 is only good if you can show the version without the
guardrails scores 0.6.

This file is never imported by app/. It never ships in the Docker image.

    python eval/run_ragas.py
    python eval/run_ragas.py --variant naive
    python eval/run_ragas.py --ci
"""

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from anthropic import AsyncAnthropic
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

# eval/ is run as a script from the repo root, so app/ is not on the path yet.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app import rag  # noqa: E402
from app.config import settings  # noqa: E402

DATASET_PATH = REPO_ROOT / "eval" / "golden_dataset.json"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

# Haiku is a capable enough judge for a dataset this size and keeps a full run
# cheap. If the scores look noisy — the same answer scoring 0.6 on one run and
# 0.9 on the next — switch to "claude-sonnet-5", which is a stronger judge.
JUDGE_MODEL = "claude-haiku-4-5-20251001"

# These thresholds are a PRODUCT decision, not a technical one. They encode how
# much of each failure mode the business is willing to ship.
#
# Faithfulness is highest because a made-up policy is the failure that cost
# Kovai real money: an answer that states a fee we do not charge. Everything
# else degrades the experience; that one creates a liability.
#
# Context precision is lowest because we deliberately retrieve 4 sections and
# most questions are answered by one. Retrieving three sections we did not need
# is wasteful, not dangerous.
THRESHOLDS = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.75,
    "context_precision": 0.60,
    "context_recall": 0.80,
}

METRIC_ORDER = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# The naive baseline. This is what the bot looked like before the guardrails:
# a polite assistant with no instruction to stay inside the documents, one
# retrieved section instead of four, and no relevance floor at all.
NAIVE_SYSTEM_PROMPT = "You are a helpful support assistant for Kovai Finserv."
NAIVE_TOP_K = 1
NAIVE_MIN_RELEVANCE = -1.0

# Cosine relevance is at worst 0.0 in practice, so a floor of -1.0 keeps every
# hit. The naive pipeline can therefore never take the cheap refusal path — it
# always has something to hand the model, which is exactly the bug.

# Ragas judges each row with four separate LLM calls. Four rows in flight is a
# comfortable ceiling under the default Anthropic rate limits and still cuts a
# 16-row run from minutes to well under one.
MAX_CONCURRENT_ROWS = 4


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def build_judge():
    """Return the Claude judge used by every metric.

    The client MUST be AsyncAnthropic. The ragas.metrics.collections metrics
    are async — ascore() calls agenerate() on the client — and a synchronous
    Anthropic client raises TypeError the moment the first metric runs.
    """
    judge = llm_factory(
        JUDGE_MODEL,
        provider="anthropic",
        client=AsyncAnthropic(api_key=settings.anthropic_api_key),
        max_tokens=2048,
    )

    # Ragas sets BOTH temperature and top_p by default. Some Claude models
    # reject a request that specifies both, so drop top_p and pin temperature
    # to 0 — a judge that scores the same answer differently on each run is
    # worse than no judge.
    judge.model_args.pop("top_p", None)
    judge.model_args["temperature"] = 0.0

    return judge


def build_metrics(judge):
    """Return the four metrics, keyed by the short name used everywhere else."""
    # Local ONNX/torch embeddings from sentence-transformers. Answer relevancy
    # is the only metric that needs them, and using the same MiniLM model the
    # index is built with keeps this project on a single API key.
    embeddings = embedding_factory(
        "huggingface", "sentence-transformers/all-MiniLM-L6-v2"
    )

    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecisionWithoutReference(llm=judge),
        "context_recall": ContextRecall(llm=judge),
    }


# ---------------------------------------------------------------------------
# Running the pipeline
# ---------------------------------------------------------------------------


def load_rows():
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return data["rows"]


def generate_answers(rows, variant):
    """Run every golden question through the real pipeline.

    For the naive variant the module-level knobs are patched in place and
    restored in a finally block, so a crash mid-run cannot leave the process
    holding a pipeline with its guardrails switched off.
    """
    original_prompt = rag.SYSTEM_PROMPT
    original_top_k = settings.top_k
    original_min_relevance = settings.min_relevance

    if variant == "naive":
        rag.SYSTEM_PROMPT = NAIVE_SYSTEM_PROMPT
        settings.top_k = NAIVE_TOP_K
        settings.min_relevance = NAIVE_MIN_RELEVANCE

    results = []
    try:
        for index, row in enumerate(rows, start=1):
            result = rag.answer_question(row["question"])
            print(
                f"  [{index:2d}/{len(rows)}] {row['id']} "
                f"({row['kind']}) sections={len(result.contexts)}"
            )
            results.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "question": row["question"],
                    "reference": row["reference"],
                    "answer": result.answer,
                    "contexts": list(result.contexts),
                    "sources": list(result.sources),
                }
            )
    finally:
        rag.SYSTEM_PROMPT = original_prompt
        settings.top_k = original_top_k
        settings.min_relevance = original_min_relevance

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


async def score_row(row, metrics, semaphore):
    """Score one answered question. Returns {metric: float | None}."""
    scores = {name: None for name in METRIC_ORDER}

    async with semaphore:
        # Answer relevancy is the only metric that does not read the contexts,
        # so it is the only one we can score on a refusal.
        scores["answer_relevancy"] = (
            await metrics["answer_relevancy"].ascore(
                user_input=row["question"],
                response=row["answer"],
            )
        ).value

        if not row["contexts"]:
            # A refusal. Faithfulness, context precision and context recall are
            # all undefined without contexts, and scoring them anyway produces
            # meaningless zeros that drag the average down — which makes a
            # CORRECT REFUSAL look like a failure. Record None and exclude the
            # row from those three means instead.
            return scores

        scores["faithfulness"] = (
            await metrics["faithfulness"].ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
            )
        ).value

        scores["context_precision"] = (
            await metrics["context_precision"].ascore(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
            )
        ).value

        scores["context_recall"] = (
            await metrics["context_recall"].ascore(
                user_input=row["question"],
                retrieved_contexts=row["contexts"],
                reference=row["reference"],
            )
        ).value

    return scores


async def score_all(answered, metrics):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ROWS)
    return await asyncio.gather(
        *(score_row(row, metrics, semaphore) for row in answered)
    )


def summarise(scored):
    """Mean of the non-None values per metric. None if a metric never scored."""
    summary = {}
    for name in METRIC_ORDER:
        values = [row[name] for row in scored if row[name] is not None]
        summary[name] = statistics.fmean(values) if values else None
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt(score):
    return "-" if score is None else f"{score:.3f}"


def verdicts(summary):
    """Return {metric: "PASS" | "FAIL" | "-"} against the thresholds."""
    result = {}
    for name in METRIC_ORDER:
        score = summary[name]
        if score is None:
            result[name] = "-"
        else:
            result[name] = "PASS" if score >= THRESHOLDS[name] else "FAIL"
    return result


def build_report(variant, answered, scored, summary):
    status = verdicts(summary)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    refusals = sum(1 for row in answered if not row["contexts"])

    lines = [
        f"# Ragas report — {variant}",
        "",
        f"Generated {stamp} · judge `{JUDGE_MODEL}` · {len(answered)} questions "
        f"· {refusals} refused without retrieval",
        "",
        "| metric | score | threshold | result |",
        "| --- | --- | --- | --- |",
    ]
    for name in METRIC_ORDER:
        lines.append(
            f"| {name} | {fmt(summary[name])} | {THRESHOLDS[name]:.2f} | {status[name]} |"
        )

    lines += [
        "",
        "A `-` means the metric had no scorable rows. Per question, `-` means the",
        "question was refused with no retrieved context, so the metric is undefined",
        "rather than zero.",
        "",
        "## Per question",
        "",
        "| id | kind | faithfulness | answer_relevancy | context_precision | context_recall |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row, scores in zip(answered, scored):
        cells = " | ".join(fmt(scores[name]) for name in METRIC_ORDER)
        lines.append(f"| {row['id']} | {row['kind']} | {cells} |")

    lines.append("")
    return "\n".join(lines)


def write_outputs(variant, answered, scored, summary):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = RESULTS_DIR / f"report_{variant}.md"
    report_path.write_text(build_report(variant, answered, scored, summary), encoding="utf-8")

    raw_path = RESULTS_DIR / f"raw_{variant}.json"
    raw_path.write_text(
        json.dumps(
            {
                "variant": variant,
                "judge_model": JUDGE_MODEL,
                "answer_model": settings.answer_model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "thresholds": THRESHOLDS,
                "summary": summary,
                "rows": [
                    {**row, "scores": scores} for row, scores in zip(answered, scored)
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return report_path, raw_path


def print_summary(variant, summary):
    status = verdicts(summary)
    failed = [name for name in METRIC_ORDER if status[name] == "FAIL"]

    print()
    print(f"  {'metric':<20} {'score':>7}  {'threshold':>9}  result")
    print(f"  {'-' * 20} {'-' * 7}  {'-' * 9}  ------")
    for name in METRIC_ORDER:
        print(
            f"  {name:<20} {fmt(summary[name]):>7}  "
            f"{THRESHOLDS[name]:>9.2f}  {status[name]}"
        )
    print()
    if failed:
        print(f"  FAIL ({variant}) — below threshold: {', '.join(failed)}")
    else:
        print(f"  PASS ({variant}) — every metric met its threshold.")
    print()

    return failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--variant",
        choices=["hardened", "naive"],
        default="hardened",
        help="hardened is the pipeline as it ships; naive turns the guardrails off",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="exit 1 if any metric is below its threshold",
    )
    args = parser.parse_args()

    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set. Add it to .env and try again.")
        return 1

    rows = load_rows()

    print(f"Answering {len(rows)} golden questions ({args.variant} pipeline)...")
    answered = generate_answers(rows, args.variant)

    print(f"Judging with {JUDGE_MODEL} ({MAX_CONCURRENT_ROWS} rows at a time)...")
    metrics = build_metrics(build_judge())
    scored = asyncio.run(score_all(answered, metrics))

    summary = summarise(scored)
    report_path, raw_path = write_outputs(args.variant, answered, scored, summary)

    failed = print_summary(args.variant, summary)
    print(f"  report {report_path.relative_to(REPO_ROOT)}")
    print(f"  raw    {raw_path.relative_to(REPO_ROOT)}")
    print()

    if args.ci and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

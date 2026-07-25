# Kovai Finserv RAG — Project Constitution

## WHAT THIS IS
A retrieval-augmented Q&A API over Kovai Finserv's loan policy documents
(a lending company in Coimbatore, India), with a separate Streamlit web UI.
Deployed on free cloud tiers.

## THE PRODUCT RULE (non-negotiable)
The bot must **NEVER** state a policy that is not in the retrieved documents.
A wrong number costs the company real money. When in doubt, refuse and hand
off to a human.

## STACK
- Python 3.12
- Claude via the Anthropic SDK. Model `claude-haiku-4-5-20251001` for answers
  AND for guardrail screening.
- ChromaDB with its default local ONNX embeddings (all-MiniLM-L6-v2).
  No second API key, no OpenAI.
- FastAPI + Uvicorn for the API
- Streamlit for the UI, in `ui/`, talking to the API over HTTP
- Ragas 0.4.3 for offline evaluation, with Claude as the judge

## LAYOUT
- `app/`     the API runtime only. Keep it small; it becomes the Docker image.
- `ui/`      the Streamlit frontend. Deployed separately. It must ONLY talk to
             the API over HTTP — it must never import from `app/`.
- `eval/`    Ragas. Never imported by `app/`. Never ships.
- `scripts/ingest.py` builds the Chroma index into `chroma_db/`
- `tests/`   pytest, offline, no API key required

## RULES FOR YOU, CLAUDE
1. Never add a dependency without telling me why. `requirements.txt` IS the
   Docker image and every megabyte matters on a 512 MB free instance.
2. Pin every dependency to an exact version.
3. Only `app/config.py` reads `os.environ`. Nothing else, ever.
4. Never log question or answer text at INFO level — it can contain customer
   PII. Log lengths, latencies, decisions and IDs.
5. Every guardrail needs a test proving it blocks the bad case AND lets a
   normal question through.
6. Answers must cite their source section. If retrieval finds nothing
   relevant, return the refusal string — never a guess.
7. Prefer boring, readable code. This is teaching material as well as
   production code.

## COMMANDS
```
python scripts/ingest.py
uvicorn app.main:app --reload --port 8000
streamlit run ui/streamlit_app.py
pytest -q
python eval/run_ragas.py
```

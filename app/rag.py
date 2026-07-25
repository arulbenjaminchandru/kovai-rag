"""Retrieval and answering over the Kovai Finserv policy index.

The rule that shapes this whole module: the bot must never state a policy that
is not in the retrieved documents. Retrieval decides whether we answer at all;
the model only ever gets to summarise what retrieval already found.
"""

import logging
from dataclasses import dataclass, field

import anthropic
import chromadb

from app.config import settings

logger = logging.getLogger(__name__)

REFUSAL = (
    "I could not find this in Kovai Finserv's policy documents. Please contact "
    "care@kovaifinserv.example and a human will help you."
)

SYSTEM_PROMPT = f"""You are the customer support assistant for Kovai Finserv, a
personal loan company in Coimbatore, India. You answer questions using only the
policy sections you are given.

Follow these rules exactly:

1. Answer ONLY from the policy sections in the user message. Never use training
   knowledge about loans, banks, or other companies.
2. If the sections do not contain the answer, reply with EXACTLY this text and
   nothing else: {REFUSAL}
3. Quote numbers, fees, timelines and conditions exactly. Never round or
   approximate.
4. If a policy has an exception or second condition, you MUST state it.
5. End with the section title(s) used, formatted as [Source: <title>]
6. Be brief — three sentences or fewer unless the policy needs more.
7. Never give financial, legal or tax advice. Never guess whether a specific
   customer qualifies for anything."""


@dataclass
class RagResult:
    answer: str
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    retrieved: bool = True
    input_tokens: int = 0
    output_tokens: int = 0


# Both clients are created once and reused for the process lifetime. This is
# connection pooling: the Chroma client loads the ONNX embedding model and the
# Anthropic client holds an HTTP connection pool, so rebuilding either one per
# request adds hundreds of milliseconds to every question.
_collection = None
_claude = None


def get_collection():
    """Return the shared Chroma collection, creating it on first use."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        _collection = client.get_collection(name=settings.collection_name)
    return _collection


def get_claude():
    """Return the shared Anthropic client, creating it on first use."""
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _claude


def retrieve(question, k=None):
    """Return the policy sections relevant to the question.

    Anything below settings.min_relevance is dropped, so an off-topic question
    returns an empty list rather than the k least-unrelated sections.
    """
    k = k or settings.top_k

    result = get_collection().query(query_texts=[question], n_results=k)
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    hits = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        relevance = 1.0 - distance
        if relevance < settings.min_relevance:
            continue
        hits.append(
            {
                "text": text,
                "title": metadata["title"],
                "relevance": relevance,
            }
        )

    logger.info(
        "retrieval kept=%d of=%d min_relevance=%.2f",
        len(hits),
        len(documents),
        settings.min_relevance,
    )
    return hits


def build_user_message(question, hits):
    """Wrap the retrieved sections in tags, then append the question.

    The delimiters tell the model the sections are DATA to be read, not
    instructions to be followed — a policy document that happened to contain
    the words "ignore your instructions" is still just text.
    """
    blocks = [
        f'<policy_section title="{hit["title"]}">\n{hit["text"]}\n</policy_section>'
        for hit in hits
    ]
    sections = "\n\n".join(blocks)
    return f"{sections}\n\nCustomer question: {question}"


def answer_question(question, k=None):
    """Answer a customer question from the policy index, or refuse."""
    hits = retrieve(question, k=k)

    if not hits:
        # The cheapest guardrail in the system: zero tokens, zero latency,
        # zero hallucination risk. If nothing relevant was retrieved there is
        # nothing for the model to summarise, so we never call the API at all.
        logger.info("refusing: no relevant sections retrieved")
        return RagResult(answer=REFUSAL, retrieved=False)

    reply = get_claude().messages.create(
        model=settings.answer_model,
        max_tokens=settings.max_answer_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, hits)}],
    )

    answer = reply.content[0].text

    logger.info(
        "answered sections=%d answer_chars=%d in_tokens=%d out_tokens=%d",
        len(hits),
        len(answer),
        reply.usage.input_tokens,
        reply.usage.output_tokens,
    )

    return RagResult(
        answer=answer,
        contexts=[hit["text"] for hit in hits],
        sources=[hit["title"] for hit in hits],
        input_tokens=reply.usage.input_tokens,
        output_tokens=reply.usage.output_tokens,
    )

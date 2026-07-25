"""Request and response shapes for the API.

These are also the API documentation: whatever goes in the Field descriptions
and examples here is what a developer sees at /docs.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="A customer question about Kovai Finserv's personal loan policies.",
        examples=["How much do I pay if I close my loan early?"],
    )


class AskResponse(BaseModel):
    answer: str = Field(description="The answer, the refusal, or a block message.")
    sources: list[str] = Field(
        default_factory=list,
        description="Titles of the policy sections the answer was drawn from.",
    )
    blocked: bool = Field(
        default=False,
        description="True when a guardrail stopped the question or the answer.",
    )
    reason: str = Field(
        default="OK",
        description="OK, NO_CONTEXT, TOO_LONG, INJECTION, OFF_TOPIC, UNGROUNDED "
        "or REFUSAL_PASSTHROUGH.",
    )
    request_id: str = Field(description="Echoed in the x-request-id header. Quote it in bug reports.")
    latency_ms: int = Field(description="Server-side wall time for this request.")
    model: str = Field(description="The model that produced the answer.")


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    indexed_chunks: int
    model: str

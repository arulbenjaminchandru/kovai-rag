"""All environment reading for the whole application happens here.

Nothing else in the codebase may touch os.environ. If you need a new knob,
add a field here and read it from settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""

    answer_model: str = "claude-haiku-4-5-20251001"
    guard_model: str = "claude-haiku-4-5-20251001"

    chroma_path: str = "chroma_db"
    collection_name: str = "kovai_policies"

    top_k: int = 4

    # Chroma is configured with cosine DISTANCE, so relevance = 1 - distance.
    # A relevance of 1.0 is a perfect match and 0.0 is unrelated.
    #
    # This threshold is a SAFETY control, not a tuning knob. A vector search
    # always returns k results, even when the index contains nothing relevant
    # to the question — ask about home loans and you will still get back the
    # four "least unrelated" personal-loan sections. Without this floor the
    # model would be handed plausible-looking but wrong context and would
    # answer from it. Dropping everything below the floor is what lets us
    # refuse instead of guess.
    min_relevance: float = 0.25

    max_answer_tokens: int = 500
    max_question_chars: int = 500

    rate_limit: str = "20/minute"

    log_level: str = "INFO"
    app_version: str = "1.0.0"


settings = Settings()

import logging
from typing import TypeVar

from langchain_core.messages import BaseMessage
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel

from backend.config import MISTRAL_API_KEY, MISTRAL_MODEL

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_llm: ChatMistralAI | None = None


def get_llm() -> ChatMistralAI:
    global _llm
    if _llm is None:
        if not MISTRAL_API_KEY:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _llm = ChatMistralAI(model=MISTRAL_MODEL, api_key=MISTRAL_API_KEY, temperature=0.2)
    return _llm


def call_structured(schema: type[T], messages: list[BaseMessage], retries: int = 1) -> T:
    """Invoke the LLM with structured output, retrying once on a validation failure.

    Callers are responsible for handling the case where every attempt fails
    (they should keep existing state untouched and ask the recruiter to rephrase).
    """
    structured_llm = get_llm().with_structured_output(schema)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = structured_llm.invoke(messages)
            if isinstance(result, schema):
                return result
            return schema.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - structured-output failures are heterogeneous
            last_error = exc
            logger.warning("Structured output attempt %s failed: %s", attempt + 1, exc)
    assert last_error is not None
    raise last_error

import logging
import time
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from backend.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    LLM_PROVIDER,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """Provider is chosen by LLM_PROVIDER ("deepseek" or "mistral") — everything downstream
    (call_structured, every node that calls it) only depends on the standard LangChain
    BaseChatModel + with_structured_output() interface, so swapping providers never needs to
    touch any calling code, only this one function.
    """
    global _llm
    if _llm is None:
        if LLM_PROVIDER == "mistral":
            from langchain_mistralai import ChatMistralAI

            if not MISTRAL_API_KEY:
                raise RuntimeError(
                    "MISTRAL_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            _llm = ChatMistralAI(model=MISTRAL_MODEL, api_key=MISTRAL_API_KEY, temperature=0.2)
        else:
            from langchain_deepseek import ChatDeepSeek

            if not DEEPSEEK_API_KEY:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            _llm = ChatDeepSeek(model=DEEPSEEK_MODEL, api_key=DEEPSEEK_API_KEY, temperature=0.2)
    return _llm


def _is_rate_limited(exc: Exception) -> bool:
    return "429" in str(exc) or "rate_limited" in str(exc).lower()


def call_structured(schema: type[T], messages: list[BaseMessage], retries: int = 1) -> T:
    """Invoke the LLM with structured output, retrying on failure.

    A 429 gets a short backoff before the next attempt. Mistral's per-second rate-limit window
    clears fast, but retrying instantly (the old behavior) was guaranteed to fail again on a
    sustained burst and surface a confusing "didn't catch that" fallback to the recruiter for
    what was really just a transient spike — a couple seconds of backoff turns a real user's
    occasional 429 into a slightly slower reply instead of a dropped message.

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
            if attempt < retries and _is_rate_limited(exc):
                time.sleep(2.5 * (attempt + 1))
    assert last_error is not None
    raise last_error

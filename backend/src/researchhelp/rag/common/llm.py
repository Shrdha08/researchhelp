"""LLM factory. Generation uses Groq-hosted open models (free tier).

``max_retries`` makes the Groq client back off and retry on rate-limit (HTTP 429) and transient
errors, which the free tier hits quickly during evaluation runs.
"""

from langchain_core.language_models import BaseChatModel

from researchhelp.config import get_settings


class MissingAPIKeyError(RuntimeError):
    pass


def get_chat_model(model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    settings = get_settings()
    if not settings.groq_api_key:
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(https://console.groq.com/keys)."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_retries=settings.llm_max_retries,
        api_key=settings.groq_api_key,
    )

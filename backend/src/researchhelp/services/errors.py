"""Service-level errors. The API layer maps them to HTTP status codes; services never raise
HTTP exceptions themselves, so they stay usable from the CLI and tests."""


class ServiceError(Exception):
    pass


class InvalidUpload(ServiceError):  # 400
    pass


class PaperNotFound(ServiceError):  # 404
    pass


class PaperNotReady(ServiceError):  # 409: still processing, or ingestion failed
    pass


class PaperBusy(ServiceError):  # 409: cannot delete while ingestion is running
    pass


class LLMUnavailable(ServiceError):  # 503: e.g. GROQ_API_KEY missing
    pass


class ConversationNotFound(ServiceError):  # 404
    pass

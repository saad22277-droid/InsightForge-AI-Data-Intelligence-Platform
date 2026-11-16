class AIServiceError(Exception):
    """A failure of the AI layer, classified so the API can respond usefully.

    ``kind`` is one of: not_configured, auth, rate_limited, quota, timeout,
    model_unavailable, blocked, bad_request, server, busy, empty, unknown.
    """

    STATUS = {
        "not_configured": 503,
        "auth": 502,
        "rate_limited": 429,
        "quota": 429,
        "timeout": 504,
        "model_unavailable": 502,
        "blocked": 422,
        "bad_request": 502,
        "server": 502,
        "busy": 429,
        "empty": 502,
        "unknown": 502,
    }

    def __init__(self, kind: str, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retry_after = retry_after

    @property
    def status_code(self) -> int:
        return self.STATUS.get(self.kind, 502)

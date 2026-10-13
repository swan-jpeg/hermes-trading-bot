"""Redaction of secret-like fields before persistence."""
from __future__ import annotations

import re
from typing import Any

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|token|secret|password|credential|private[_-]?key|auth[_-]?header|"
    r"access[_-]?token|refresh[_-]?token|bearer|apikey|passwd|pwd)",
    re.IGNORECASE,
)

_SECRET_VALUE_RE = re.compile(
    r"^(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{30,}|\\b[A-Za-z0-9_]{32,}\\b)$"
)

_REDACTED = "__REDACTED__"


class SecretError(ValueError):
    """Raised when a candidate payload contains a non-redacted secret."""


_SECRET_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "token",
    "secret",
    "secret_key",
    "secret-key",
    "password",
    "credential",
    "credentials",
    "private_key",
    "private-key",
    "auth_header",
    "auth-header",
    "access_token",
    "access-token",
    "refresh_token",
    "refresh-token",
    "bearer",
    "passwd",
    "pwd",
}


def _looks_like_secret_key(key: str) -> bool:
    """Detect keys that commonly carry credentials."""
    normalized = key.lower().replace(" ", "_").replace("-", "_")
    return normalized in _SECRET_KEYS or _SECRET_KEY_RE.search(key) is not None


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _REDACTED if _looks_like_secret_key(k) else _redact_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(v) for v in value]
    return value


def redact_payload(data: Any) -> dict[str, Any]:
    """Return a deep copy with secret-like values replaced by ``__REDACTED__``.

    Strings that look like high-entropy tokens are also redacted in-place.
    """
    if not isinstance(data, dict):
        data = {"value": data}
    redacted = _redact_value(data)
    if isinstance(redacted, dict):
        return redacted
    return {"value": redacted}


def reject_or_redact(data: Any) -> dict[str, Any]:
    """Redact payload and raise ``SecretError`` if a secret value was present."""
    redacted = redact_payload(data)
    _assert_no_secrets(redacted)
    return redacted


def _assert_no_secrets(data: Any) -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            if _looks_like_secret_key(k) and v != _REDACTED:
                raise SecretError(f"secret-like key '{k}' is not redacted: {type(v).__name__}")
            _assert_no_secrets(v)
    elif isinstance(data, list):
        for item in data:
            _assert_no_secrets(item)

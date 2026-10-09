import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

ConnectionStatus = Literal[
    "connected",
    "missing_credentials",
    "network_or_firewall_error",
    "invalid_credentials",
    "permission_or_project_error",
    "quota_or_rate_limit_error",
    "model_not_available",
    "billing_or_prepayment_required",
    "api_error",
]


@dataclass(frozen=True)
class GeminiConnectionResult:
    status: ConnectionStatus
    model: str
    detail: str
    http_status: int | None = None

    @property
    def connected(self) -> bool:
        return self.status == "connected"


def _resolve_api_key(api_key: str | None = None) -> str:
    load_gemini_config()
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Set GEMINI_API_KEY in the process environment before creating a Gemini client."
        )
    return resolved_key


def load_gemini_config() -> dict[str, str | bool]:
    """Load repo-root .env without overriding process variables; never return the key."""
    load_dotenv(dotenv_path=DEFAULT_ENV_FILE, override=False)
    return {
        "api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
        "model": os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    }


def _resolve_model(model: str | None) -> str:
    config = load_gemini_config()
    return model or str(config["model"])


def create_gemini_openai_client(api_key: str | None = None) -> OpenAI:
    """Create an OpenAI SDK client configured for the Gemini Developer API."""
    return OpenAI(
        api_key=_resolve_api_key(api_key),
        base_url=GEMINI_OPENAI_BASE_URL,
    )


def create_gemini_chat_model(
    model: str | None = None,
    api_key: str | None = None,
    **model_options: Any,
) -> ChatOpenAI:
    """Create a LangChain chat model usable by the Task 2 prompt chains."""
    return ChatOpenAI(
        model=_resolve_model(model),
        api_key=_resolve_api_key(api_key),
        base_url=GEMINI_OPENAI_BASE_URL,
        **model_options,
    )


def _transport_detail(error: APIConnectionError, api_key: str) -> str:
    cause = error.__cause__ or error
    detail = str(cause).replace(api_key, "[redacted]").splitlines()[0]
    normalized = detail.lower()

    if "certificate_verify_failed" in normalized or "ssl" in normalized:
        return f"TLS/certificate failure; check company TLS inspection and trusted CA configuration. {detail}"
    if "proxy" in normalized:
        return f"Proxy connection failure; check HTTPS_PROXY and company proxy settings. {detail}"
    if any(token in normalized for token in ("timed out", "timeout", "name or service", "getaddrinfo", "connecterror")):
        return f"Network, DNS, firewall, or proxy may be blocking Google API access. {detail}"
    return f"Could not establish the API connection. Check network/firewall/proxy settings. {detail}"


def test_gemini_connection(
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> GeminiConnectionResult:
    """Send one small request and classify auth, access, model, quota, or transport failures."""
    resolved_model = _resolve_model(model)
    try:
        resolved_key = _resolve_api_key(api_key)
    except ValueError as error:
        return GeminiConnectionResult(
            status="missing_credentials",
            model=resolved_model,
            detail=str(error),
        )

    try:
        client = create_gemini_openai_client(resolved_key)
        client.with_options(timeout=timeout).chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=4,
            temperature=0,
        )
        return GeminiConnectionResult(
            status="connected",
            model=resolved_model,
            detail="Gemini returned a chat completion successfully.",
        )
    except AuthenticationError as error:
        return GeminiConnectionResult(
            status="invalid_credentials",
            model=resolved_model,
            detail=f"Gemini rejected the API key (HTTP 401). Create/restrict a valid Gemini API key. {error.message}",
            http_status=401,
        )
    except PermissionDeniedError as error:
        return GeminiConnectionResult(
            status="permission_or_project_error",
            model=resolved_model,
            detail=f"The key/project lacks permission for this API or model (HTTP 403). Check project/API access. {error.message}",
            http_status=403,
        )
    except RateLimitError as error:
        return GeminiConnectionResult(
            status="quota_or_rate_limit_error",
            model=resolved_model,
            detail=f"The API was reached, but quota/rate limits blocked the request (HTTP 429). {error.message}",
            http_status=429,
        )
    except NotFoundError as error:
        return GeminiConnectionResult(
            status="model_not_available",
            model=resolved_model,
            detail=f"The API was reached, but the model was not found or is unavailable to this project (HTTP 404). Check the model name/access. {error.message}",
            http_status=404,
        )
    except APIConnectionError as error:
        return GeminiConnectionResult(
            status="network_or_firewall_error",
            model=resolved_model,
            detail=_transport_detail(error, resolved_key),
        )
    except APIStatusError as error:
        if error.status_code == 402:
            return GeminiConnectionResult(
                status="billing_or_prepayment_required",
                model=resolved_model,
                detail=(
                    "Gemini was reached, but the project's prepayment credits are depleted. "
                    "Add or replenish Gemini API credits in AI Studio billing; this is not "
                    "a firewall or invalid-key error. "
                    f"{error.message}"
                ),
                http_status=402,
            )
        return GeminiConnectionResult(
            status="api_error",
            model=resolved_model,
            detail=f"Gemini returned HTTP {error.status_code}: {error.message}",
            http_status=error.status_code,
        )
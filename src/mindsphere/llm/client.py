"""
HTTP wrapper for OpenAI-compatible /v1/chat/completions endpoints.

Supports any provider (Groq, Mistral, Together, etc.) with automatic
fallback: when the primary provider returns 429, tries the fallback.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class MistralAPIError(Exception):
    """Raised when the LLM API returns an error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Mistral API error {status_code}: {message}")


@dataclass
class MistralClient:
    """
    HTTP wrapper for OpenAI-compatible /v1/chat/completions endpoints.

    Supports automatic fallback: if the primary provider is rate-limited (429),
    the request is retried on the fallback provider.

    Configure via environment variables:
        LLM_API_KEY / GROQ_API_KEY / MISTRAL_API_KEY — API key
        LLM_BASE_URL — API base URL (default: Mistral)
        LLM_MODEL — Default model name
    """

    api_key: str = ""
    model: str = ""
    base_url: str = ""
    timeout: float = 30.0
    max_retries: int = 2
    # Fallback provider (auto-configured from env)
    _fallback: Optional[Tuple[str, str, str]] = field(default=None, repr=False)

    def __post_init__(self):
        # Load .env file into os.environ so all config is accessible
        self._load_dotenv()
        if not self.base_url:
            self.base_url = os.environ.get(
                "LLM_BASE_URL", "https://api.mistral.ai/v1"
            ).strip().rstrip("/")
        if not self.model:
            self.model = os.environ.get("LLM_MODEL", "mistral-small-latest").strip()
        if not self.api_key:
            self.api_key = self._load_api_key()
        # Set up fallback provider
        self._fallback = self._load_fallback()

    def _load_dotenv(self) -> None:
        """Load .env file into os.environ (only vars not already set)."""
        for parent in [Path.cwd()] + list(Path(__file__).resolve().parents):
            env_path = parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key, value = key.strip(), value.strip()
                        if key and key not in os.environ:
                            os.environ[key] = value
                break  # only load the first .env found

    def _load_api_key(self) -> str:
        """Load API key from environment variable."""
        for env_var in ("LLM_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
            key = os.environ.get(env_var, "").strip()
            if key:
                return key

        raise MistralAPIError(401, "No LLM_API_KEY, GROQ_API_KEY, or MISTRAL_API_KEY found in env or .env file")

    def _load_fallback(self) -> Optional[Tuple[str, str, str]]:
        """Load fallback provider if a second API key is available."""
        # If primary is Groq, fallback to Mistral (and vice versa)
        is_groq = "groq.com" in self.base_url
        is_mistral = "mistral.ai" in self.base_url

        if is_groq:
            fallback_key = os.environ.get("MISTRAL_API_KEY", "").strip()
            if fallback_key and fallback_key != self.api_key:
                logger.info("[LLMClient] Fallback provider: Mistral")
                return ("https://api.mistral.ai/v1", "mistral-small-latest", fallback_key)
        elif is_mistral:
            fallback_key = os.environ.get("GROQ_API_KEY", "").strip()
            if fallback_key and fallback_key != self.api_key:
                logger.info("[LLMClient] Fallback provider: Groq")
                return ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant", fallback_key)

        return None

    def _headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key or self.api_key}",
            "Content-Type": "application/json",
        }

    def _do_request(
        self,
        url: str,
        body: Dict[str, Any],
        api_key: Optional[str] = None,
    ) -> str:
        """Make a single chat completion request. Returns content or raises."""
        resp = requests.post(
            url,
            headers=self._headers(api_key),
            json=body,
            timeout=self.timeout,
        )

        if resp.status_code == 200:
            data = resp.json()
            message = data["choices"][0]["message"]
            if message.get("content"):
                return message["content"]
            return message.get("content", "")

        if resp.status_code in (400, 401, 403, 404):
            raise MistralAPIError(resp.status_code, resp.text)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "?")
            raise MistralAPIError(429, f"Rate limited (Retry-After: {retry_after}s)")

        raise MistralAPIError(resp.status_code, resp.text)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Call /v1/chat/completions on the primary provider.
        On 429, automatically tries the fallback provider if configured.

        Returns the assistant's response content as a string.
        Raises MistralAPIError on failure.
        """
        body: Dict[str, Any] = {
            "model": model_override or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if tools is not None:
            body["tools"] = tools

        url = f"{self.base_url}/chat/completions"

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._do_request(url, body)
            except MistralAPIError as e:
                if e.status_code == 429 and self._fallback:
                    # Try fallback provider
                    fb_url, fb_model, fb_key = self._fallback
                    fb_body = {**body, "model": fb_model}
                    logger.info(f"[LLMClient] Primary rate-limited, trying fallback ({fb_url})")
                    try:
                        return self._do_request(
                            f"{fb_url}/chat/completions", fb_body, api_key=fb_key
                        )
                    except MistralAPIError as fb_e:
                        logger.warning(f"[LLMClient] Fallback also failed: {fb_e}")
                        last_error = e  # report primary error
                        break
                elif e.status_code == 429:
                    logger.warning(f"[LLMClient] Rate limited (429), no fallback — failing fast")
                    raise
                else:
                    last_error = e
            except requests.exceptions.Timeout:
                logger.warning(f"[LLMClient] Request timed out (attempt {attempt + 1}/{self.max_retries + 1})")
                last_error = MistralAPIError(408, "Request timed out")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[LLMClient] Connection error: {e}")
                last_error = MistralAPIError(0, f"Connection error: {e}")

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        raise last_error  # type: ignore

    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 300,
        model_override: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream chat completion, yielding text chunks as they arrive.
        On 429, automatically tries the fallback provider if configured.
        """
        url = f"{self.base_url}/chat/completions"
        body: Dict[str, Any] = {
            "model": model_override or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                json=body,
                timeout=self.timeout,
                stream=True,
            )

            # On 429, try fallback for streaming too
            if resp.status_code == 429 and self._fallback:
                fb_url, fb_model, fb_key = self._fallback
                fb_body = {**body, "model": fb_model}
                logger.info(f"[LLMClient] Stream rate-limited, trying fallback ({fb_url})")
                resp = requests.post(
                    f"{fb_url}/chat/completions",
                    headers=self._headers(fb_key),
                    json=fb_body,
                    timeout=self.timeout,
                    stream=True,
                )

            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        except Exception as e:
            logger.warning(f"[LLMClient] Streaming error: {e}")

    @property
    def is_available(self) -> bool:
        """Check if the client has a valid API key configured."""
        try:
            return bool(self.api_key)
        except MistralAPIError:
            return False

"""
Thin HTTP wrapper for Mistral's /v1/chat/completions endpoint.

Copied from NEXT-prototype with zero modifications.
Handles authentication, retries, and response parsing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class MistralAPIError(Exception):
    """Raised when the Mistral API returns an error."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Mistral API error {status_code}: {message}")


@dataclass
class MistralClient:
    """
    Thin HTTP wrapper for Mistral's /v1/chat/completions endpoint.

    Handles authentication, retries, and response parsing.
    The classifier and generator use this as their backend.
    """

    api_key: str = ""
    model: str = "mistral-small-latest"
    base_url: str = "https://api.mistral.ai/v1"
    timeout: float = 30.0
    max_retries: int = 2

    def __post_init__(self):
        if not self.api_key:
            self.api_key = self._load_api_key()

    def _load_api_key(self) -> str:
        """Load API key from environment variable or .env file."""
        key = os.environ.get("MISTRAL_API_KEY", "")
        if key:
            return key

        for parent in [Path.cwd()] + list(Path(__file__).resolve().parents):
            env_path = parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("MISTRAL_API_KEY=") and not line.startswith("#"):
                        return line.split("=", 1)[1].strip()

        raise MistralAPIError(401, "No MISTRAL_API_KEY found in env or .env file")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
        Call Mistral /v1/chat/completions.

        Returns the assistant's response content as a string.
        When tools are used (e.g. web_search), the response includes
        tool results inlined by the API.

        Raises MistralAPIError on failure.
        """
        url = f"{self.base_url}/chat/completions"

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

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=body,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    message = data["choices"][0]["message"]
                    # Standard text response
                    if message.get("content"):
                        return message["content"]
                    # If the model used tools, the content may be empty
                    # but tool_calls will have results — return what we have
                    return message.get("content", "")

                if resp.status_code in (400, 401, 403, 404):
                    raise MistralAPIError(resp.status_code, resp.text)

                last_error = MistralAPIError(resp.status_code, resp.text)

            except requests.exceptions.Timeout:
                last_error = MistralAPIError(408, "Request timed out")
            except requests.exceptions.ConnectionError as e:
                last_error = MistralAPIError(0, f"Connection error: {e}")

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        raise last_error  # type: ignore

    @property
    def is_available(self) -> bool:
        """Check if the client has a valid API key configured."""
        try:
            return bool(self.api_key)
        except MistralAPIError:
            return False

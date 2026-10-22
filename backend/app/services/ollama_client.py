"""Minimal Ollama HTTP client (local only, no cloud fallback)."""
import logging
from typing import Optional

import httpx

from app.core.errors import AppError

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 180.0,
        health_timeout_seconds: float = 3.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.health_timeout_seconds = health_timeout_seconds
        self._transport = transport

    def _client(self, timeout) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=timeout, transport=self._transport)

    def _has_model(self, names: list[str]) -> bool:
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return wanted in names or self.model in names

    def status(self) -> dict:
        """Never raises. Reports whether Ollama is reachable and the model is downloaded."""
        info = {"available": False, "model": self.model, "model_available": False, "message": ""}
        try:
            with self._client(self.health_timeout_seconds) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
        except httpx.HTTPError:
            info["message"] = (
                f"Ollama is not reachable at {self.base_url}. Install it from https://ollama.com "
                "and start it (run `ollama serve` if it is not already running)."
            )
            return info
        except ValueError:
            info["message"] = f"Unexpected response from Ollama at {self.base_url}."
            return info

        names = [m.get("name") or m.get("model") or "" for m in models]
        info["available"] = True
        info["model_available"] = self._has_model(names)
        info["message"] = (
            "Ollama is ready."
            if info["model_available"]
            else f"Model '{self.model}' is not downloaded. Run: ollama pull {self.model}"
        )
        return info

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        status = self.status()
        if not status["available"]:
            raise AppError(503, "ollama_unavailable", status["message"])
        if not status["model_available"]:
            raise AppError(503, "model_missing", status["message"])

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.2, "num_ctx": 4096},
        }
        timeout = httpx.Timeout(self.timeout_seconds, connect=5.0)
        try:
            with self._client(timeout) as client:
                response = client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise AppError(
                504,
                "ollama_timeout",
                f"The model took longer than {int(self.timeout_seconds)} seconds to answer. "
                "Try again, or increase OLLAMA_TIMEOUT_SECONDS.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama returned %s: %s", exc.response.status_code, exc.response.text[:300])
            raise AppError(502, "generation_failed", "Ollama returned an error while generating the answer.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise AppError(502, "generation_failed", "Could not get an answer from Ollama.") from exc

        answer = (data.get("message") or {}).get("content", "").strip()
        if not answer:
            raise AppError(502, "generation_failed", "The model returned an empty answer.")
        return answer

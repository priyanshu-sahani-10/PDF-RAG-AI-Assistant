import httpx
import pytest

from app.core.errors import AppError
from app.services.ollama_client import OllamaClient


def make_client(handler, model="gemma:2b"):
    return OllamaClient("http://ollama.test", model, timeout_seconds=5, transport=httpx.MockTransport(handler))


def test_not_running():
    def handler(request):
        raise httpx.ConnectError("refused")

    status = make_client(handler).status()
    assert status["available"] is False
    assert "ollama serve" in status["message"]


def test_model_missing():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    client = make_client(handler)
    assert client.status()["model_available"] is False
    with pytest.raises(AppError) as exc:
        client.generate("sys", "hi")
    assert exc.value.code == "model_missing"
    assert "ollama pull gemma:2b" in exc.value.message


def test_generate_success():
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma:2b"}]})
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " Hello "}})

    client = make_client(handler)
    assert client.status()["model_available"] is True
    assert client.generate("sys", "hi") == "Hello"


def test_generate_timeout():
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "gemma:2b"}]})
        raise httpx.ReadTimeout("slow")

    with pytest.raises(AppError) as exc:
        make_client(handler).generate("sys", "hi")
    assert exc.value.code == "ollama_timeout"

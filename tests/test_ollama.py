import json

import httpx
import pytest

from freezer_rag.config import Settings
from freezer_rag.embeddings.ollama import Embeddings
from freezer_rag.models import RagError
from freezer_rag.ollama import Ollama


@pytest.mark.parametrize(
    "model,dimension,prefix",
    [
        ("nomic-embed-text:v1.5", 768, "search_query: "),
        ("qwen3-embedding:0.6b", 1024, "Instruct: "),
    ],
)
def test_embedding_adapter_formats_normalizes_and_rejects_bad_vectors(model, dimension, prefix):
    settings = Settings(_env_file=None, embedding_model=model)
    bad = False
    requests = []

    def respond(request):
        nonlocal bad
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": model, "digest": "abc"}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"test.context_length": 2048}})
        payload = json.loads(request.content)
        requests.append(payload)
        vector = [float("nan")] if bad else [2.0] + [0.0] * (dimension - 1)
        return (
            httpx.Response(200, json={"embeddings": [vector]})
            if not bad
            else httpx.Response(
                200, content='{"embeddings":[[0.1]]}', headers={"content-type": "application/json"}
            )
        )

    with httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(respond)
    ) as client:
        adapter = Embeddings(Ollama(settings, client))
        vector = adapter.embed(["When does a reservation expire?"], query=True)[0]
        assert vector[0] == 1.0
        assert requests[-1]["input"][0].startswith(prefix)
        assert requests[-1]["truncate"] is False
        assert requests[-1]["keep_alive"] == "0"
        bad = True
        with pytest.raises(RagError, match="dimension mismatch"):
            adapter.embed(["query"], query=True)


def test_embedding_retries_only_transient_errors_and_reports_missing_model(monkeypatch):
    settings = Settings(_env_file=None)
    attempts = []
    monkeypatch.setattr("freezer_rag.ollama.time.sleep", lambda seconds: None)

    def respond(request):
        attempts.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(503 if len(attempts) == 1 else 400, json={"error": "bad input"})

    with httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(respond)
    ) as client:
        ollama = Ollama(settings, client)
        with pytest.raises(RagError, match="bad input"):
            ollama.request("POST", "/api/embed", {}, retries=2)
        assert attempts == ["/api/embed", "/api/embed"]
        with pytest.raises(RagError, match="ollama pull"):
            ollama.model(settings.embedding_model)


def test_base_model_environment_override_drives_generation_and_rejects_embedding_model(monkeypatch):
    monkeypatch.setenv("RAG_LLM_MODEL", "different-local-model:4b")
    settings = Settings(_env_file=None)
    requests = []
    capabilities = ["completion"]

    def respond(request):
        payload = json.loads(request.content) if request.content else {}
        requests.append((request.url.path, payload))
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": settings.llm_model, "digest": "different-digest"}]}
            )
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": capabilities})
        return httpx.Response(
            200, json={"done": True, "done_reason": "stop", "message": {"content": '{"ok":true}'}}
        )

    with httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(respond)
    ) as client:
        ollama = Ollama(settings, client)
        assert ollama.chat([{"role": "user", "content": "answer"}], {}) == '{"ok":true}'
        assert requests[-1][1]["model"] == "different-local-model:4b"
        assert "think" not in requests[-1][1]
        ollama._info.clear()
        capabilities[:] = ["embedding"]
        with pytest.raises(RagError, match="cannot generate chat"):
            ollama.chat([], {})


def test_transport_timeout_and_truncated_generation_are_recoverable():
    settings = Settings(_env_file=None)

    def timeout(request):
        raise httpx.ReadTimeout("local request timed out", request=request)

    with httpx.Client(
        base_url="http://ollama.test", transport=httpx.MockTransport(timeout)
    ) as client:
        with pytest.raises(RagError, match="timeout"):
            Ollama(settings, client).request("POST", "/api/chat", {})

    with httpx.Client(
        base_url="http://ollama.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"done": True, "done_reason": "length", "message": {"content": "partial"}}
            )
        ),
    ) as client:
        ollama = Ollama(settings, client)
        ollama.check_chat_model = lambda: {"capabilities": ["completion"]}
        with pytest.raises(RagError, match="budget|incomplete|truncated"):
            ollama.chat([], {})

"""Small Ollama HTTP transport; no cloud fallback or implicit model downloads."""

import time

import httpx

from freezer_rag.config import Settings
from freezer_rag.models import RagError


class Ollama:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(
            base_url=str(settings.ollama_base_url).rstrip("/"),
            timeout=settings.request_timeout_seconds,
            trust_env=False,
        )
        self._models: dict[str, dict] | None = None
        self._info: dict[str, dict] = {}

    def close(self) -> None:
        self.client.close()

    def request(
        self, method: str, path: str, payload: dict | None = None, retries: int = 0
    ) -> dict:
        for attempt in range(retries + 1):
            try:
                response = self.client.request(method, path, json=payload)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(0.25 * (2**attempt))
                    continue
                if response.is_error:
                    try:
                        detail = str(response.json().get("error", response.reason_phrase))
                    except ValueError:
                        detail = response.reason_phrase
                    raise RagError(f"Ollama {path}: {detail[:500]}")
                result = response.json()
                if not isinstance(result, dict) or result.get("error"):
                    raise RagError(f"Ollama returned an invalid response for {path}")
                return result
            except (httpx.TransportError, ValueError) as exc:
                if attempt < retries:
                    time.sleep(0.25 * (2**attempt))
                    continue
                raise RagError(
                    f"Cannot complete Ollama {path}: {exc}. Check Ollama is running "
                    "and the request timeout is sufficient."
                ) from exc
        raise AssertionError("Unreachable retry state")

    def model(self, name: str) -> dict:
        if self._models is None:
            models = self.request("GET", "/api/tags").get("models", [])
            if not isinstance(models, list) or any(
                not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
                for entry in models
            ):
                raise RagError("Ollama returned a malformed installed-model list")
            self._models = {entry["name"]: entry for entry in models if "name" in entry}
        actual = name if ":" in name else f"{name}:latest"
        if actual not in self._models:
            raise RagError(f"Missing Ollama model {name}. Install it with: ollama pull {name}")
        entry = self._models[actual]
        if not entry.get("digest"):
            raise RagError(f"Ollama did not return an installed digest for {name}")
        return entry

    def show(self, name: str) -> dict:
        self.model(name)
        if name not in self._info:
            self._info[name] = self.request("POST", "/api/show", {"model": name})
        return self._info[name]

    def check_chat_model(self) -> dict:
        info = self.show(self.settings.llm_model)
        capabilities = info.get("capabilities", [])
        if capabilities and "completion" not in capabilities:
            raise RagError(
                f"{self.settings.llm_model} cannot generate chat answers. "
                "Set RAG_LLM_MODEL to an installed chat model."
            )
        return info

    def chat(self, messages: list[dict], schema: dict) -> str:
        settings = self.settings
        info = self.check_chat_model()
        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "keep_alive": settings.llm_keep_alive,
            "options": {
                "num_ctx": settings.llm_num_ctx,
                "num_predict": settings.llm_num_predict,
                "temperature": settings.llm_temperature,
            },
        }
        if "thinking" in info.get("capabilities", []):
            payload["think"] = False
        response = self.request("POST", "/api/chat", payload)
        if response.get("done") is not True or response.get("done_reason") == "length":
            raise RagError("Generation was incomplete. Increase RAG_LLM_NUM_PREDICT and retry.")
        if response.get("prompt_eval_count", 0) >= settings.llm_num_ctx:
            raise RagError(
                "Prompt reached the context limit; reduce evidence/history or increase context."
            )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RagError("Ollama returned an empty answer")
        return content

    def unload_embeddings(self) -> None:
        self.request(
            "POST",
            "/api/embed",
            {
                "model": self.settings.embedding_model,
                "input": [],
                "keep_alive": 0,
            },
        )

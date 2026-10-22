"""Optional UI's meaningful boundary: reruns, recovery, and private conversations."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from freezer_rag.models import Answer, RagError


def test_ui_reruns_do_not_regenerate_and_sessions_clear_or_change_models_independently(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit import cache_resource
    from streamlit.testing.v1 import AppTest

    from freezer_rag import ui_runtime

    calls, opened = [], []

    class Shared:
        def __init__(self, key):
            opened.append(key)
            self.store = SimpleNamespace(chunks=[1, 2])

        def session(self, settings):
            class Session:
                def __init__(self):
                    self.history = []

                def clear(self):
                    self.history.clear()

                def ask(self, question):
                    calls.append((settings.llm_model, question, list(self.history)))
                    if question == "fail":
                        raise RagError("Ollama unavailable; retry")
                    self.history.append(question)
                    return Answer(text=f"Document answer for {question}")

            return Session()

    monkeypatch.setattr(ui_runtime, "SharedIndex", Shared)
    monkeypatch.setattr(sys, "argv", ["ui_app.py", "--no-env", "ui"])
    cache_resource.clear()
    path = Path(__file__).parents[1] / "src/freezer_rag/ui_app.py"
    try:
        first = AppTest.from_file(str(path), default_timeout=10)
        first.run()
        assert not first.exception
        first.chat_input[0].set_value("first question").run()
        first.run()
        assert len(calls) == 1 and len(first.chat_message) == 2
        second = AppTest.from_file(str(path), default_timeout=10)
        second.run()
        assert not second.chat_message and len(opened) == 1
        first.chat_input[0].set_value("fail").run()
        first.run()
        assert len(calls) == 2 and first.error[0].value == "Ollama unavailable; retry"
        first.chat_input[0].set_value("retry question").run()
        assert calls[-1][2] == ["first question"]
        first.button(key="clear").click().run()
        assert not first.chat_message
        first.text_input(key="model").set_value("another-chat:latest")
        first.button(key="apply").click().run()
        first.chat_input[0].set_value("new model question").run()
        assert calls[-1] == ("another-chat:latest", "new model question", [])
        assert len(opened) == 1 and not first.exception and not second.exception
    finally:
        cache_resource.clear()

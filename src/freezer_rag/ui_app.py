"""Optional Streamlit view. Launch with freezer-rag ui from the repository root."""

import sys

import streamlit as st
from pydantic import ValidationError

from freezer_rag.cli import parser
from freezer_rag.config import Settings
from freezer_rag.models import RagError
from freezer_rag.ui_runtime import SharedIndex, index_key


@st.cache_resource(show_spinner=False)
def get_index(key: str) -> SharedIndex:
    return SharedIndex(key)


def load_settings() -> Settings:
    args = parser().parse_args(sys.argv[1:])
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in Settings.model_fields and value is not None
    }
    return Settings(_env_file=None if args.no_env else ".env", **overrides)


def show_answer(answer) -> None:
    st.text(answer.text)
    if answer.sources:
        with st.expander(f"sources> {len(answer.sources)} passages"):
            for label, chunk in answer.sources.items():
                st.caption(f"[{label}] {chunk.source} | {chunk.locator}")
                st.text(chunk.text)


def main() -> None:
    st.set_page_config(
        page_title="Freezer docs | terminal", page_icon=":material/terminal:", layout="centered"
    )
    st.title("freezer / docs")
    st.caption("LOCAL DOCUMENT CHAT    /    OLLAMA")
    try:
        if "base_settings" not in st.session_state:
            st.session_state.base_settings = load_settings()
        base = st.session_state.base_settings
        with st.sidebar:
            st.subheader("session> settings")
            with st.form("settings_form"):
                model = st.text_input("Answer model", value=base.llm_model, key="model")
                temperature = st.number_input(
                    "Temperature", 0.0, 2.0, value=base.llm_temperature, step=0.05
                )
                dense = st.number_input("Dense top-k", min_value=1, value=base.dense_top_k)
                lexical = st.number_input("Lexical top-k", min_value=1, value=base.lexical_top_k)
                final = st.number_input("Final top-k", min_value=1, value=base.final_top_k)
                context = st.number_input("Context tokens", min_value=1024, value=base.llm_num_ctx)
                output = st.number_input("Output tokens", min_value=1, value=base.llm_num_predict)
                evidence = st.number_input(
                    "Evidence characters", min_value=1, value=base.max_context_chars
                )
                apply = st.form_submit_button("Apply and clear", key="apply")
            st.caption(
                "Changes affect this session. Embedding model and index follow startup settings."
            )
            clear = st.button("Clear conversation", key="clear")
        if apply or "settings" not in st.session_state:
            try:
                selected = Settings(
                    _env_file=None,
                    **{
                        **base.model_dump(),
                        "llm_model": model,
                        "llm_temperature": temperature,
                        "dense_top_k": dense,
                        "lexical_top_k": lexical,
                        "final_top_k": final,
                        "llm_num_ctx": context,
                        "llm_num_predict": output,
                        "max_context_chars": evidence,
                    },
                )
                st.session_state.settings = selected
                st.session_state.pop("conversation", None)
                st.session_state.messages = []
            except ValidationError as exc:
                st.error("; ".join(error["msg"] for error in exc.errors(include_input=False)))
        settings = st.session_state.settings
        with st.spinner("index> opening local documents..."):
            shared = get_index(index_key(base))
        if "conversation" not in st.session_state:
            st.session_state.conversation = shared.session(settings)
        conversation = st.session_state.conversation
        if clear:
            conversation.clear()
            st.session_state.messages = []
        st.caption(f"model> {settings.llm_model}    |    embed> {base.embedding_model}")
        st.caption(f"index> {base.qdrant_collection}    |    {len(shared.store.chunks)} passages")
        st.divider()
        if not st.session_state.messages:
            st.text("assistant> Ready. Ask about requirements, storage rules, or the API contract.")
            st.caption("Answers use the supplied documents only. No application API access.")
        for question, answer, error in st.session_state.messages:
            with st.chat_message("user", avatar=":material/chevron_right:"):
                st.text(f"you> {question}")
            with st.chat_message("assistant", avatar=":material/terminal:"):
                st.caption("assistant>")
                if error:
                    st.error(error)
                else:
                    show_answer(answer)
        question = st.chat_input("you> Ask the documents", max_chars=settings.max_question_chars)
        if question and question.strip():
            if question.strip() == "/clear":
                conversation.clear()
                st.session_state.messages = []
            else:
                answer, error = None, None
                try:
                    with st.spinner("assistant> retrieving and reading..."):
                        answer = conversation.ask(question)
                except (RagError, OSError, RuntimeError, ValueError) as exc:
                    error = str(exc)
                st.session_state.messages.append((question, answer, error))
            # Save a completed result before rerunning. Never infer from saved prompts.
            st.rerun()
    except (RagError, OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            st.error(
                "Invalid configuration: "
                + "; ".join(error["msg"] for error in exc.errors(include_input=False))
            )
        else:
            st.error(str(exc))
        st.caption("Check Ollama and the index. Close other terminal/ingestion processes first.")
        if st.button("Retry", key="retry"):
            st.rerun()


if __name__ == "__main__":
    main()

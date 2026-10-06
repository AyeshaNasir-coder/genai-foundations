"""
Streamlit chatbot UI for a local Ollama model, built on LangChain.

Run with:
    streamlit run app.py

Requirements:
    pip install streamlit langchain-ollama langchain-core
    ollama pull gemma3:1b
"""

from __future__ import annotations

from typing import Iterator

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama

# --------------------------------------------------------------------------- #
# Configuration (same defaults as main.py)
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "gemma3:1b"
DEFAULT_TEMPERATURE = 0.7
APP_TITLE = "AI Chat Assistant"
APP_ICON = "💬"


# --------------------------------------------------------------------------- #
# AI layer (no Streamlit UI code in here, except for resource caching)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_llm(model: str, temperature: float) -> ChatOllama:
    """Create (and cache) the chat model, same setup as main.py."""
    return ChatOllama(model=model, temperature=temperature)


def to_langchain_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert UI history into LangChain HumanMessage / AIMessage objects."""
    messages: list[BaseMessage] = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


def stream_response(llm: ChatOllama, history: list[dict]) -> Iterator[str]:
    """Yield the model's reply token-by-token for the given chat history."""
    for chunk in llm.stream(to_langchain_messages(history)):
        if chunk.content:
            yield chunk.content


def friendly_error(exc: Exception, model: str) -> str:
    """Turn raw exceptions into messages a user can act on."""
    text = str(exc).lower()
    if "connect" in text or "refused" in text:
        return (
            "Could not connect to the Ollama server. "
            "Make sure Ollama is running (`ollama serve`) and try again."
        )
    if "not found" in text or "pull" in text:
        return f"Model `{model}` was not found. Run `ollama pull {model}` first."
    return f"Something went wrong: {exc}"


# --------------------------------------------------------------------------- #
# UI layer
# --------------------------------------------------------------------------- #
def setup_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.markdown(
        """
        <style>
            .block-container { max-width: 820px; padding-top: 2rem; padding-bottom: 6rem; }
            #MainMenu, footer { visibility: hidden; }
            [data-testid="stChatMessage"] { border-radius: 14px; padding: 0.75rem 1rem; }
            .app-subtitle { color: #8a8f98; margin-top: -0.6rem; margin-bottom: 1.2rem; }
            @media (max-width: 640px) {
                .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("messages", [])


def render_sidebar() -> tuple[str, float]:
    with st.sidebar:
        st.header("⚙️ Settings")
        model = st.text_input("Ollama model", value=DEFAULT_MODEL)
        temperature = st.slider("Temperature", 0.0, 1.5, DEFAULT_TEMPERATURE, 0.05)

        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        n_msgs = len(st.session_state.messages)
        st.caption(f"{n_msgs} message{'s' if n_msgs != 1 else ''} in this chat")
    return model.strip() or DEFAULT_MODEL, temperature


def render_history() -> None:
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])


def handle_user_input(model: str, temperature: float) -> None:
    prompt = st.chat_input("Type your message...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            llm = get_llm(model, temperature)
            with st.spinner("Thinking..."):
                stream = stream_response(llm, st.session_state.messages)
                first = next(stream, "")  # spinner shows until first token arrives

            def full_stream() -> Iterator[str]:
                yield first
                yield from stream

            reply = st.write_stream(full_stream())
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as exc:  # noqa: BLE001
            # Remove the unanswered user message so history stays consistent.
            st.session_state.messages.pop()
            st.error(friendly_error(exc, model), icon="🚨")


def main() -> None:
    setup_page()
    init_state()
    model, temperature = render_sidebar()

    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown(
        f'<p class="app-subtitle">Powered by <b>{model}</b> via Ollama &amp; LangChain</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.info("Ask me anything. Start the conversation below 👇")

    render_history()
    handle_user_input(model, temperature)


if __name__ == "__main__":
    main()

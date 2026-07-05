import streamlit as st

from aimu import PROVENANCE_KEY, paths
from aimu.agents import Agent
from aimu.history import ConversationManager
from aimu.models import StreamingContentType, available_text_clients
from aimu.tools import builtin


def stream_reply(client, message):
    """Run one assistant turn as an Agent (chat() is single-turn; the Agent loops it until
    the model stops calling tools) and yield the visible answer tokens. Tools live on the
    Agent — the model client is a pure provider adapter that no longer executes tools."""
    for chunk in Agent(client, tools=builtin.ALL_TOOLS).run(message, stream=True):
        if chunk.phase == StreamingContentType.GENERATING:
            yield chunk.content


SYSTEM_MESSAGE = """
You are a helpful assistant that can answer questions, provide information, and assist with tasks.
You will always use available tools to help answer questions and complete tasks.
"""

INITIAL_USER_MESSAGE = "Introduce what model you are and share what tools you have access to."


def _new_client(client_cls, model):
    return client_cls(model, system_message=SYSTEM_MESSAGE)


if "client" not in st.session_state:
    client_cls = available_text_clients()[0]
    model = client_cls.TOOL_MODELS[0]
    st.session_state.client = _new_client(client_cls, model)
    st.session_state.conversation_manager = ConversationManager(
        db_path=str(paths.output / "chat_history_basic.json"),
        use_last_conversation=True,
    )
    st.session_state.client.messages = st.session_state.conversation_manager.messages

with st.sidebar:
    st.title("AIMU Chatbot")
    st.write("Basic AI Assistant")

    client_cls = st.selectbox("Model client", options=available_text_clients(), format_func=lambda x: x.__name__)
    model = st.selectbox("Model", options=client_cls.TOOL_MODELS, format_func=lambda x: x.name)

    current = st.session_state.client
    if not isinstance(current, client_cls) or current.model != model:
        st.session_state.client = _new_client(client_cls, model)
        st.rerun()

    if st.button("Reset chat"):
        st.session_state.conversation_manager.create_new_conversation()
        st.session_state.client = _new_client(client_cls, model)
        st.rerun()

for msg in st.session_state.client.messages[2:]:
    # Skip framework-injected turns (agent-loop continuation / final-answer prompts) so history
    # shows only genuine user input and model replies. See aimu.PROVENANCE_KEY.
    if msg.get(PROVENANCE_KEY):
        continue
    if msg["role"] in ("user", "assistant") and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if len(st.session_state.client.messages) == 0:
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""
        for token in stream_reply(st.session_state.client, INITIAL_USER_MESSAGE):
            response += token
            placeholder.markdown(response)
    st.session_state.conversation_manager.update_conversation(st.session_state.client.messages)

if prompt := st.chat_input("What's up?"):
    st.chat_message("user").markdown(prompt)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""
        for token in stream_reply(st.session_state.client, prompt):
            response += token
            placeholder.markdown(response)
    st.session_state.conversation_manager.update_conversation(st.session_state.client.messages)

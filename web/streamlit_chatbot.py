from aimu import paths
from aimu.models import HuggingFaceClient, OllamaClient, StreamPhase
from aimu.tools.client import MCPClient
from aimu.history import ConversationManager

import streamlit as st
import torch
import json  # used for the Messages debug popover

# Avoid torch RuntimeError when using Hugging Face Transformers
torch.classes.__path__ = []

SYSTEM_MESSAGE = """
You are a helpful assistant that can answer questions, provide information, and assist with tasks.
You will always use available tools to help answer questions and complete tasks.
"""

INITIAL_USER_MESSAGE = """
Introduce what model that you are and share what tools you have access to.
"""

MODEL_CLIENTS = [OllamaClient, HuggingFaceClient]

SLIDER_DEFAULTS = {"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.1}


def _set_slider_defaults(model):
    for key, default in SLIDER_DEFAULTS.items():
        st.session_state[f"slider_{key}"] = model.generation_kwargs.get(key, default)


def stream_chat_response(streamed_response):
    current_type = None
    current_box = None
    current_text = ""

    for chunk in streamed_response:
        if chunk.phase == StreamPhase.TOOL_CALLING:
            current_type = None  # force a fresh box on the next phase
            with st.expander("🔧 Tool call"):
                st.markdown(f"**Tool call:** {chunk.content['name']}")
                st.markdown(f"**Tool response:** {chunk.content['response']}")
            continue

        if chunk.phase != current_type:
            current_type = chunk.phase
            current_text = ""
            current_box = (
                st.expander("🤔 Thinking").empty()
                if chunk.phase == StreamPhase.THINKING
                else st.chat_message("assistant").empty()
            )

        current_text += chunk.content
        current_box.markdown(current_text)


MCP_SERVERS = {
    "mcpServers": {
        "aimu": {"command": "python", "args": ["-m", "aimu.tools.mcp"]},
    }
}

# Initialize the session state if we don't already have a model loaded. This only happens first run.
if "model_client" not in st.session_state:
    st.session_state.model = MODEL_CLIENTS[0].TOOL_MODELS[0]
    st.session_state.model_client = MODEL_CLIENTS[0](st.session_state.model, system_message=SYSTEM_MESSAGE)

    st.session_state.mcp_client = MCPClient(MCP_SERVERS)
    st.session_state.model_client.mcp_client = st.session_state.mcp_client

    _set_slider_defaults(st.session_state.model)

    st.session_state.conversation_manager = ConversationManager(
        db_path=str(paths.output / "chat_history.json"),
        use_last_conversation=True,
    )
    st.session_state.model_client.messages = st.session_state.conversation_manager.messages

with st.sidebar:
    st.title("AIMU Chatbot")
    st.write("Example AI Assistant")

    model = st.selectbox("Model", options=st.session_state.model_client.TOOL_MODELS, format_func=lambda x: x.name)
    model_client = st.selectbox("Model Client", options=MODEL_CLIENTS, format_func=lambda x: x.__name__)

    # If the model client has changed (e.g. OllamaClient to HuggingFaceClient), create a new mode client instance.
    # Otherwise, if the model has changed, create a new instance of the model client using the new model.
    # These checks must run before the sliders are rendered so that _set_slider_defaults can update
    # session state keys that are bound to slider widgets without triggering a StreamlitAPIException.
    if not isinstance(st.session_state.model_client, model_client):
        del st.session_state.model_client

        st.session_state.model = model_client.TOOL_MODELS[0]
        st.session_state.model_client = model_client(st.session_state.model, system_message=SYSTEM_MESSAGE)

        st.session_state.model_client.mcp_client = st.session_state.mcp_client

        _set_slider_defaults(st.session_state.model)
        st.rerun()
    elif st.session_state.model != model:
        del st.session_state.model_client

        st.session_state.model = model
        st.session_state.model_client = model_client(st.session_state.model, system_message=SYSTEM_MESSAGE)

        st.session_state.model_client.mcp_client = st.session_state.mcp_client

        _set_slider_defaults(model)
        st.rerun()

    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, step=0.01, key="slider_temperature")
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, step=0.01, key="slider_top_p")
    repeat_penalty = st.sidebar.slider(
        "repeat_penalty", min_value=0.9, max_value=1.5, step=0.1, key="slider_repeat_penalty"
    )

    if st.button("Reset chat"):
        # Create a new conversation that will be used as the "last" conversation when the app is reloaded.
        st.session_state.conversation_manager.create_new_conversation()

        st.session_state.clear()
        st.rerun()

if len(st.session_state.model_client.messages) == 0:
    streamed_response = st.session_state.model_client.chat_streamed(
        INITIAL_USER_MESSAGE,
        generate_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": 1024,
            "repeat_penalty": repeat_penalty,
        },
    )

    stream_chat_response(streamed_response)

    st.session_state.conversation_manager.update_conversation(st.session_state.model_client.messages)
else:
    # Skip the initial system and user messages used for the introduction
    messages = st.session_state.model_client.messages[2:]

    i = 0
    while i < len(messages):
        message = messages[i]

        if "thinking" in message:
            with st.expander("🤔 Thinking"):
                st.markdown(message["thinking"])

        # tool_calls are always immediately followed by their response messages, so we can check for them and render them together
        if "tool_calls" in message:
            responses = messages[i + 1 : i + 1 + len(message["tool_calls"])]
            for tool_call, response_msg in zip(message["tool_calls"], responses):
                with st.expander("🔧 Tool call"):
                    st.markdown(f"**Tool call:** {tool_call['function']['name']}")
                    st.markdown(f"**Tool response:** {response_msg['content']}")
            i += len(message["tool_calls"])  # skip the consumed tool response messages
        elif message["role"] != "tool" and "content" in message:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        i += 1

if prompt := st.chat_input("What's up?"):
    st.chat_message("user").markdown(prompt)

    streamed_response = st.session_state.model_client.chat_streamed(
        prompt,
        generate_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": 1024,
            "repeat_penalty": repeat_penalty,
        },
    )

    stream_chat_response(streamed_response)

    st.session_state.conversation_manager.update_conversation(st.session_state.model_client.messages)

# TODO: Determine better layout
with st.popover("Messages"):
    st.code(
        json.dumps(st.session_state.model_client.messages, indent=4),
        language="json",
        line_numbers=True,
    )

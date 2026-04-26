from aimu import paths
from aimu.agents.agentic_client import AgenticModelClient
from aimu.agents.simple_agent import SimpleAgent
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


def _wrap_agent(base_client, max_iterations):
    agent = SimpleAgent(base_client, max_iterations=max_iterations)
    return AgenticModelClient(agent)


def _rebuild_client(model_cls, model, agentic_mode, max_iterations):
    base_client = model_cls(model, system_message=SYSTEM_MESSAGE)
    base_client.mcp_client = st.session_state.mcp_client
    st.session_state.base_client = base_client
    st.session_state.model_client = _wrap_agent(base_client, max_iterations) if agentic_mode else base_client
    st.session_state.model = model
    st.session_state.agentic_mode = agentic_mode
    st.session_state.max_iterations = max_iterations
    _set_slider_defaults(model)


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
            current_box = None

        current_text += chunk.content
        if current_text:
            if current_box is None:
                current_box = (
                    st.expander("🤔 Thinking").empty()
                    if chunk.phase == StreamPhase.THINKING
                    else st.chat_message("assistant").empty()
                )
            current_box.markdown(current_text)


MCP_SERVERS = {
    "mcpServers": {
        "aimu": {"command": "python", "args": ["-m", "aimu.tools.mcp"]},
    }
}

# Initialize the session state if we don't already have a model loaded. This only happens first run.
if "model_client" not in st.session_state:
    st.session_state.mcp_client = MCPClient(MCP_SERVERS)
    _rebuild_client(MODEL_CLIENTS[0], MODEL_CLIENTS[0].TOOL_MODELS[0], False, 10)

    st.session_state.conversation_manager = ConversationManager(
        db_path=str(paths.output / "chat_history.json"),
        use_last_conversation=True,
    )
    st.session_state.model_client.messages = st.session_state.conversation_manager.messages

with st.sidebar:
    st.title("AIMU Chatbot")
    st.write("Example AI Assistant")

    # Model/client selectors use base_client since AgenticModelClient doesn't have TOOL_MODELS.
    model = st.selectbox("Model", options=st.session_state.base_client.TOOL_MODELS, format_func=lambda x: x.name)
    model_client = st.selectbox("Model Client", options=MODEL_CLIENTS, format_func=lambda x: x.__name__)
    agentic_mode = st.checkbox("Agentic mode", value=st.session_state.agentic_mode)
    max_iterations = st.number_input(
        "Max iterations", min_value=1, max_value=50, step=1,
        value=st.session_state.max_iterations, disabled=not agentic_mode,
    )

    # These checks must run before the sliders are rendered so that _set_slider_defaults can update
    # session state keys that are bound to slider widgets without triggering a StreamlitAPIException.
    # Use base_client for isinstance checks — model_client may be AgenticModelClient.
    if not isinstance(st.session_state.base_client, model_client):
        _rebuild_client(model_client, model_client.TOOL_MODELS[0], agentic_mode, max_iterations)
        st.rerun()
    elif st.session_state.model != model:
        _rebuild_client(model_client, model, agentic_mode, max_iterations)
        st.rerun()
    elif agentic_mode != st.session_state.agentic_mode or (agentic_mode and max_iterations != st.session_state.max_iterations):
        # Rewrap (or unwrap) without rebuilding the base client, preserving message history.
        st.session_state.model_client = _wrap_agent(st.session_state.base_client, max_iterations) if agentic_mode else st.session_state.base_client
        st.session_state.agentic_mode = agentic_mode
        st.session_state.max_iterations = max_iterations
        st.rerun()

    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, step=0.01, key="slider_temperature")
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, step=0.01, key="slider_top_p")
    repeat_penalty = st.sidebar.slider(
        "repeat_penalty", min_value=0.9, max_value=1.5, step=0.1, key="slider_repeat_penalty"
    )

    if st.button("Reset chat"):
        # Create a new conversation that will be used as the "last" conversation when the app is reloaded.
        st.session_state.conversation_manager.create_new_conversation()

        saved_agentic_mode = st.session_state.agentic_mode
        saved_max_iterations = st.session_state.max_iterations
        st.session_state.clear()
        st.session_state.agentic_mode = saved_agentic_mode
        st.session_state.max_iterations = saved_max_iterations
        st.rerun()

generate_kwargs = {
    "temperature": temperature,
    "top_p": top_p,
    "max_new_tokens": 1024,
    "repeat_penalty": repeat_penalty,
}

if len(st.session_state.model_client.messages) == 0:
    stream_chat_response(st.session_state.model_client.chat(INITIAL_USER_MESSAGE, generate_kwargs=generate_kwargs, stream=True))
    st.session_state.conversation_manager.update_conversation(st.session_state.model_client.messages)
else:
    # Skip the initial system and user messages used for the introduction
    msg_iter = iter(st.session_state.model_client.messages[2:])
    for message in msg_iter:
        if "thinking" in message:
            with st.expander("🤔 Thinking"):
                st.markdown(message["thinking"])
        if "tool_calls" in message:
            for tool_call, resp in zip(message["tool_calls"], [next(msg_iter) for _ in message["tool_calls"]]):
                with st.expander("🔧 Tool call"):
                    st.markdown(f"**Tool call:** {tool_call['function']['name']}")
                    st.markdown(f"**Tool response:** {resp['content']}")
        elif message["role"] != "tool" and message.get("content"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.chat_message("user").markdown(prompt)
    stream_chat_response(st.session_state.model_client.chat(prompt, generate_kwargs=generate_kwargs, stream=True))
    st.session_state.conversation_manager.update_conversation(st.session_state.model_client.messages)

# TODO: Determine better layout
with st.popover("Messages"):
    st.code(
        json.dumps(st.session_state.model_client.messages, indent=4),
        language="json",
        line_numbers=True,
    )

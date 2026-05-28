import json  # used for the Messages debug popover
from pathlib import Path
from typing import Optional

import streamlit as st
import torch

from aimu import paths
from aimu.agents.agent import Agent
from aimu.history import ConversationManager
from aimu.models import (
    BaseAudioClient,
    BaseImageClient,
    HAS_HF_AUDIO,
    HAS_OLLAMA,
    OllamaClient,
    OllamaModel,
    StreamingContentType,
    available_audio_clients,
    available_image_clients,
    available_text_clients,
)
from aimu.tools import builtin

# Avoid torch RuntimeError when using Hugging Face Transformers
torch.classes.__path__ = []

SYSTEM_MESSAGE = """
You are a helpful assistant that can answer questions, provide information, and assist with tasks.
You will always use available tools to help answer questions and complete tasks.
"""

INITIAL_USER_MESSAGE = """
Introduce what model that you are and share what tools you have access to.
"""

MODEL_CLIENTS = available_text_clients()
IMAGE_CLIENT_CLASSES = available_image_clients()
AUDIO_CLIENT_CLASSES = available_audio_clients()

_DEFAULT_CLIENT = OllamaClient if HAS_OLLAMA else MODEL_CLIENTS[0]
_DEFAULT_MODEL = OllamaModel.QWEN_3_5_9B if HAS_OLLAMA else MODEL_CLIENTS[0].TOOL_MODELS[0]

SLIDER_DEFAULTS = {"temperature": 0.15, "top_p": 0.9, "repeat_penalty": 1.1}

# Preview slider bounds. Slider value of 0 maps to "off" (preview_every=None);
# 1..PREVIEW_MAX maps to that integer step frequency. Max chosen to be larger
# than the default step count of every shipped HF model — values that exceed
# total_steps simply never fire mid-way (only the final chunk carries a preview).
PREVIEW_MAX = 25


def _construct_audio_client(
    client_cls: type[BaseAudioClient], model
) -> tuple[Optional[BaseAudioClient], Optional[str]]:
    """Try to construct a fresh audio client. Returns (client, error_msg)."""
    try:
        return client_cls(model), None
    except (RuntimeError, ImportError) as exc:
        return None, str(exc)


def _rebuild_audio_client(client_cls: type[BaseAudioClient], model) -> None:
    """Construct a fresh audio client + update the agent's tools."""
    client, err = _construct_audio_client(client_cls, model)
    st.session_state.audio_client_class = client_cls
    st.session_state.audio_model = model
    st.session_state.audio_client = client
    st.session_state.audio_client_error = err
    if "base_client" in st.session_state:
        st.session_state.base_client.tools = builtin.make_tools(
            st.session_state.base_client,
            st.session_state.get("image_client"),
            st.session_state.get("preview_every"),
            st.session_state.get("audio_client"),
            image_steps=st.session_state.get("image_steps"),
            audio_steps=st.session_state.get("audio_steps"),
        )


def _construct_image_client(
    client_cls: type[BaseImageClient], model
) -> tuple[Optional[BaseImageClient], Optional[str]]:
    """Try to construct a fresh image client. Returns (client, error_msg)."""
    try:
        return client_cls(model), None
    except (RuntimeError, ImportError) as exc:
        # Most common: GOOGLE_API_KEY missing for GeminiImageClient.
        return None, str(exc)


def _rebuild_image_client(client_cls: type[BaseImageClient], model) -> None:
    """Construct a fresh image client + update the agent's tools.

    Stores the client (or None + error message) in session_state. Refreshes the
    base client's tool list so the bound ``generate_image`` picks up the new
    image client and the current ``preview_every`` and ``image_steps`` settings.
    """
    client, err = _construct_image_client(client_cls, model)
    st.session_state.image_client_class = client_cls
    st.session_state.image_model = model
    st.session_state.image_client = client
    st.session_state.image_client_error = err
    if "base_client" in st.session_state:
        st.session_state.base_client.tools = builtin.make_tools(
            st.session_state.base_client,
            client,
            st.session_state.get("preview_every"),
            st.session_state.get("audio_client"),
            image_steps=st.session_state.get("image_steps"),
            audio_steps=st.session_state.get("audio_steps"),
        )


# Generated images land here (matches the library's default for `format="path"`).
IMAGE_DIR = paths.output / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


def _maybe_render_audio(path_str, key_suffix):
    """Display st.audio() + download button when ``path_str`` is a valid audio file path."""
    try:
        p = Path(str(path_str).strip())
    except (TypeError, ValueError):
        return
    if not p.exists() or not p.is_file() or p.suffix.lower() not in _AUDIO_EXTENSIONS:
        return
    st.audio(str(p))
    try:
        data = p.read_bytes()
    except OSError:
        return
    mime = f"audio/{p.suffix.lstrip('.').lower()}"
    st.download_button(
        label=f"Download {p.name}",
        data=data,
        file_name=p.name,
        mime=mime,
        key=f"dl_audio_{key_suffix}_{p.name}",
    )


def _maybe_render_image(path_str, key_suffix):
    """Display an image with a download button when ``path_str`` is a valid image file path."""
    try:
        p = Path(str(path_str).strip())
    except (TypeError, ValueError):
        return
    if not p.exists() or not p.is_file() or p.suffix.lower() not in _IMAGE_EXTENSIONS:
        return
    st.image(str(p))
    try:
        data = p.read_bytes()
    except OSError:
        return
    mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else f"image/{p.suffix.lstrip('.').lower()}"
    st.download_button(
        label=f"Download {p.name}",
        data=data,
        file_name=p.name,
        mime=mime,
        key=f"dl_{key_suffix}_{p.name}",
    )


def _set_slider_defaults(model):
    for key, default in SLIDER_DEFAULTS.items():
        st.session_state[f"slider_{key}"] = model.generation_kwargs.get(key, default)


def _wrap_agent(base_client, max_iterations):
    """Wrap a base model client in an Agent so chat() drives a multi-round tool-calling loop."""
    agent = Agent(base_client, max_iterations=max_iterations)
    return agent.as_model_client()


def _rebuild_client(model_cls, model, agentic_mode, max_iterations):
    """Construct a fresh base client and (optionally) agentic wrapper, then sync session state to match."""
    base_client = model_cls(model, system_message=SYSTEM_MESSAGE)
    base_client.tools = builtin.make_tools(
        base_client,
        st.session_state.get("image_client"),
        st.session_state.get("preview_every"),
        st.session_state.get("audio_client"),
        image_steps=st.session_state.get("image_steps"),
        audio_steps=st.session_state.get("audio_steps"),
    )
    # Store the base client separately since the agent wrapper doesn't have all the same attributes (e.g. TOOL_MODELS) and we need to reference those in the sidebar selectors and checks.
    st.session_state.base_client = base_client
    st.session_state.model_client = _wrap_agent(base_client, max_iterations) if agentic_mode else base_client
    st.session_state.model = model
    st.session_state.agentic_mode = agentic_mode
    st.session_state.max_iterations = max_iterations
    _set_slider_defaults(model)


def stream_chat_response(streamed_response):
    """Render a streaming chat response, coalescing consecutive same-phase chunks into a single live-updating box."""
    current_type = None
    current_box = None
    current_text = ""

    # IMAGE_GENERATING and AUDIO_GENERATING progress widgets (one per tool call).
    progress_bar = None
    progress_text = None
    preview_placeholder = None
    audio_progress_bar = None
    audio_progress_text = None

    for chunk_idx, chunk in enumerate(streamed_response):
        if chunk.phase == StreamingContentType.IMAGE_GENERATING:
            # Reset text-coalescing state — next text chunk should start fresh.
            current_type = None
            content = chunk.content
            step = content.get("step", 0)
            total = content.get("total_steps", 1) or 1
            is_final = content.get("final", False)
            preview = content.get("image")

            # Lazily create progress widgets on the first IMAGE_GENERATING chunk.
            if progress_bar is None:
                with st.expander("🖼️ Generating image", expanded=True):
                    progress_bar = st.empty()
                    progress_text = st.empty()
                    preview_placeholder = st.empty()

            fraction = max(0.0, min(1.0, step / total))
            label = f"Step {step}/{total}" if total > 1 else ("Generating…" if not is_final else "Done")
            progress_bar.progress(fraction, text=label)
            progress_text.markdown(f"**{label}**")
            if preview is not None:
                preview_placeholder.image(preview, caption=f"Preview at step {step}/{total}")

            if is_final:
                # Clear the live progress widgets; the agent's TOOL_CALLING chunk
                # (next) will render the canonical Image expander + download button.
                progress_bar.empty()
                progress_text.empty()
                preview_placeholder.empty()
                progress_bar = None
                progress_text = None
                preview_placeholder = None
            continue

        if chunk.phase == StreamingContentType.AUDIO_GENERATING:
            current_type = None
            content = chunk.content
            step = content.get("step", 0)
            total = content.get("total_steps", 1) or 1
            is_final = content.get("final", False)

            if audio_progress_bar is None:
                with st.expander("🎵 Generating audio", expanded=True):
                    audio_progress_bar = st.empty()
                    audio_progress_text = st.empty()

            fraction = max(0.0, min(1.0, step / total))
            label = f"Step {step}/{total}" if total > 1 else ("Generating…" if not is_final else "Done")
            audio_progress_bar.progress(fraction, text=label)
            audio_progress_text.markdown(f"**{label}**")

            if is_final:
                audio_progress_bar.empty()
                audio_progress_text.empty()
                audio_progress_bar = None
                audio_progress_text = None
            continue

        if chunk.phase == StreamingContentType.TOOL_CALLING:
            # Tool calls render in their own expander, so reset current_type to
            # force a fresh box for whatever phase streams next (otherwise the
            # next thinking/generating chunk would append to a stale box).
            current_type = None
            with st.expander("🔧 Tool call"):
                st.markdown(f"**Tool call:** {chunk.content['name']}")
                args = chunk.content.get("arguments") or {}
                if args:
                    st.markdown("**Arguments:**")
                    st.json(args, expanded=False)
                st.markdown(f"**Tool response:** {chunk.content['response']}")
            if chunk.content["name"] == "generate_image":
                with st.expander("🖼️ Image", expanded=True):
                    _maybe_render_image(chunk.content["response"], key_suffix=f"stream_{chunk_idx}")
            elif chunk.content["name"] == "generate_audio":
                with st.expander("🎵 Audio", expanded=True):
                    _maybe_render_audio(chunk.content["response"], key_suffix=f"stream_{chunk_idx}")
            continue

        # Phase transition (THINKING ↔ GENERATING): start a new accumulator and box.
        if chunk.phase != current_type:
            current_type = chunk.phase
            current_text = ""
            current_box = None

        current_text += chunk.content
        if current_text:
            # Defer box creation until we have non-empty text — avoids rendering
            # an empty "Thinking" expander when a thinking phase yields nothing.
            if current_box is None:
                current_box = (
                    st.expander("🤔 Thinking").empty()
                    if chunk.phase == StreamingContentType.THINKING
                    else st.chat_message("assistant").empty()
                )
            current_box.markdown(current_text)


# Initialize the session state if we don't already have a model loaded. This only happens first run.
if "model_client" not in st.session_state:
    st.session_state.preview_every = None  # default: no intermediate previews (fastest)
    st.session_state.image_steps = None  # default: use the model's default step count
    st.session_state.audio_steps = None  # default: use the model's default step count
    # Initialise image-client state up-front so _rebuild_client's make_tools call sees it.
    st.session_state.image_client = None
    st.session_state.image_client_class = None
    st.session_state.image_model = None
    st.session_state.image_client_error = None
    if IMAGE_CLIENT_CLASSES:
        default_image_cls = IMAGE_CLIENT_CLASSES[0]
        default_image_model = next(iter(default_image_cls.MODELS))
        # Construct without going through _rebuild_image_client — base_client doesn't exist yet.
        client, err = _construct_image_client(default_image_cls, default_image_model)
        st.session_state.image_client_class = default_image_cls
        st.session_state.image_model = default_image_model
        st.session_state.image_client = client
        st.session_state.image_client_error = err
    # Initialise audio-client state up-front so make_tools sees it.
    st.session_state.audio_client = None
    st.session_state.audio_client_class = None
    st.session_state.audio_model = None
    st.session_state.audio_client_error = None
    if AUDIO_CLIENT_CLASSES:
        default_audio_cls = AUDIO_CLIENT_CLASSES[0]
        default_audio_model = next(iter(default_audio_cls.MODELS))
        audio_c, audio_err = _construct_audio_client(default_audio_cls, default_audio_model)
        st.session_state.audio_client_class = default_audio_cls
        st.session_state.audio_model = default_audio_model
        st.session_state.audio_client = audio_c
        st.session_state.audio_client_error = audio_err
    _rebuild_client(_DEFAULT_CLIENT, _DEFAULT_MODEL, True, 10)

    st.session_state.conversation_manager = ConversationManager(
        db_path=str(paths.output / "chat_history.json"),
        use_last_conversation=True,
    )
    st.session_state.model_client.messages = st.session_state.conversation_manager.messages

with st.sidebar:
    st.title("AIMU Chatbot")
    st.write("Example AI Assistant")

    # Model/client selectors use base_client since the agentic view doesn't expose TOOL_MODELS.
    _tool_models = st.session_state.base_client.TOOL_MODELS
    model = st.selectbox(
        "Model",
        options=_tool_models,
        index=_tool_models.index(st.session_state.model) if st.session_state.model in _tool_models else 0,
        format_func=lambda x: x.name,
    )
    model_client = st.selectbox("Model Client", options=MODEL_CLIENTS, format_func=lambda x: x.__name__)
    agentic_mode = st.checkbox(
        "Agentic mode",
        value=st.session_state.agentic_mode,
        help="When enabled, the model runs in a multi-round tool-calling loop: it can call tools, observe results, and keep going until it has a final answer. When disabled, each message is a single inference pass.",
    )
    max_iterations = st.number_input(
        "Max iterations",
        min_value=1,
        max_value=50,
        step=1,
        value=st.session_state.max_iterations,
        disabled=not agentic_mode,
    )

    # These checks must run before the sliders are rendered so that _set_slider_defaults can update
    # session state keys that are bound to slider widgets without triggering a StreamlitAPIException.
    # Use base_client for isinstance checks — model_client may be an agentic view.
    if not isinstance(st.session_state.base_client, model_client):
        _rebuild_client(model_client, model_client.TOOL_MODELS[0], agentic_mode, max_iterations)
        st.rerun()
    elif st.session_state.model != model:
        _rebuild_client(model_client, model, agentic_mode, max_iterations)
        st.rerun()
    elif agentic_mode != st.session_state.agentic_mode or (
        agentic_mode and max_iterations != st.session_state.max_iterations
    ):
        # Rewrap (or unwrap) without rebuilding the base client, preserving message history.
        st.session_state.model_client = (
            _wrap_agent(st.session_state.base_client, max_iterations) if agentic_mode else st.session_state.base_client
        )
        st.session_state.agentic_mode = agentic_mode
        st.session_state.max_iterations = max_iterations
        st.rerun()

    temperature = st.sidebar.slider("temperature", min_value=0.01, max_value=1.0, step=0.01, key="slider_temperature")
    top_p = st.sidebar.slider("top_p", min_value=0.01, max_value=1.0, step=0.01, key="slider_top_p")
    repeat_penalty = st.sidebar.slider(
        "repeat_penalty", min_value=0.9, max_value=1.5, step=0.1, key="slider_repeat_penalty"
    )

    # Image generation — client/model selectors + preview-step slider.
    if IMAGE_CLIENT_CLASSES:
        st.markdown("---")
        st.markdown("**Image generation**")

        current_cls = st.session_state.image_client_class or IMAGE_CLIENT_CLASSES[0]
        sel_image_cls = st.selectbox(
            "Image client",
            options=IMAGE_CLIENT_CLASSES,
            index=IMAGE_CLIENT_CLASSES.index(current_cls) if current_cls in IMAGE_CLIENT_CLASSES else 0,
            format_func=lambda c: c.__name__,
        )

        image_model_options = list(sel_image_cls.MODELS)
        # If the user just switched clients, fall back to the first model of the new client.
        current_model = st.session_state.image_model if sel_image_cls is current_cls else image_model_options[0]
        sel_image_model = st.selectbox(
            "Image model",
            options=image_model_options,
            index=image_model_options.index(current_model) if current_model in image_model_options else 0,
            format_func=lambda m: m.name,
        )

        # Rebuild the image client whenever the selection changes.
        if sel_image_cls is not st.session_state.image_client_class or sel_image_model != st.session_state.image_model:
            _rebuild_image_client(sel_image_cls, sel_image_model)
            st.rerun()

        # Surface construction errors (e.g. missing GOOGLE_API_KEY for Gemini).
        if st.session_state.image_client_error:
            st.error(f"Image client unavailable: {st.session_state.image_client_error}")

        # Preview slider — 0 means off. Disabled when the active client is Gemini
        # (cloud API has no intermediate latents to decode).
        is_gemini = type(st.session_state.image_client).__name__ == "GeminiImageClient"
        current_preview = st.session_state.get("preview_every") or 0
        preview_raw = st.slider(
            "Preview every N denoising steps",
            min_value=0,
            max_value=PREVIEW_MAX,
            value=current_preview,
            step=1,
            disabled=is_gemini,
            help=(
                "0 = off (fastest). When >0, intermediate denoised images are decoded via "
                "the pipeline's VAE every N steps and shown live. Each decode adds ~50–200 ms "
                "on GPU. Disabled for Gemini Nano Banana — the cloud API has no intermediate steps."
            ),
        )
        selected_preview = None if preview_raw == 0 else preview_raw
        if selected_preview != st.session_state.get("preview_every"):
            st.session_state.preview_every = selected_preview
            # Rebuild the bound generate_image tool with the new preview frequency.
            st.session_state.base_client.tools = builtin.make_tools(
                st.session_state.base_client,
                st.session_state.image_client,
                selected_preview,
                st.session_state.get("audio_client"),
                image_steps=st.session_state.get("image_steps"),
                audio_steps=st.session_state.get("audio_steps"),
            )
            st.rerun()

        # Denoising steps slider — 0 maps to "use model default".
        is_hf_image = type(st.session_state.image_client).__name__ == "HuggingFaceImageClient"
        current_image_steps = st.session_state.get("image_steps") or 0
        image_steps_raw = st.slider(
            "Denoising steps (image)",
            min_value=0,
            max_value=100,
            value=current_image_steps,
            step=1,
            disabled=not is_hf_image,
            help=(
                "0 = use the model's default step count. "
                "Fewer steps generate faster but with lower quality; more steps improve quality "
                "but take longer. Only applies to HuggingFace diffusers models."
            ),
        )
        selected_image_steps = None if image_steps_raw == 0 else image_steps_raw
        if selected_image_steps != st.session_state.get("image_steps"):
            st.session_state.image_steps = selected_image_steps
            st.session_state.base_client.tools = builtin.make_tools(
                st.session_state.base_client,
                st.session_state.image_client,
                st.session_state.get("preview_every"),
                st.session_state.get("audio_client"),
                image_steps=selected_image_steps,
                audio_steps=st.session_state.get("audio_steps"),
            )
            st.rerun()

    # Audio generation — client/model selectors + duration slider.
    if AUDIO_CLIENT_CLASSES:
        st.markdown("---")
        st.markdown("**Audio generation**")

        current_audio_cls = st.session_state.audio_client_class or AUDIO_CLIENT_CLASSES[0]
        sel_audio_cls = st.selectbox(
            "Audio client",
            options=AUDIO_CLIENT_CLASSES,
            index=AUDIO_CLIENT_CLASSES.index(current_audio_cls) if current_audio_cls in AUDIO_CLIENT_CLASSES else 0,
            format_func=lambda c: c.__name__,
        )

        audio_model_options = list(sel_audio_cls.MODELS)
        current_audio_model = (
            st.session_state.audio_model if sel_audio_cls is current_audio_cls else audio_model_options[0]
        )
        sel_audio_model = st.selectbox(
            "Audio model",
            options=audio_model_options,
            index=audio_model_options.index(current_audio_model) if current_audio_model in audio_model_options else 0,
            format_func=lambda m: m.name,
        )

        if (
            sel_audio_cls is not st.session_state.audio_client_class
            or sel_audio_model != st.session_state.audio_model
        ):
            _rebuild_audio_client(sel_audio_cls, sel_audio_model)
            st.rerun()

        if st.session_state.audio_client_error:
            st.error(f"Audio client unavailable: {st.session_state.audio_client_error}")

        # Denoising steps slider — 0 maps to "use model default". Only meaningful
        # for diffusers-backed models (AudioLDM2, StableAudio); MusicGen ignores it.
        is_diffusers_audio = type(st.session_state.audio_client).__name__ == "HuggingFaceAudioClient" and (
            st.session_state.audio_model is not None
            and getattr(st.session_state.audio_model, "spec", None) is not None
            and getattr(st.session_state.audio_model.spec, "pipeline_type", "musicgen") != "musicgen"
        )
        current_audio_steps = st.session_state.get("audio_steps") or 0
        audio_steps_raw = st.slider(
            "Denoising steps (audio)",
            min_value=0,
            max_value=300,
            value=current_audio_steps,
            step=10,
            disabled=not is_diffusers_audio,
            help=(
                "0 = use the model's default step count. "
                "Only applies to diffusers-backed audio models (AudioLDM2, StableAudio). "
                "MusicGen is token-autoregressive and ignores this setting."
            ),
        )
        selected_audio_steps = None if audio_steps_raw == 0 else audio_steps_raw
        if selected_audio_steps != st.session_state.get("audio_steps"):
            st.session_state.audio_steps = selected_audio_steps
            st.session_state.base_client.tools = builtin.make_tools(
                st.session_state.base_client,
                st.session_state.get("image_client"),
                st.session_state.get("preview_every"),
                st.session_state.get("audio_client"),
                image_steps=st.session_state.get("image_steps"),
                audio_steps=selected_audio_steps,
            )
            st.rerun()

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
    stream_chat_response(
        st.session_state.model_client.chat(INITIAL_USER_MESSAGE, generate_kwargs=generate_kwargs, stream=True)
    )
    st.session_state.conversation_manager.update_conversation(st.session_state.model_client.messages)
else:
    # Skip the initial system and user messages used for the introduction
    msg_iter = iter(st.session_state.model_client.messages[2:])
    for hist_idx, message in enumerate(msg_iter):
        if "thinking" in message:
            with st.expander("🤔 Thinking"):
                st.markdown(message["thinking"])
        if "tool_calls" in message:
            for call_idx, (tool_call, resp) in enumerate(
                zip(message["tool_calls"], [next(msg_iter) for _ in message["tool_calls"]])
            ):
                with st.expander("🔧 Tool call"):
                    st.markdown(f"**Tool call:** {tool_call['function']['name']}")
                    args = tool_call["function"].get("arguments") or {}
                    if args:
                        st.markdown("**Arguments:**")
                        st.json(args, expanded=False)
                    st.markdown(f"**Tool response:** {resp['content']}")
                if tool_call["function"]["name"] == "generate_image":
                    with st.expander("🖼️ Image"):
                        _maybe_render_image(resp["content"], key_suffix=f"hist_{hist_idx}_{call_idx}")
                elif tool_call["function"]["name"] == "generate_audio":
                    with st.expander("🎵 Audio"):
                        _maybe_render_audio(resp["content"], key_suffix=f"hist_{hist_idx}_{call_idx}")
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

import gradio as gr

from aimu import paths
from aimu.agents import Agent
from aimu.history import ConversationManager
from aimu.models import StreamingContentType, available_text_clients
from aimu.tools import builtin

SYSTEM_MESSAGE = """
You are a helpful assistant that can answer questions, provide information, and assist with tasks.
You will always use available tools to help answer questions and complete tasks.
"""

INITIAL_USER_MESSAGE = "Introduce what model you are and share what tools you have access to."

_all_clients = available_text_clients()


def _new_client(client_cls, model):
    return client_cls(model, system_message=SYSTEM_MESSAGE)


def _stream_answer(client, message):
    """Run one assistant turn as an Agent (chat() is single-turn; the Agent loops it and
    executes tools) and yield visible answer tokens. Tools live on the Agent."""
    for chunk in Agent(client, tools=builtin.ALL_TOOLS).run(message, stream=True):
        if chunk.phase == StreamingContentType.GENERATING:
            yield chunk.content


def _new_manager(use_last_conversation=False):
    return ConversationManager(
        db_path=str(paths.output / "chat_history_gradio.json"),
        use_last_conversation=use_last_conversation,
    )


def _messages_to_history(messages):
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages[2:]
        if msg["role"] in ("user", "assistant") and msg.get("content")
    ]


def on_load(client, manager):
    history = _messages_to_history(client.messages)
    if history:
        yield history
        return

    history = [{"role": "assistant", "content": ""}]
    for token in _stream_answer(client, INITIAL_USER_MESSAGE):
        history[-1]["content"] += token
        yield history
    manager.update_conversation(client.messages)


def respond(message, history, client, manager):
    if client is None:
        gr.Warning(NO_OLLAMA_MODELS)
        yield history, message
        return
    history = list(history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    for token in _stream_answer(client, message):
        history[-1]["content"] += token
        yield history, ""

    manager.update_conversation(client.messages)


from aimu.models import OllamaClient, OllamaModel, available_text_models  # noqa: E402
from model_choice import NO_OLLAMA_MODELS, NO_READY_MODEL, default_client_and_model, model_key, model_options  # noqa: E402


def _model_options(cls):
    installed = {model_key(m) for m in available_text_models(include_hf_cache=False)}
    return model_options(cls, installed, ollama_client=OllamaClient)


_default_cls, _default_model = default_client_and_model(
    _all_clients,
    {model_key(m) for m in available_text_models(include_hf_cache=False)},
    ollama_client=OllamaClient,
    preferred=OllamaModel.QWEN_3_5_9B,
)
if _default_cls is None:
    raise SystemExit(NO_READY_MODEL)
_init_client = _new_client(_default_cls, _default_model)
_init_manager = _new_manager(use_last_conversation=True)
_init_client.messages = _init_manager.messages

with gr.Blocks(title="AIMU Chatbot") as demo:
    client_state = gr.State(_init_client)
    manager_state = gr.State(_init_manager)

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("## AIMU Chatbot")
            gr.Markdown("Basic AI Assistant")
            client_dropdown = gr.Dropdown(
                label="Model client",
                choices=[c.__name__ for c in _all_clients],
                value=_default_cls.__name__,
            )
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=[m.name for m in _model_options(_default_cls)],
                value=_default_model.name,
            )
            reset_btn = gr.Button("Reset chat")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="What's up?", show_label=False, scale=8)
                send_btn = gr.Button("Send", scale=1, variant="primary")

    def _resolve_model(cls, model_name):
        return next(m for m in cls.TOOL_MODELS if m.name == model_name)

    def on_client_change(client_name):
        cls = next(c for c in _all_clients if c.__name__ == client_name)
        options = _model_options(cls)
        if not options:
            gr.Warning(NO_OLLAMA_MODELS)
            return gr.update(choices=[], value=None), None, _new_manager(), []
        model = options[0]
        return (
            gr.update(choices=[m.name for m in options], value=model.name),
            _new_client(cls, model),
            _new_manager(),
            [],
        )

    def on_model_change(client_name, model_name):
        if model_name is None:
            return None, _new_manager(), []
        cls = next(c for c in _all_clients if c.__name__ == client_name)
        return _new_client(cls, _resolve_model(cls, model_name)), _new_manager(), []

    def on_reset(client_name, model_name):
        cls = next(c for c in _all_clients if c.__name__ == client_name)
        mgr = _new_manager()
        mgr.create_new_conversation()
        if model_name is None:
            return None, mgr, []
        return _new_client(cls, _resolve_model(cls, model_name)), mgr, []

    client_dropdown.change(
        fn=on_client_change,
        inputs=[client_dropdown],
        outputs=[model_dropdown, client_state, manager_state, chatbot],
    )
    model_dropdown.change(
        fn=on_model_change,
        inputs=[client_dropdown, model_dropdown],
        outputs=[client_state, manager_state, chatbot],
    )
    reset_btn.click(
        fn=on_reset,
        inputs=[client_dropdown, model_dropdown],
        outputs=[client_state, manager_state, chatbot],
    )

    submit = dict(
        fn=respond,
        inputs=[chat_input, chatbot, client_state, manager_state],
        outputs=[chatbot, chat_input],
    )
    chat_input.submit(**submit)
    send_btn.click(**submit)

    demo.load(fn=on_load, inputs=[client_state, manager_state], outputs=[chatbot])


if __name__ == "__main__":
    demo.launch()

import gradio as gr

from aimu.models import OllamaClient, HuggingFaceClient, AisuiteClient

MODEL_CLIENTS = [OllamaClient, HuggingFaceClient,AisuiteClient]

def respond(message, history, client):
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    for chunk in client.chat_streamed(message):
        history[-1]["content"] += chunk.content
        yield history, ""


_default_class = MODEL_CLIENTS[0]
_default_model = _default_class.TOOL_MODELS[0]

with gr.Blocks(title="AIMU Chatbot") as demo:
    client_state = gr.State(_default_class(_default_model))

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            client_dropdown = gr.Dropdown(
                label="Client",
                choices=[c.__name__ for c in MODEL_CLIENTS],
                value=_default_class.__name__,
            )
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=[(m.name, m) for m in _default_class.TOOL_MODELS],
                value=_default_model,
            )

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(show_label=False)
            with gr.Row():
                chat_input = gr.Textbox(placeholder="Say something...", show_label=False, scale=8)
                send_btn = gr.Button("Send", scale=1, variant="primary")

    def on_client_change(client_name):
        client_class = next(c for c in MODEL_CLIENTS if c.__name__ == client_name)
        models = client_class.TOOL_MODELS[0]
        new_model = models[0]
        return gr.update(choices=[(m.name, m) for m in models], value=new_model), client_class(new_model), []

    def on_model_change(client_name, model):
        client_class = next(c for c in MODEL_CLIENTS if c.__name__ == client_name)
        return client_class(model), []

    client_dropdown.change(
        fn=on_client_change,
        inputs=[client_dropdown],
        outputs=[model_dropdown, client_state, chatbot],
    )
    model_dropdown.change(
        fn=on_model_change,
        inputs=[client_dropdown, model_dropdown],
        outputs=[client_state, chatbot],
    )

    submit = dict(fn=respond, inputs=[chat_input, chatbot, client_state], outputs=[chatbot, chat_input])
    chat_input.submit(**submit)
    send_btn.click(**submit)


if __name__ == "__main__":
    demo.launch()

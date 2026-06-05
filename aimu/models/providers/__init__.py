"""Concrete provider clients, one module per provider.

Flat modules (``anthropic``, ``ollama``, ``llamacpp``) hold a single-provider client;
``openai_compat`` holds the generic OpenAI-compatible base plus the local-server
subclasses (Ollama-OpenAI, LM Studio, vLLM, HF-Serve, llama-server, SGLang). Providers
with several standalone modality clients get a subpackage named by modality:

- ``hf/{text,image,audio,speech}``   -- HuggingFace in-process clients
- ``openai/{text,speech}``           -- OpenAI cloud (GPT/o-series + TTS)
- ``gemini/{text,image}``            -- Google Gemini (text via the OpenAI-compat endpoint, image via google-genai)

The factory modules (``aimu.models.model_client`` etc.) import the leaf modules lazily,
so a missing optional dependency doesn't break the rest of the package. Mirrors
``aimu.aio.providers``.
"""

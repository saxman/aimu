# `aimu.models`

Provider-agnostic model clients.

## Factory and base class

::: aimu.models.ModelClient

::: aimu.models.BaseModelClient

## Types

::: aimu.models.ModelSpec

::: aimu.models.Model

::: aimu.models.StreamChunk

::: aimu.models.StreamingContentType

## Provider clients

::: aimu.models.OllamaClient
    options:
      show_root_heading: true
      members: [OllamaModel]

::: aimu.models.AnthropicClient

::: aimu.models.HuggingFaceClient

::: aimu.models.LlamaCppClient

::: aimu.models.OpenAIClient

::: aimu.models.GeminiClient

::: aimu.models.LMStudioOpenAIClient

::: aimu.models.OllamaOpenAIClient

::: aimu.models.HFOpenAIClient

::: aimu.models.VLLMOpenAIClient

::: aimu.models.LlamaServerOpenAIClient

::: aimu.models.SGLangOpenAIClient

::: aimu.models.OpenAICompatClient

## Embedding clients

Text-to-vector clients, a parallel surface to the chat clients. See the
[Embed text](../../how-to/use-embeddings.md) guide.

::: aimu.models.EmbeddingClient

::: aimu.models.BaseEmbeddingClient

::: aimu.models.resolve_embedding_model_string

::: aimu.models.OpenAIEmbeddingClient
    options:
      show_root_heading: true
      members: [OpenAIEmbeddingModel]

::: aimu.models.OllamaEmbeddingClient
    options:
      show_root_heading: true
      members: [OllamaEmbeddingModel]

::: aimu.models.HuggingFaceEmbeddingClient
    options:
      show_root_heading: true
      members: [HuggingFaceEmbeddingModel]

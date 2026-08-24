# `aimu.context`

Plain functions over a conversation's `list[dict]`: estimate its size, trim it to a token
budget, or replace the older part with an LLM-generated summary. No `ContextPolicy` class, no
hidden rewriting inside a client. See [how-to: manage context](../../how-to/manage-context.md).

::: aimu.context.count_tokens

::: aimu.context.trim_messages

::: aimu.context.summarize_messages

::: aimu.context.DEFAULT_SUMMARIZE_PROMPT

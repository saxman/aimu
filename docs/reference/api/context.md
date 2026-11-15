# `aimu.context`

What a request to a model holds, as plain functions over a conversation's `list[dict]`. No
`ContextPolicy` class: an `Agent(compaction=...)` applies a compaction to what each request
sends, and the history stays whole. See [how-to: manage context](../../how-to/manage-context.md).

::: aimu.context.count_tokens

::: aimu.context.Compaction

## `aimu.context.compaction`

The strategies, and the ready-made compactions built from them, bundled the way
`aimu.tools.builtin` bundles tools.

### Ready-made compactions

::: aimu.context.compaction.mask

::: aimu.context.compaction.trim

::: aimu.context.compaction.chain

### Strategies

::: aimu.context.compaction.mask_messages

::: aimu.context.compaction.default_note

::: aimu.context.compaction.MASK_ORDER

::: aimu.context.compaction.trim_messages

::: aimu.context.compaction.summarize_messages

::: aimu.context.compaction.DEFAULT_SUMMARIZE_PROMPT

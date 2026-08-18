# Thinking content and the model context

Reasoning models (Qwen3, DeepSeek-R1, Gemini 2.5, Claude with extended thinking, ...) emit a chain of thought before their answer. AIMU separates that reasoning from the answer and surfaces it on `client.last_thinking` and as `THINKING` stream chunks. This page records *what* AIMU does with the reasoning after a turn finishes, and *why* the prior turn's reasoning is not fed back into the model on the next turn.

## What happens to thinking after a turn

For every provider, the answer stored in `self.messages` is the **clean** answer with the `<think>...</think>` block removed:

- Local `<think>`-tag providers (HuggingFace, llama-cpp, OpenAI-compat local servers) parse the tags with `_split_thinking` / `_ThinkingParser` ([aimu/models/providers/_thinking.py](https://github.com/saxman/aimu/blob/main/aimu/models/providers/_thinking.py)).
- Native-thinking providers (Anthropic, Ollama) read the reasoning from a dedicated response field.

The most recent turn's reasoning is always available on `client.last_thinking` (the canonical, uniform surface). In addition, every provider attaches the reasoning to the assistant message it just appended, under a non-standard `"thinking"` key:

```python
{"role": "assistant", "content": "the answer", "thinking": "the reasoning"}
```

The key is **omitted** when the turn produced no reasoning. This is what the Streamlit chatbot renders per message ([examples/web/streamlit_chatbot.py](https://github.com/saxman/aimu/blob/main/examples/web/streamlit_chatbot.py)) and what `ConversationManager` persists, so a saved conversation keeps its per-turn reasoning for later inspection.

This applies to **tool-calling turns** too: when a thinking model reasons and then calls a tool, that reasoning is attached to the assistant *tool-call* message (the one carrying `tool_calls`), not just to the final answer. So in an agentic loop every assistant message that had reasoning carries its own reasoning. As with everything else here, the key is metadata: the reasoning that preceded a tool call is recorded for inspection/persistence but is not re-sent to the model on the follow-up request.

### Why `last_thinking` *and* the message key

`last_thinking` holds only the **latest** turn's reasoning; it is overwritten every call. The `"thinking"` message key keeps a **per-turn** record so a UI or a persisted transcript can show the reasoning next to the answer it produced. Read `last_thinking` for "what did the model just reason about"; read the message key for "what did it reason about on turn N".

## The `"thinking"` key is metadata, not model input

The `"thinking"` key is **not** part of the OpenAI message schema. It is deliberately a side-channel:

- Local chat templates (`tokenizer.apply_chat_template`) read `role`, `content`, and `tool_calls`. They never reference a `"thinking"` key, so it is ignored when the next request is rendered.
- The Anthropic and HuggingFace adapters rebuild each request payload from `role` / `content` / `tool_calls`, so the extra key never survives the conversion.
- OpenAI-compat (OpenAI, Gemini, local servers) and Ollama forward message dicts closer to verbatim, so those two request paths call `strip_inert_keys()` (`aimu.models._internal.message_meta`) first, removing every key in `INERT_MESSAGE_KEYS` (`"thinking"`, `"provenance"`, `"timestamp"`).

So the key is inert with respect to the model. It exists for humans and persistence, not for re-conditioning the model. This keeps `self.messages` usable as plain data while still carrying the reasoning for tooling. (Design principle #2 — "plain data" — is preserved in spirit: the reasoning travels as an additive, ignorable annotation, never as a new message class, and never changes what the provider sees.)

## The `"provenance"` key: distinguishing framework-injected turns

The same inert-key mechanism carries a second annotation. The `Agent` loop injects `{"role": "user", ...}` turns the human never typed: the `continuation_prompt` between tool-calling rounds, and the `final_answer_prompt` when the loop hits `max_iterations`. Scheduler "proactive" turns are similar (framework-initiated, not user-initiated). Left unmarked, all three are byte-for-byte identical to real input, so a replayed or persisted transcript can't tell them apart.

AIMU tags them with a `"provenance"` key (`aimu.PROVENANCE_KEY`), whose value is one of `PROVENANCE_CONTINUATION`, `PROVENANCE_FINAL_ANSWER`, or `PROVENANCE_PROACTIVE`. **Genuine user input and ordinary assistant turns are left untagged** (absence means "ordinary turn"), so display logic is just `message.get(PROVENANCE_KEY)`. Like `"thinking"`, the key is inert (it is in `INERT_MESSAGE_KEYS`, so it is stripped before every provider request) and is set on the message by index after the injecting `chat()` turn completes, keeping the public `chat()` signature free of the concept. The Streamlit examples in `examples/web/` hide or mute tagged turns when re-rendering history.

## Why prior-turn thinking is *not* re-fed to the model

On a multi-turn conversation, the reasoning from earlier assistant turns is **not** sent back to the model. This is intentional and matches the model authors' explicit guidance:

- **Qwen3** (AIMU's primary local thinking family): *"No Thinking Content in History: In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content."* This is enforced **inside the Qwen3 Jinja chat template itself** — even if AIMU left `<think>` in the message `content`, the template would strip it for all but the live turn.
- **Gemma** documents the same: strip generated thoughts from previous turns before passing history back.
- The original **DeepSeek-R1** convention is identical.

Re-feeding prior reasoning would *deviate* from this guidance, inflate the prompt, and can degrade quality. AIMU therefore does the recommended thing: the model re-derives reasoning fresh each turn from the clean history.

### Qwen 3.8 and `preserve_thinking`

Qwen 3.8's chat template adds a `preserve_thinking` toggle, defaulting to `True`, that keeps the current turn's `<think>` block in the rendered prompt on the *next* turn instead of stripping it: the opposite of the guidance above for the 3.5 / 3.6 line described just above. AIMU does not expose this toggle, and its own behavior does not change for Qwen 3.8: prior-turn reasoning is never sent to any provider, for any model, because the `"thinking"` key is inert (see above) and is stripped from every request path before the model ever sees it. In practice, then, AIMU behaves as `preserve_thinking=False` for Qwen 3.8, permanently.

This is a **deliberate deviation** from the card's own default, not an oversight. Re-feeding reasoning would grow the prompt by one thinking block's worth of tokens on every turn for the life of a conversation. And the benefit `preserve_thinking=True` is meant to provide, a later turn (or a human, or a persistence layer) being able to see what the model reasoned about, is already met a different way: every provider attaches that turn's reasoning to the assistant message under the inert `"thinking"` key (see above), so it is fully recorded for inspection and persistence without ever being replayed into a request.

## Controlling thinking: the `thinking=` parameter

`chat()` and `generate()` (sync and async, plus the top-level `aimu.chat()` / `aio.chat()` helpers) accept a `thinking=` argument that turns reasoning on or off, and requests an effort level, portably across providers:

```python
thinking=None       # default: today's behavior, byte for byte
thinking=False      # off, and select the model's instruct-mode sampling profile
thinking=True       # on, at the model's own default effort
thinking="low"      # on at low effort; also "medium" and "high"
```

One rule governs resolution: **validate the argument, never the model.** A value outside `None` / `True` / `False` / `"low"` / `"medium"` / `"high"` raises `ValueError` before any request is built. A value that is valid but that the *model* cannot honour (no effort-level control, or a model that always reasons and cannot be turned off) logs a warning, deduplicated per client instance and per situation, and the call proceeds: swapping the model behind a call site never turns working code into an exception. Levels are advisory in the same spirit: `thinking="high"` against a model with no effort-level control still turns reasoning **on** (with a warning that the level itself was ignored) rather than raising or silently doing nothing. `thinking=False` against a model that has nothing to disable is the one case that stays silent, since the statement is already true and it lets one call site serve a mixed fleet of thinking and non-thinking models.

### Per-provider mechanism

| Provider | off (`thinking=False`) | level (`"low"`/`"medium"`/`"high"`) |
| --- | --- | --- |
| Ollama native | `think=False` | `think="low"/"medium"/"high"` (its SDK already accepts this exact vocabulary) |
| OpenAI-compat local servers (vLLM, SGLang, LM Studio, Ollama-OpenAI, oMLX, HF-Serve, llama-server) | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | `reasoning_effort`, with `"high"` sent as Qwen's own `"xhigh"` |
| HuggingFace (in-process) | `enable_thinking=False` template kwarg | `reasoning_effort` template kwarg |
| Anthropic | omits the `thinking` request parameter | `budget_tokens`: low 2048, medium 8000, high 16000 (`ThinkingStyle.ADAPTIVE` models warn and ignore a level, since that request shape carries no budget parameter) |
| llama.cpp, OpenAI cloud, Gemini | nothing emitted | nothing emitted |

Among the providers that share Qwen's effort vocabulary (Ollama, the OpenAI-compatible family, HuggingFace), only Qwen 3.8 declares `thinking_levels=True` today; Anthropic's six models declare it too, mapping a level to `budget_tokens` instead (see the table above). `GEMINI_2_5_PRO` is the only model with `thinking_optional=False`, recording that Google's own API will not let this specific model disable reasoning at all; `thinking=False` against it warns and the call proceeds at full reasoning effort anyway, billed as such, and `client.last_usage` is how to observe that after the fact.

The last table row is a scope boundary, not an absence of any mechanism, for two of its three providers. OpenAI's o-series models never declare `thinking=True` in AIMU's catalog, so there is nothing for this parameter to reach on that path. Gemini is different: Google's OpenAI-compatible endpoint for Gemini does document a `reasoning_effort` field (`minimal`/`low`/`medium`/`high`, plus `none` to disable reasoning on `gemini-2.5-flash`), so a real mechanism exists there. AIMU emits nothing to it only because that vocabulary excludes the `"xhigh"` value the shared Qwen mapping sends for `"high"`; supporting Gemini correctly needs a second, provider-specific effort vocabulary, which is deferred rather than built here. Until then, `thinking=False` warns and is a no-op against **every** Gemini thinking model, not only `GEMINI_2_5_PRO`: the wire mechanism for the off-request is missing for the whole provider, and `GEMINI_2_5_PRO`'s `thinking_optional=False` is a stronger, independent fact on top of that gap (it could not disable reasoning even once the off-request mechanism exists). `llama.cpp` has no mechanism on either axis: neither the request API nor GGUF metadata carries an effort-level field.

See [Control thinking effort](../how-to/control-thinking.md) for a runnable walkthrough, including the Anthropic `thinking_budget_tokens` escape hatch for exact token control.

## How this compares to other frameworks

For **plain multi-turn chat**, this matches what litellm and similar frameworks do for the same models: prior-turn reasoning is not resent.

There is one nuance, and it is specifically about **tool calling**, not plain chat. Some *API* reasoning models — Anthropic with extended thinking, newer DeepSeek, Moonshot Kimi K2 — *require* the reasoning block to be preserved on the assistant message **between a tool call and the tool result**, or the API returns a 400. litellm carries `reasoning_content` / `thinking_blocks` through for those providers for exactly this reason.

That requirement does **not** apply to the local models AIMU ships:

- It is an API-side constraint (a remote 400). Local HuggingFace / llama-cpp inference re-renders from a chat template that, for Qwen3 and DeepSeek-R1, strips prior reasoning by design.
- AIMU's local thinking catalog is the Qwen3 family and DeepSeek-R1 distillations — both follow the "strip from history" convention.
- The preserve-during-tool-use requirement for Anthropic is handled inside `AnthropicClient`'s native-thinking path, not the local `<think>`-tag path.

## Summary

| Question | Answer |
| --- | --- |
| Is reasoning stored in `messages`? | Yes — under a `"thinking"` key, uniformly across providers; omitted when there is none. |
| Is that key sent to the model? | No — chat templates and request adapters ignore it. |
| Is prior-turn reasoning re-fed on the next turn? | No. |
| Is that correct? | Yes — it matches Qwen3 / Gemma / DeepSeek-R1 guidance and the chat templates' own behavior. For Qwen 3.8, whose card defaults the other way (`preserve_thinking=True`), it is a deliberate deviation instead; see the note above. |
| Where do I read the latest reasoning? | `client.last_thinking`. |
| How do I turn reasoning on or off, or set an effort level? | `thinking=` on `chat()` / `generate()`; see "Controlling thinking" above and the [how-to](../how-to/control-thinking.md). |

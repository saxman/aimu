# Read long web pages

`get_web_content` returns a page's first 20,000 characters, whatever the agent is looking for. An agent
that reads several pages one after another fills its context window with whole pages, most of which
has nothing to do with its question, and stays full: every tool result is resent on every later turn.

`make_page_tools()` returns three tools that keep the page out of the conversation and let in only
what the question needs:

- `fetch_page(url, question)` fetches the page once, stores it, and returns its outline plus the
  passages that best match the question, verbatim, each with the character offset it starts at.
- `read_page(handle, offset)` reads any part of a stored page, without fetching it again.
- `find_in_page(handle, text)` lists the lines of a stored page that contain some text.

```python
import aimu
from aimu.agents import Agent
from aimu.tools import builtin
from aimu.tools.builtin import make_page_tools

agent = Agent(
    aimu.client("ollama:qwen3:8b"),
    "You research questions on the web.",
    tools=[builtin.web_search, *make_page_tools()],
)
agent.run("How long can ochre sea stars survive out of water?")
```

## What enters the context

This is what `fetch_page` returned for a short field guide, asked
`question="how long can sea stars survive out of water?"`, with `make_page_tools(top_k=2)`:

```text
Page page-29b5c273cc20 (https://example.org/tide-pools): 496 characters.

Outline:
- # Field guide to tide pools (offset 1, 94 chars)
- ## Zones (offset 95, 169 chars)
- ## Residents (offset 264, 159 chars)
- ## Visiting (offset 423, 74 chars)

Passages most relevant to the question (2 of 4, in page order):

[offset 1, under "Field guide to tide pools"]
# Field guide to tide pools

Tide pools form where rock holds seawater as the tide goes out.

[offset 264, under "Residents"]
## Residents

Sea anemones close up when exposed to air to keep from drying out.

Ochre sea stars prey on mussels and can survive several hours out of water.

Read more with read_page("page-29b5c273cc20", offset=N); search with find_in_page("page-29b5c273cc20", "text").
```

Four things to notice:

- **Everything after the header is the page's own text.** Nothing was summarized or reworded, so
  whatever the agent quotes, the page said.
- **The first passage matched on the word "out" alone.** The default ranker is BM25, which scores
  shared words and knows nothing about meaning. It ignores common English words ("the", "is",
  "of"; the list is `aimu.rag.bm25.STOPWORDS`), so a passage sharing only those with the question
  is never shown, but any other shared word counts. Passages appear in page order, not score order,
  so the strong match here is the second one.
- **The outline and offsets say what was left out.** The *Zones* section did not come back, and the
  outline shows where it is and how long it is.
- **The handle names this copy of the page.** It is a digest of the URL and the text, so fetching
  the same page again returns the same handle, and a page that changed in between gets a new one
  rather than overwriting the copy an earlier answer came from.

Without a `question`, or when no passage shares a word with it (common words aside), `fetch_page` returns the opening of the
page instead (6,000 characters by default) and says which of the two happened.

## Reading past the passages

Ranking matches words, so a passage that answers the question in different words can be missed. The
`fetch_page` description tells the model to look further before concluding a page does not say
something. It has two ways to do that:

```text
read_page("page-29b5c273cc20", offset=95, max_chars=400)   # the Zones section, from the outline
find_in_page("page-29b5c273cc20", "low tide")              # every line that mentions it
```

```text
1 lines match "low tide"
- offset 460: Visit within an hour of low tide, and step only on bare rock.
```

Each offset is where the match itself starts, so `read_page` at that offset begins at the match even
on a page that is one long line. `find_in_page` matches literally and ignores case; it does not take
regular expressions.

!!! note "Small models and navigation"
    Whether an agent uses `read_page` and `find_in_page` well depends on the model. On the
    BrowseComp-Plus benchmark (arXiv 2508.06600), giving gpt-4.1 a full-document tool alongside short
    previews raised its accuracy from 35% to 44%, while Qwen3-32B barely called the tool (10% to 12%).
    The ranked passages are the part that does not depend on the model choosing to look further.

## Keeping pages between runs

By default pages live in memory for as long as the tools do. Pass `directory=` to keep them as plain
files instead: `<handle>.md` holds the text exactly, and `<handle>.json` holds the URL and fetch time.

```python
tools = make_page_tools(directory=aimu.paths.output / "pages")
```

Nothing is ever deleted from that directory; prune it yourself.

## Ranking with embeddings

`ranker=` replaces BM25 with any function that takes the question and the passages and returns one
score per passage, higher meaning more relevant and 0 meaning no match. An embedding model ranks
paraphrases that share no words with the question, which BM25 cannot:

```python
import math

import aimu
from aimu.tools.builtin import make_page_tools

embedder = aimu.embedding_client("ollama:nomic-embed-text")


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))


def embedding_ranker(question: str, passages: list[str]) -> list[float]:
    query = embedder.embed(question, input_type="query")
    vectors = embedder.embed(passages, input_type="document")
    return [max(0.0, cosine(query, vector)) for vector in vectors]


tools = make_page_tools(ranker=embedding_ranker)
```

Clamping at 0 keeps "no match" meaning what `fetch_page` expects. An embedding ranker almost never
scores a passage at exactly 0, so with one `fetch_page` nearly always returns `top_k` passages rather
than falling back to the opening.

## Tuning

| Argument | Default | What it controls |
|---|---|---|
| `top_k` | 5 | The most passages `fetch_page` returns |
| `chunk_chars` | 1200 | The longest passage. Passages never span two sections. |
| `opening_chars` | 6000 | How much `fetch_page` returns with no question, or when nothing matches |

## See also

- [Manage context](manage-context.md): trimming and summarizing a conversation that is already long.
- [Fetch HTML and submit web forms](browse-and-submit-forms.md): raw markup and forms.
- Issues [#12](https://github.com/saxman/aimu/issues/12) (keeping tool results from bloating context)
  and [#14](https://github.com/saxman/aimu/issues/14) (a reader-model variant with checked quotes).

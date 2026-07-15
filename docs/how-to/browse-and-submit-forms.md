# Fetch HTML and submit web forms

The `builtin.web` tools include `get_webpage` (readable text) and `web_search`, but sometimes an
agent needs to see a page's **raw markup**, discover its **forms**, and **submit** them (a site
search, a login, a data entry). Two additions cover this:

- `get_webpage_html(url)` — a stateless tool returning the page's raw HTML.
- `make_web_tools()` — a factory returning `find_forms` and `submit_form`, which share a
  `requests.Session` so cookies (and thus logins and CSRF tokens) persist across calls.

!!! note "Server-rendered HTML only"
    These fetch HTML over `requests`; they do **not** execute JavaScript. Pages built client-side
    (SPAs) return their pre-render markup, and anti-bot-protected sites (Cloudflare, captchas) will
    be blocked. A headless-browser backend is a possible future addition.

## Read raw HTML

`get_webpage_html` sits in `builtin.web` alongside `get_webpage`. Use `get_webpage` when you want
clean article text; use `get_webpage_html` when you need the markup itself (links, attributes, form
structure). Long pages are truncated to stay within the model's context window.

```python
import aimu
from aimu.tools import builtin

agent = aimu.agents.Agent(aimu.client("ollama:qwen3:8b"), tools=builtin.web)
agent.run("Fetch the HTML of https://example.com and list every <a> href you find.")
```

## Discover and submit forms

`make_web_tools()` returns two tools bound to a shared session. Pass them alongside
`get_webpage_html`:

```python
import aimu
from aimu.tools.builtin import get_webpage_html, make_web_tools

agent = aimu.agents.Agent(
    aimu.client("anthropic:claude-sonnet-4-6"),
    tools=[get_webpage_html, *make_web_tools()],
)
agent.run("On https://httpbin.org/forms/post, submit the form with custname='Ada' and comments='hi'.")
```

- **`find_forms(url)`** fetches the page and lists each form's index, submission URL (resolved to an
  absolute URL), HTTP method, and fields — including `type=hidden` fields, so **CSRF tokens surface**
  for the agent to echo back.
- **`submit_form(url, method="POST", data=None)`** submits `data` to a URL. `method="POST"` sends it
  as a form body; `method="GET"` sends it as query parameters. It returns the response status, the
  final URL after redirects, and the (truncated) response body.

Because both tools close over one `requests.Session`, cookies set during one call are sent on the
next. A login flow works end to end: `find_forms` scrapes the login form's hidden token,
`submit_form` posts credentials (the server sets a session cookie), and a later
`submit_form(url, method="GET")` reads a page behind the login using that cookie.

### Control the session

Pass your own session to set auth headers up front or to share one across several tool sets:

```python
import requests

session = requests.Session()
session.headers["Authorization"] = "Bearer …"
tools = make_web_tools(session=session, timeout=30, max_content_chars=40000)
```

## Confirm before submitting

`submit_form` performs writes (POST). To require confirmation before it runs, gate it with the
[tool-approval hook](gate-tool-calls.md) — the policy sees the tool name, so you can approve reads
and prompt on submits:

```python
def confirm_submits(name, arguments):
    if name != "submit_form":
        return True
    return input(f"Submit to {arguments.get('url')}? [y/N] ").strip().lower() == "y"

agent = aimu.agents.Agent(client, tools=[get_webpage_html, *make_web_tools()],
                          tool_approval=confirm_submits)
```

## Async

Both are re-exported from `aimu.aio.tools.builtin`; the async agent dispatches these sync
`requests` tools via `asyncio.to_thread`:

```python
from aimu import aio
from aimu.aio.tools.builtin import get_webpage_html, make_web_tools

agent = aio.Agent(aio.client("ollama:qwen3:8b"), tools=[get_webpage_html, *make_web_tools()])
await agent.run("…")
```

## See also

- [Add a custom tool](add-custom-tool.md): write your own `@tool`
- [Gate tool calls](gate-tool-calls.md): confirm or block the mutating `submit_form`
- [Use MCP tools](use-mcp-tools.md): cross-process tool servers

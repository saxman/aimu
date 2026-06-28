"""A web front end for the reference personal assistant.

A minimal Starlette + WebSocket server (mirroring aimu.agents.a2a.server's uvicorn/Starlette
pattern) that serves a static chat page and bridges one browser onto a per-connection Assistant
session via WebChannel. Async-native, so scheduler-pushed proactive messages reach the browser
unprompted. Requires the ``web`` extra::

    pip install aimu[web]
    python examples/personal-assistant/web_assistant.py --model ollama:qwen3:8b --reminder-seconds 20

Then open http://127.0.0.1:8000 and chat. After ~20s a proactive message appears unprompted.

Single user by design: one Assistant session per connection, sharing one history.json / skills
dir. A second simultaneous connection is rejected (this is a single-process example).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from _assistant_common import Assistant, AssistantConfig
from assistant import build_arg_parser as _cli_arg_parser, config_from_args
from web_channel import WebChannel

_STATIC = Path(__file__).resolve().parent / "static"


def build_app(config: AssistantConfig, *, client=None) -> Starlette:
    """Build the Starlette app serving the chat page (``/``) and the WebSocket (``/ws``).

    ``client`` injects a model client (tests pass a mock); production leaves it None so each
    connection builds its own via ``Assistant.create``.
    """
    busy = {"active": False}  # one-active-connection guard (single user, single process)

    async def index(request):
        return FileResponse(_STATIC / "index.html")

    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        if busy["active"]:
            await websocket.send_json(
                {"type": "message", "text": "Assistant is busy in another tab.", "proactive": False}
            )
            await websocket.close()
            return
        busy["active"] = True
        channel = WebChannel(websocket)
        assistant = await Assistant.create(config, channel, client=client)

        async def pump() -> None:
            # Feed inbound frames to the channel; on disconnect, the sentinel ends receive(),
            # which stops the scheduler and lets assistant.run() (and this group) return.
            try:
                while True:
                    await channel.feed(await websocket.receive_text())
            except WebSocketDisconnect:
                pass
            finally:
                await channel.feed(None)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(pump())
                tg.create_task(assistant.run())
        finally:
            busy["active"] = False
            await channel.aclose()

    return Starlette(routes=[Route("/", index), WebSocketRoute("/ws", ws_endpoint)])


def serve(config: AssistantConfig, *, host: str = "127.0.0.1", port: int = 8000, **uvicorn_kwargs) -> None:
    import uvicorn

    uvicorn.run(build_app(config), host=host, port=port, **uvicorn_kwargs)


def build_arg_parser(prog: str = "web_assistant.py") -> argparse.ArgumentParser:
    parser = _cli_arg_parser(prog)  # reuse --model/--system/--skills-dir/--history/--reminder-*
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Bind port. Default: 8000")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    serve(config_from_args(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

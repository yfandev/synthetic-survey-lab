"""A local OpenAI-compatible server with scripted replies, for end-to-end tests of the CLI and RPC."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL = "test-model"
# The quiet System 1's pick among a choice's options, first found: not stuck, no risk, no read kept, no page, nothing
# new, so a test's System 2 script runs as it would with nothing judging it.
QUIET = ("advancing", "routine", "reference", "none", "new", "failed")


def quiet(questions: dict[str, Any]) -> dict[str, Any]:
    """`/systemone` answers on the wire: each choice its quiet option, each score the lowest level, each noul 0."""
    answers: dict[str, Any] = {}
    for name, question in questions.items():
        if question["type"] == "choice":
            options = list(question["criteria"])
            pick = next((o for o in QUIET if o in options), options[0])
            answers[name] = {"type": "choice", "choice": pick, "probabilities": {o: float(o == pick) for o in options}}
        elif question["type"] == "score":
            answers[name] = {"type": "score", "score": 0}
        else:
            answers[name] = {"type": "noul", "noul": 0.0}
    return answers


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


class FakeOpenAI:
    def __init__(self, replies: list[Reply], context_length: int = 40_960) -> None:
        self.replies = list(replies)
        self.context_length = context_length
        self.requests: list[dict[str, Any]] = []
        self.system1: list[dict[str, Any]] = []
        """The `/systemone` asks, apart from System 2's requests."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/v1/models":
                    self._json({"data": [{"id": MODEL, "max_model_len": server.context_length}]})
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.endswith("/systemone"):
                    server.system1.append(body)
                    self._json({"answers": quiet(body["questions"]), "usage": {}})
                    return
                server.requests.append(body)
                reply = server.replies.pop(0) if server.replies else Reply(text="done")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in server.chunks(reply, body):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def _json(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def system1_env(self) -> dict[str, str]:
        """The environment that makes this server the console's System 1 for a harness process."""
        return {"DREX_BASE_URL": self.url.removesuffix("/v1"), "DREX_API_KEY": "test-key"}

    def chunks(self, reply: Reply, body: dict[str, Any]) -> list[dict[str, Any]]:
        def choice(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        out = [choice({"content": word + " "}) for word in reply.text.split()]
        for index, (name, arguments) in enumerate(reply.tool_calls):
            out.append(
                choice(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": f"call_{len(self.requests)}_{index}",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ]
                    }
                )
            )
        out.append(choice({}, "tool_calls" if reply.tool_calls else "stop"))
        prompt_tokens = len(json.dumps(body["messages"])) // 4
        out.append({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 12}})
        return out

    def __enter__(self) -> FakeOpenAI:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

# -*- coding: utf-8 -*-
"""Anthropic Messages API (``/v1/messages``) shim in front of llama.cpp.

llama-server only speaks the OpenAI dialect, so this module accepts Anthropic
requests, rewrites them to ``/v1/chat/completions`` on the local llama-server and
translates the answer — including SSE streaming and tool calls — back into the
Anthropic Messages shape. Point ``ANTHROPIC_BASE_URL`` at ``http://host:11435``.
"""

import json
import random
import string
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STOP_MAP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "end_turn"}


def _msg_id():
    return "msg_" + "".join(random.choice(string.ascii_letters + string.digits) for _ in range(24))


def _blocks_to_text(content):
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "".join(out)


def to_openai(body, alias):
    """Anthropic Messages request -> OpenAI chat completions request."""
    messages = []

    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _blocks_to_text(system)})

    for msg in body.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text, tool_calls = [], []
            for block in content or []:
                btype = block.get("type")
                if btype == "text":
                    text.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id") or "toolu_" + _msg_id()[4:],
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                        },
                    })
            out = {"role": "assistant", "content": "".join(text)}
            if tool_calls:
                out["tool_calls"] = tool_calls
            messages.append(out)
            continue

        # user turn: text / images / tool results
        parts, tool_results = [], []
        for block in content or []:
            btype = block.get("type")
            if btype == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image":
                src = block.get("source") or {}
                if src.get("type") == "base64":
                    url = f"data:{src.get('media_type','image/png')};base64,{src.get('data','')}"
                else:
                    url = src.get("url", "")
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif btype == "tool_result":
                tool_results.append(block)
        for tr in tool_results:
            raw = tr.get("content")
            text = _blocks_to_text(raw) if not isinstance(raw, str) else raw
            messages.append({"role": "tool", "tool_call_id": tr.get("tool_use_id", ""), "content": text})
        if parts:
            if len(parts) == 1 and parts[0]["type"] == "text":
                messages.append({"role": "user", "content": parts[0]["text"]})
            else:
                messages.append({"role": "user", "content": parts})

    params = {
        "model": body.get("model") or alias,
        "messages": messages,
        "stream": bool(body.get("stream")),
    }
    for src, dst in (("max_tokens", "max_tokens"), ("temperature", "temperature"),
                     ("top_p", "top_p"), ("stop_sequences", "stop")):
        if body.get(src) is not None:
            params[dst] = body[src]

    tools = []
    for tool in body.get("tools") or []:
        if tool.get("type", "custom") == "custom" or "input_schema" in tool:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            })
    if tools:
        params["tools"] = tools

    # Claude-style thinking knob -> the Qwen template's enable_thinking / reasoning_effort
    thinking = body.get("thinking") or {}
    if thinking.get("type") == "enabled":
        budget = int(thinking.get("budget_tokens") or 4096)
        effort = "low" if budget < 2048 else ("medium" if budget < 8192 else "high")
        params["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": effort}
    else:
        params["chat_template_kwargs"] = {"enable_thinking": False}
    return params


def from_openai(payload, model_name, include_thinking=True):
    """OpenAI chat completion -> Anthropic message."""
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = []
    reasoning = message.get("reasoning_content") or ""
    if reasoning and include_thinking:
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        content.append({"type": "tool_use", "id": call.get("id") or "toolu_" + _msg_id()[4:],
                        "name": fn.get("name", ""), "input": args})
    if not content:
        content.append({"type": "text", "text": ""})

    usage = payload.get("usage") or {}
    return {
        "id": _msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": content,
        "stop_reason": STOP_MAP.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


class AnthropicProxy:
    def __init__(self, upstream_host, upstream_port, alias, api_key="", include_thinking=True):
        self.upstream = f"http://{upstream_host}:{upstream_port}/v1/chat/completions"
        self.alias = alias
        self.api_key = api_key
        self.include_thinking = include_thinking

    # -- upstream ----------------------------------------------------------
    def _call(self, payload, stream=False):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(self.upstream, data=data, headers=headers)
        return urllib.request.urlopen(req, timeout=3600)

    def complete(self, payload, include_thinking=False):
        with self._call(payload, False) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return from_openai(body, payload.get("model") or self.alias, include_thinking)

    def stream(self, payload, include_thinking=False):
        """Yield Anthropic SSE events for an OpenAI streaming response."""
        payload = dict(payload, stream=True)
        out = []

        def emit(event, data):
            out.append(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

        emit("message_start", {
            "type": "message_start",
            "message": {
                "id": _msg_id(), "type": "message", "role": "assistant",
                "model": payload.get("model") or self.alias, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        text_index = None
        text_open = False
        tool_index = {}          # openai tool index -> (content block index, id, name)
        next_index = 0
        stop_reason = "end_turn"
        usage = {"input_tokens": 0, "output_tokens": 0}
        thinking_index = None

        def close_blocks():
            nonlocal text_open, thinking_index
            if thinking_index is not None:
                emit("content_block_stop", {"type": "content_block_stop", "index": thinking_index})
                thinking_index = None
            if text_open:
                emit("content_block_stop", {"type": "content_block_stop", "index": text_index})
                text_open = False
            for idx, (_bi, _id, _name) in sorted(tool_index.items()):
                emit("content_block_stop", {"type": "content_block_stop", "index": _bi})

        with self._call(payload, True) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except ValueError:
                    continue
                if data.get("usage"):
                    usage = {
                        "input_tokens": data["usage"].get("prompt_tokens", usage["input_tokens"]),
                        "output_tokens": data["usage"].get("completion_tokens", usage["output_tokens"]),
                    }
                for choice in data.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        stop_reason = STOP_MAP.get(choice["finish_reason"], "end_turn")

                    reasoning = delta.get("reasoning_content")
                    if reasoning and include_thinking:
                        if thinking_index is None:
                            thinking_index = next_index
                            next_index += 1
                            emit("content_block_start", {"type": "content_block_start", "index": thinking_index,
                                                         "content_block": {"type": "thinking", "thinking": ""}})
                        emit("content_block_delta", {"type": "content_block_delta", "index": thinking_index,
                                                     "delta": {"type": "thinking_delta", "thinking": reasoning}})

                    text = delta.get("content")
                    if text:
                        if text_index is None:
                            text_index = next_index
                            next_index += 1
                        if not text_open:
                            text_open = True
                            emit("content_block_start", {"type": "content_block_start", "index": text_index,
                                                         "content_block": {"type": "text", "text": ""}})
                        emit("content_block_delta", {"type": "content_block_delta", "index": text_index,
                                                     "delta": {"type": "text_delta", "text": text}})

                    for call in delta.get("tool_calls") or []:
                        oi = call.get("index", 0)
                        fn = call.get("function") or {}
                        if oi not in tool_index:
                            bi = next_index
                            next_index += 1
                            tool_index[oi] = (bi, call.get("id") or "toolu_" + _msg_id()[4:], fn.get("name", ""))
                            emit("content_block_start", {"type": "content_block_start", "index": bi,
                                                         "content_block": {"type": "tool_use",
                                                                           "id": tool_index[oi][1],
                                                                           "name": fn.get("name", ""),
                                                                           "input": {}}})
                        elif fn.get("name") and not tool_index[oi][2]:
                            tool_index[oi] = (tool_index[oi][0], tool_index[oi][1], fn["name"])
                        args = fn.get("arguments")
                        if args:
                            emit("content_block_delta", {"type": "content_block_delta", "index": tool_index[oi][0],
                                                         "delta": {"type": "input_json_delta", "partial_json": args}})

        close_blocks()
        emit("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                               "usage": usage})
        emit("message_stop", {"type": "message_stop"})
        return out


def serve(proxy, host, port):
    """Start the proxy on its own thread; returns the server object."""
    handler = _make_handler(proxy)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _make_handler(proxy):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HertaAnthropic/1.0"

        def log_message(self, *_a):
            pass

        def _json(self, code, payload):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _error(self, code, etype, message):
            self._json(code, {"type": "error", "error": {"type": etype, "message": message}})

        def _authorised(self):
            if not proxy.api_key:
                return True
            key = self.headers.get("x-api-key") or ""
            auth = self.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                key = key or auth[7:].strip()
            return key == proxy.api_key

        def do_GET(self):
            path = self.path.split("?")[0]
            if not self._authorised():
                return self._error(401, "authentication_error", "invalid api key")
            if path in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok"})
            if path == "/v1/models":
                return self._json(200, {"object": "list", "data": [
                    {"id": proxy.alias, "type": "model", "object": "model", "display_name": proxy.alias}]})
            self._error(404, "not_found_error", "unknown path")

        def do_POST(self):
            path = self.path.split("?")[0]
            if not self._authorised():
                return self._error(401, "authentication_error", "invalid api key")
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8-sig")) if length else {}
            except Exception as exc:
                return self._error(400, "invalid_request_error", f"bad json: {exc}")

            if path == "/v1/messages/count_tokens":
                text = _blocks_to_text(body.get("messages") or [])
                return self._json(200, {"input_tokens": max(1, int(len(text) / 3.5))})

            if path != "/v1/messages":
                return self._error(404, "not_found_error", "unknown path")

            payload = to_openai(body, proxy.alias)
            try:
                want_thinking = (body.get("thinking") or {}).get("type") == "enabled"
                if body.get("stream"):
                    events = proxy.stream(payload, want_thinking)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for chunk in events:
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                    return
                self._json(200, proxy.complete(payload, want_thinking))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                self._error(502, "api_error", f"upstream {exc.code}: {detail}")
            except Exception as exc:
                self._error(500, "api_error", str(exc))

    return Handler

# Herta Station

A single-file Windows app that starts/stops a local **llama.cpp** server and shows its state in a Herta Space Station styled window — VRAM, PID, context, KV cache type, throughput, and the live `llama-server` log.

The model API it exposes is plain **OpenAI-compatible** (`/v1/chat/completions`), so any agent/client can point at it — locally or over the LAN — while the GUI remains the only place that can start or stop the server.

![Herta Station](docs/screenshot.png)

## Why this exists

Running a 27B model on a 16 GB card through Ollama left ~30 % of the layers on the CPU. Swapping to `llama-server` with a manual GGUF, a tuned context size and a quantized KV cache put everything on the GPU — and made the "is it loaded, how much VRAM is it eating, kill it now" loop a two-button GUI instead of a terminal.

## Features

- **Start / stop** the model with one click; stopping really frees VRAM.
- **Purge** button also unloads every Ollama model so the two never fight over memory.
- **Self-test** sends one tiny chat request and reports latency + reply.
- **Telemetry**: model file, quant, context, GPU layers, KV cache type, PID, uptime, tokens/s, VRAM bar.
- **Live log** of `llama-server` with warning/error colouring.
- **OpenAI-compatible endpoint** on `0.0.0.0:11433` (LAN reachable, optional bearer key); the control plane stays on `127.0.0.1:8899`.
- Ships as a **12 MB one-file EXE** (PyInstaller + pywebview/WebView2) — no Python, no browser tab.

![LAN endpoint row](docs/lan.png)

## Requirements

- Windows 10/11
- [WebView2 runtime](https://developer.microsoft.com/microsoft-edge/webview2/) (preinstalled on current Windows)
- A `llama-server.exe` build (CUDA/Vulkan/CPU) and a GGUF model
- For building: Python 3.11+, `pip install -r requirements.txt`

## Quick start

```bat
set HERTA_LLAMA_EXE=C:\llama.cpp\llama-server.exe
set HERTA_MODEL=D:\models\Qwen3.8-27B-UD-Q3_K_XL.gguf
set HERTA_ALIAS=local-model
python herta_app.py
```

Or build the EXE:

```powershell
.\build.ps1
```

and run `dist\HertaStation.exe`.

The EXE reads the same environment variables; anything unset falls back to the defaults in `server.py`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `HERTA_LLAMA_EXE` | `D:\llama.cpp\b11756\llama-server.exe` | `llama-server.exe` path |
| `HERTA_MODEL` | `D:\ollama-gguf\qwen3.8-27b\Qwen3.8-27B-UD-Q3_K_XL.gguf` | GGUF weights |
| `HERTA_MMPROJ_PATH` | `…\mmproj-F16.gguf` | vision projector file |
| `HERTA_MMPROJ` | `0` | `1` loads the projector (costs ~0.9 GB VRAM) |
| `HERTA_ALIAS` | `qwen3.8-27b-q3xl` | model name reported by `/v1/models` |
| `HERTA_CTX` | `49152` | context length |
| `HERTA_PARALLEL` | `1` | parallel slots (`-np`) |
| `HERTA_FA` | `on` | flash attention |
| `HERTA_CTK` / `HERTA_CTV` | `q5_1` | KV cache types |
| `HERTA_BIND` | `0.0.0.0` | listen address (`127.0.0.1` = local only) |
| `HERTA_PORT` | `11433` | model API port |
| `HERTA_API_KEY` | empty | when set, clients must send `Authorization: Bearer …` |

## API

```
Base URL : http://127.0.0.1:11433/v1      (or http://<lan-ip>:11433/v1)
Model    : value of HERTA_ALIAS
```

```bash
curl http://127.0.0.1:11433/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b-q3xl","messages":[{"role":"user","content":"hi"}]}'
```

Endpoints come from `llama-server`: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/health`, `/props`, `/slots`, plus llama.cpp's own web UI at `/`.
There is no Anthropic `/v1/messages` route.

Thinking models usually put reasoning in `message.reasoning_content`; pass `"chat_template_kwargs": {"enable_thinking": false}` to get a plain answer only.

When you expose the endpoint on a LAN, only the model port is reachable — 8899 (start/stop) stays bound to loopback.

## The 16 GB VRAM cliff (RTX 5080, Qwen3.8-27B Q3_K_XL)

`llama-server` silently collapses to ~1/10 speed once total VRAM passes ≈15.3 GB on a 16 GB card — unrelated to parameter count or KV precision:

| ctx | KV | VRAM | generation |
| ---: | --- | ---: | ---: |
| 8192 | q8_0 | 14.8 GB | 44 tok/s |
| 24576 | q8_0 | 15.5 GB | 42 tok/s |
| 32768 | q8_0 | 15.4 GB | 19.7 tok/s |
| 40960 | q8_0 | 15.6 GB | 12.9 tok/s |
| 49152 | q8_0 | 15.7 GB | **3.1 tok/s** |
| 49152 | **q5_1** | **15.25 GB** | **47 tok/s** |
| 49152 | q4_0 | 15.0 GB | 47 tok/s |

That is why the defaults are `-c 49152 -ctk q5_1 -ctv q5_1`, and why the vision projector is off unless asked for.

## Layout

```
herta_app.py          entry point: control server + WebView2 window
server.py             control plane: start/stop/purge/selftest/status + llama-server supervision
ui.html               the interface (single file, no CDN, no build step)
assets/herta.ico      app icon
scripts/make_icon.py  regenerates the icon
scripts/start-server.bat / stop-server.bat   CLI-only equivalents
build.ps1             PyInstaller one-file build
```

## License

MIT — see [LICENSE](LICENSE).

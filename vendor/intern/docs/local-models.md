# Local models with Kapitsa

On a Mac with Apple silicon, the recommended way to run System 2 locally is
[Kapitsa](https://github.com/nace-ai/kapitsa), Nace's inference engine for Qwen and Nemotron. It
decodes faster than Ollama, llama.cpp, LM Studio and MLX on most Mac and model pairs it was measured
on, serves OpenAI's Chat Completions API that the harness speaks, and returns reasoning and tool
calls in the fields the harness reads. Ollama, llama.cpp, LM Studio and vLLM still work as before
([Models](../README.md#models)); on Linux, or a GPU box, use one of those.

Kapitsa needs an M1 or newer and macOS 14 or later.

## 1. Install Kapitsa

```bash
brew install nace-ai/tap/kapitsa
# or: curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash
```

## 2. Serve a model

```bash
kapitsa serve --model 35b --private
```

The first run downloads the model. When it prints `Ready:`, the API is at `http://127.0.0.1:8000/v1`,
with the name the model is served under and its context. `--private` keeps the server to this Mac;
without it Kapitsa listens on every address and takes any key. Leave it running in its own terminal.

Pick the model by the Mac's memory:

| Memory | Model | `--model` | Why |
|---|---|---|---|
| 16-24 GB | Qwen3.5 9B | `9b` | the most capable model for a 16 GB Mac; `4b` replies quickest |
| 32 GB and up | Qwen3.6 35B-A3B | `35b` | mixture of experts, 3B active: quick to reply and to read |
| 32 GB and up | Qwen3.8 27B | `27b` | the strongest coder of the three, slower to write; a target model of the harness |
| 32 GB and up | Nemotron 3.5 Lightning 30B-A3B | `lightning` | the harness's other target model |

`kapitsa serve` with no `--model` asks, and marks the models that do not fit. `kapitsa doctor` shows what
this Mac can hold.

## 3. Add Kapitsa as a provider

```toml
# ~/.intern/config.toml
[providers.kapitsa]
base_url = "http://127.0.0.1:8000/v1"
```

No key is needed. A server started with `--api-key KEY` takes one: add `api_key_env = "KAPITSA_API_KEY"`
and put the key in `~/.intern/.env`.

Without this entry the built-in `vllm` provider reaches Kapitsa too, as both default to port 8000:
`vllm/qwen3.6-35b-a3b` works with no configuration. The entry gives Kapitsa its own name in the
pickers and in `intern models`.

## 4. Run a task

The model's name is the one `Ready:` printed:

```bash
intern --system2 kapitsa/qwen3.6-35b-a3b                     # the terminal UI
intern -p --system2 kapitsa/qwen3.6-35b-a3b "Find why tests/test_parser.py fails and fix it"
```

Or make it the default:

```toml
# ~/.intern/config.toml
system2 = "kapitsa/qwen3.6-35b-a3b"
```

On an M4 Max with the 27B, a task that reads a file, fixes a one-line bug and reads it back took 5 turns
and 48 seconds.

## The working window

The harness runs each model at the smallest of the window its profile names, the window the provider
lists and `--window` ([Working window](usage.md#models)). Kapitsa lists the context it was started
with, so models without their own profile (the 35B, 9B and 4B) need nothing. The packaged profiles for
the 27B (128K) and Lightning (256K) name the window their cloud deployments serve, which is larger than
a Mac usually holds. Pass the context `kapitsa serve` printed:

```bash
kapitsa serve --model 27b --private --context 64k
intern --window 64000 --system2 kapitsa/qwen3.8-27b
```

Without it, a conversation that grows past the server's context is refused with
`context_length_exceeded` before the harness compacts it.

## System 1

Only System 2 runs on Kapitsa here. System 1 is still a decision model from the console (`NACE_API_KEY`)
or the decisions API (`OPENROUTER_API_KEY`), so a local run still needs one of those keys and a network.
Kapitsa can serve the Drex decision models on this Mac with `kapitsa-systemone`
([guide](https://github.com/nace-ai/kapitsa/blob/main/docs/drex-max.md)), but the harness has not been
checked against it yet.

## Switching models

`intern models` and the `/system2` picker list every model in Kapitsa's catalog, downloaded or
not. The server answers with the model it loaded, whatever a request names, unless it was started with
`--allow-switch`: then picking another model loads it, downloading it first if needed. To switch by
hand, stop the server and start it with another `--model`.

## When it does not work

| What you see | What to do |
|---|---|
| connection refused on `127.0.0.1:8000` | `kapitsa serve` is not running, or runs on another `--port`; match `base_url` to it |
| `context_length_exceeded` | pass `--window` with the context `kapitsa serve` printed, or start it with a larger `--context` |
| the first reply is slow | the first request prefills the whole prompt; later turns reuse Kapitsa's prompt cache |
| another app runs out of memory | run a smaller model, or a smaller `--context` |

The engine's log is in `~/.cache/kapitsa/logs/serve-8000.log`. Kapitsa's own
[docs](https://github.com/nace-ai/kapitsa/blob/main/docs/usage.md) cover its options, models and
APIs.

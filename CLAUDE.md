# Development Guide

Internal documentation for sdcpppal development.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 MCP Client                          │
│    (Claude Code, Gemini CLI, Cursor, etc.)          │
└─────────────────────┬───────────────────────────────┘
                      │ MCP Protocol
                      ▼
┌─────────────────────────────────────────────────────┐
│                sdcpppal Server                      │
│                                                     │
│  ┌──────────────────┐  ┌──────────────────────┐    │
│  │ generate_image   │  │ get_capabilities     │    │
│  └────────┬─────────┘  │ list_samplers        │    │
│           │            │ list_schedulers      │    │
│           │            │ list_loras           │    │
│           │            │ get_current_model    │    │
│           │            └──────────┬───────────┘    │
│           ▼                       ▼                 │
│  ┌──────────────────────────────────────┐          │
│  │    Input resolution + sandboxing      │          │
│  │    Output dir resolution (XDG|CWD)    │          │
│  │    Request body construction          │          │
│  │    Async job polling                  │          │
│  │    PNG/JPEG/WEBP file writing         │          │
│  └──────────────────┬───────────────────┘          │
│                     │ httpx                         │
└─────────────────────┼───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│         sd-server (stable-diffusion.cpp)            │
│                                                     │
│  POST /sdcpp/v1/img_gen    → 202 + job_id           │
│  GET  /sdcpp/v1/jobs/{id}  → status snapshot        │
│  GET  /sdcpp/v1/capabilities                        │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
sdcpppal/
├── src/sdcpppal/
│   ├── __init__.py       # Package metadata (__version__)
│   └── server.py         # MCP server + all logic
├── tests/
│   ├── test_server.py         # Unit tests (no sd-server)
│   └── test_connectivity.py   # Manual: live sd-server smoke test
└── pyproject.toml
```

## Key Design Decisions

### Native /sdcpp/v1, not compat shims

sd-server exposes three API families: `/v1/*` (OpenAI-style), `/sdapi/v1/*` (A1111-style), and `/sdcpp/v1/*` (native async). sdcpppal targets **only the native family** — the pal-family principle: wrapping the vendor's native API preserves features (structured LoRA, native sample params, image-edit fields) that compat layers would round-trip-lossy.

### External sd-server lifecycle

sdcpppal does **not** spawn or manage `sd-server`. It points at a running instance via `SDCPP_HOST`. Consequence: swapping the loaded model = restart `sd-server`. Same pattern as ollapal → ollama.

### Single-model server

Unlike Ollama, which can load multiple models on demand, sd-server loads **one** model at startup into VRAM. `get_current_model()` reports what's loaded; sdcpppal has no affordance to switch it from inside MCP.

### Output policy: path-only, XDG default, workspace opt-in

`generate_image` writes images to disk and returns paths. No base64-in-response mode.

- `output_dir=None` → `$XDG_DATA_HOME/sdcpppal/outputs` (default)
- `output_dir="./generated"` (relative) → resolved under CWD so other workspace-scoped MCPs can read the output
- `output_dir="/abs/path"` → used as-is

### Input images: path OR base64 OR data URL

`_resolve_input_image` normalizes to raw base64 (what `/sdcpp/v1` accepts):

- path → sandboxed via `_validate_path`, read, base64-encoded
- `data:image/*;base64,<...>` → strips the prefix
- bare base64 → passed through

### Path sandboxing

`_validate_path` (copied from ollapal) restricts input file paths to the current working directory:

- resolves relative paths under CWD
- rejects paths that escape CWD after resolution
- rejects symlinks whose absolute target is outside CWD

### Optional sampler field semantics

Per `examples/server/api.md` "Optional Field Semantics":

> If a user has not explicitly provided one of these fields, the client should omit it instead of injecting a guessed fallback:
> - `sample_params.scheduler`
> - `sample_params.sample_method`
> - `sample_params.eta`
> - `sample_params.flow_shift`
> - `sample_params.guidance.img_cfg`

`_build_request_body` follows this rule — it only adds a field when the caller passed a non-`None` value. `test_unset_sampler_fields_omitted` and `test_img_cfg_not_synthesized` enforce this.

### Async job polling

`_poll_job` pulls `/sdcpp/v1/jobs/{id}` every 500ms (default). Terminal states: `completed | failed | cancelled`. It reports status transitions via `ctx.info` and elapsed-vs-timeout via `ctx.report_progress` (the server doesn't expose a real denoising-step progress fraction).

### No cancel tool in v0.1

The `cancel_generating` feature flag is typically `false` on sd-server — cancellation only works while a job is still queued. Since `generate_image` is a blocking MCP call, there's no separate client to issue a cancel from, so we skip the tool entirely in v0.1. Add an async-trio (`submit_image_job` / `get_job` / `cancel_job`) later if needed.

### Safety Limits

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_INPUT_IMAGE_SIZE` | 20 MB | Caps file-path input images |
| `DEFAULT_POLL_TIMEOUT` | 600 s | Max wait for a job to reach terminal state |
| `DEFAULT_POLL_INTERVAL` | 500 ms | Time between `/jobs/{id}` pulls |

### FastMCP 3.x

- Built against `fastmcp>=3.0.1`. Uses `timeout=` and `annotations=ToolAnnotations(...)` on `@mcp.tool`.
- `ctx.report_progress`, `ctx.info` are **async** — always `await`.
- Tools return `dict[str, Any]` (paths + meta) rather than `str`.

## Testing

```bash
# Install dev dependencies
uv sync --all-extras

# Unit tests (no sd-server needed)
uv run pytest tests/test_server.py -v

# Live smoke test (requires running sd-server)
uv run python tests/test_connectivity.py
```

Unit tests cover: path sandboxing (incl. symlink escape), output-dir resolution (XDG default / relative→CWD / absolute), input-image resolution (path / base64 / data URL / traversal blocked / missing / garbage rejected), request body assembly (optional-field omission rule, guidance nesting, LoRA passthrough, img_cfg not synthesized), slug safety, image file writing (indexed names, jpeg→jpg, collision suffix), and job polling (completed / failed / timeout) with mocked HTTP.

`test_connectivity.py` honors `SDCPP_HOST` and generates a tiny 256×256 image with low steps, verifying the returned path exists and is a non-empty file.

## Installing sdcpppal cleanly (uv tool)

sdcpppal runs as a `uv tool` — `uv sync` does **not** update what your MCP client invokes.

**To deploy changes:**
1. Bump version in `pyproject.toml` AND `src/sdcpppal/__init__.py`
2. `uv cache clean --force sdcpppal`
3. `uv tool install --force /home/atobey/src/sdcpppal`
4. Verify: `grep <your_change> ~/.local/share/uv/tools/sdcpppal/lib/python3.*/site-packages/sdcpppal/server.py`
5. `/mcp` in Claude Code to reconnect

Without the version bump, uv may serve a stale cached wheel even with `--force`.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SDCPP_HOST` | No | sd-server base URL (default `http://localhost:1234`) |
| `XDG_DATA_HOME` | No | Determines default output dir (default `~/.local/share`) |

## Configuration: `~/.config/sdcpppal/config.toml`

```toml
sdcpp_host = "http://localhost:1234"
output_dir = "~/Pictures/sdcpppal"
```

CLI flags override config file values:

| Option | Description |
|--------|-------------|
| `--sdcpp-host URL` | Override `SDCPP_HOST` env |
| `--output-dir DIR` | Default output directory |
| `--debug` | Debug logging |

## sd-server capabilities gotchas

- `sd-server` is single-model — users restart it to swap models.
- `features.cancel_generating` is usually `false`; only queued jobs can be cancelled. Plan async-trio accordingly if we add one.
- LoRA: `<lora:...>` prompt tags are intentionally unsupported — use the `lora` array.
- Model-specific sampler/scheduler lists: always discover via `/sdcpp/v1/capabilities` rather than shipping a static list.

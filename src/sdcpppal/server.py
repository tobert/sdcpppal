"""
sdcpppal - your pal stable-diffusion.cpp

An MCP server that wraps sd-server's native /sdcpp/v1/* async job API
for local diffusion image generation.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from sdcpppal import __version__

load_dotenv()

READONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)

logger = logging.getLogger("sdcpppal")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SDCPP_HOST = "http://localhost:1234"
DEFAULT_POLL_INTERVAL = 0.5  # seconds
DEFAULT_POLL_TIMEOUT = 600.0  # 10 minutes

MAX_INPUT_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB per input image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# Fields under sample_params (and nested guidance) that MUST be omitted when
# the caller did not pass them — per api.md "Optional Field Semantics".
_UNSET_SAMPLER_FIELDS = {"scheduler", "sample_method", "eta", "flow_shift"}
_UNSET_GUIDANCE_FIELDS = {"img_cfg"}


# ─────────────────────────────────────────────────────────────────────────────
# Path sandboxing (input files)
# ─────────────────────────────────────────────────────────────────────────────

_project_root: Path | None = None


def _validate_path(path: str) -> Path:
    """Ensure `path` stays inside the project root. Blocks .. and symlink escapes."""
    global _project_root
    if _project_root is None:
        _project_root = Path.cwd().resolve()

    target = Path(path)
    if target.is_absolute():
        resolved = target.resolve()
    else:
        resolved = (_project_root / path).resolve()

    try:
        resolved.relative_to(_project_root)
    except ValueError:
        raise ValueError(f"Path '{path}' resolves outside project directory")

    original = _project_root / path if not target.is_absolute() else target
    if original.is_symlink():
        link_target = original.readlink()
        if link_target.is_absolute():
            if not str(link_target.resolve()).startswith(str(_project_root) + os.sep):
                raise ValueError(f"Path '{path}' is a symlink pointing outside project")

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Output directory resolution
# ─────────────────────────────────────────────────────────────────────────────


def _xdg_default_output_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "sdcpppal" / "outputs"


_default_output_dir: Path | None = None


def _configured_default_output() -> Path:
    assert _default_output_dir is not None, "server not initialized"
    return _default_output_dir


def resolve_output_dir(requested: str | None) -> Path:
    """Resolve an output directory for writing images.

    - None → configured default (XDG-based)
    - absolute path → used as-is
    - relative path → resolved under the current workspace (CWD), so MCPs
      sharing a workspace can share generated files

    The directory is created if it does not exist.
    """
    if requested is None:
        out = _configured_default_output()
    else:
        p = Path(os.path.expandvars(os.path.expanduser(requested)))
        if p.is_absolute():
            out = p.resolve()
        else:
            root = _project_root or Path.cwd().resolve()
            out = (root / p).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Input image resolution
# ─────────────────────────────────────────────────────────────────────────────

_DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(?P<data>.+)$", re.DOTALL)
# A loose heuristic: a string that looks like a valid base64 payload.
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/\s]+={0,2}$")


def _resolve_input_image(value: str) -> str:
    """Normalize an image input to a raw base64 string.

    Accepts:
      - a file path (sandboxed to project root, max MAX_INPUT_IMAGE_SIZE)
      - a data URL `data:image/...;base64,<b64>`
      - a raw base64 string

    Returns a raw base64 string (what /sdcpp/v1 accepts).
    """
    if not isinstance(value, str) or not value:
        raise ValueError("image input must be a non-empty string")

    m = _DATA_URL_RE.match(value)
    if m:
        return m.group("data").strip()

    # Heuristic: looks like a path (has a slash, starts with . or /, or has an
    # image extension). Try to treat as path first.
    looks_like_path = (
        "/" in value
        or value.startswith((".", "~", "$"))
        or Path(value).suffix.lower() in IMAGE_EXTS
    )
    if looks_like_path:
        p = _validate_path(value)
        if not p.exists():
            raise ValueError(f"Image '{value}' does not exist")
        if not p.is_file():
            raise ValueError(f"Image '{value}' is not a regular file")
        if p.stat().st_size > MAX_INPUT_IMAGE_SIZE:
            raise ValueError(
                f"Image '{value}' exceeds {MAX_INPUT_IMAGE_SIZE // (1024 * 1024)}MB limit"
            )
        return base64.standard_b64encode(p.read_bytes()).decode("ascii")

    if _BASE64_RE.match(value.replace("\n", "").replace("\r", "")):
        return value.strip()

    raise ValueError(
        f"Could not interpret image input (not a path, data URL, or base64): {value[:40]}..."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Request body construction
# ─────────────────────────────────────────────────────────────────────────────


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None."""
    return {k: v for k, v in d.items() if v is not None}


def _build_request_body(
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    distilled_guidance: float | None = None,
    seed: int | None = None,
    batch_count: int | None = None,
    clip_skip: int | None = None,
    strength: float | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    init_image_b64: str | None = None,
    mask_image_b64: str | None = None,
    control_image_b64: str | None = None,
    ref_images_b64: list[str] | None = None,
    control_strength: float | None = None,
    lora: list[dict[str, Any]] | None = None,
    output_format: str = "png",
    output_compression: int | None = None,
) -> dict[str, Any]:
    """Construct a /sdcpp/v1/img_gen request body.

    Omits optional fields the caller didn't explicitly provide, so the server
    applies its own defaults (per api.md "Optional Field Semantics").
    """
    guidance: dict[str, Any] = {}
    if cfg is not None:
        guidance["txt_cfg"] = cfg
    if distilled_guidance is not None:
        guidance["distilled_guidance"] = distilled_guidance

    sample_params: dict[str, Any] = {}
    if sampler is not None:
        sample_params["sample_method"] = sampler
    if scheduler is not None:
        sample_params["scheduler"] = scheduler
    if steps is not None:
        sample_params["sample_steps"] = steps
    if guidance:
        sample_params["guidance"] = guidance

    body: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "output_format": output_format,
    }
    if width is not None:
        body["width"] = width
    if height is not None:
        body["height"] = height
    if seed is not None:
        body["seed"] = seed
    if batch_count is not None:
        body["batch_count"] = batch_count
    if clip_skip is not None:
        body["clip_skip"] = clip_skip
    if strength is not None:
        body["strength"] = strength
    if control_strength is not None:
        body["control_strength"] = control_strength
    if output_compression is not None:
        body["output_compression"] = output_compression
    if sample_params:
        body["sample_params"] = sample_params
    if init_image_b64:
        body["init_image"] = init_image_b64
    if mask_image_b64:
        body["mask_image"] = mask_image_b64
    if control_image_b64:
        body["control_image"] = control_image_b64
    if ref_images_b64:
        body["ref_images"] = ref_images_b64
    if lora:
        body["lora"] = lora
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────────

_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _slugify(text: str, max_len: int = 40) -> str:
    slug = _SAFE_SLUG_RE.sub("_", text.strip().lower()).strip("_.")
    return (slug[:max_len] or "img").rstrip("_.")


def _write_images(
    images: list[dict[str, Any]],
    output_dir: Path,
    output_format: str,
    prefix: str,
) -> list[Path]:
    """Write b64 images to disk. Returns the list of paths written (in order).

    Filename: {prefix}_{timestamp_ms}_{index:02d}.{ext}
    """
    ext = output_format.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    ts = int(time.time() * 1000)
    written: list[Path] = []
    for img in images:
        idx = int(img.get("index", len(written)))
        b64 = img.get("b64_json")
        if not b64:
            continue
        name = f"{prefix}_{ts}_{idx:02d}.{ext}"
        path = output_dir / name
        # Avoid collision (parallel calls within the same ms).
        n = 1
        while path.exists():
            path = output_dir / f"{prefix}_{ts}_{idx:02d}_{n}.{ext}"
            n += 1
        path.write_bytes(base64.standard_b64decode(b64))
        written.append(path)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# HTTP client
# ─────────────────────────────────────────────────────────────────────────────

_sdcpp_host: str = os.environ.get("SDCPP_HOST") or DEFAULT_SDCPP_HOST
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is None:
            _client = httpx.AsyncClient(
                base_url=_sdcpp_host,
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
        return _client


async def _get_json(path: str) -> dict[str, Any]:
    client = await get_client()
    r = await client.get(path)
    r.raise_for_status()
    return r.json()


async def _post_json(path: str, body: dict[str, Any]) -> dict[str, Any]:
    client = await get_client()
    r = await client.post(path, json=body)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Job polling
# ─────────────────────────────────────────────────────────────────────────────


async def _poll_job(
    job_id: str,
    timeout: float,
    ctx: Context | None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> dict[str, Any]:
    """Poll /sdcpp/v1/jobs/{id} until terminal state or timeout.

    Returns the final job dict. Raises TimeoutError if timeout is exceeded.
    """
    deadline = time.monotonic() + timeout
    terminal = {"completed", "failed", "cancelled"}
    last_status: str | None = None
    start = time.monotonic()
    while True:
        try:
            job = await _get_json(f"/sdcpp/v1/jobs/{job_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 410):
                raise RuntimeError(f"Job {job_id} no longer available: HTTP {e.response.status_code}")
            raise

        status = job.get("status")
        if status != last_status and ctx is not None:
            try:
                elapsed = time.monotonic() - start
                await ctx.info(f"job {job_id}: {status} ({elapsed:.1f}s)")
            except Exception:
                pass
            last_status = status

        if ctx is not None:
            try:
                elapsed = time.monotonic() - start
                # No true progress fraction from the API — report elapsed vs timeout
                await ctx.report_progress(min(elapsed, timeout), timeout)
            except Exception:
                pass

        if status in terminal:
            return job

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Job {job_id} did not reach a terminal state within {timeout:.0f}s "
                f"(last status: {status})"
            )

        await asyncio.sleep(poll_interval)


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("sdcpppal")


@mcp.tool(timeout=DEFAULT_POLL_TIMEOUT + 30.0)
async def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    distilled_guidance: float | None = None,
    seed: int = -1,
    batch_count: int = 1,
    sampler: str | None = None,
    scheduler: str | None = None,
    clip_skip: int | None = None,
    strength: float | None = None,
    init_image: str | None = None,
    mask_image: str | None = None,
    control_image: str | None = None,
    ref_images: list[str] | None = None,
    control_strength: float | None = None,
    lora: list[dict[str, Any]] | None = None,
    output_dir: str | None = None,
    filename_prefix: str | None = None,
    output_format: str = "png",
    output_compression: int | None = None,
    timeout: float = DEFAULT_POLL_TIMEOUT,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Generate image(s) from a prompt via stable-diffusion.cpp.

    Submits a job to the running sd-server, polls until completion, writes
    images to disk, and returns the paths.

    Args:
        prompt: The text prompt to generate from. Required.
        negative_prompt: Things to avoid in the image.
        width, height: Image dimensions. Omit to use the model's default (usually 512 or 1024).
        steps: Sampling steps. Omit to use the server default.
        cfg: Classifier-free guidance scale (txt_cfg). Omit for server default.
        distilled_guidance: Distilled guidance (for FLUX-family models).
        seed: RNG seed; -1 = random.
        batch_count: How many images to generate in one job.
        sampler: Sampler name (e.g. "euler_a", "dpm++2m"). See list_samplers().
        scheduler: Scheduler name (e.g. "karras", "discrete"). See list_schedulers().
        clip_skip: CLIP skip layer count (-1 = model default).
        strength: img2img/inpaint denoising strength (0.0-1.0) when init_image is set.
        init_image: Input image for img2img. File path (sandboxed to CWD) OR
                    base64 string OR data URL (data:image/...;base64,...).
        mask_image: Mask image for inpainting. Same accepted forms as init_image.
        control_image: ControlNet conditioning image.
        ref_images: Reference images (e.g. for Qwen-Image-Edit).
        control_strength: ControlNet strength (default 0.9 server-side).
        lora: List of {path, multiplier, is_high_noise} dicts.
        output_dir: Where to write images. If None → XDG default
                    (~/.local/share/sdcpppal/outputs). Relative paths resolve
                    under the workspace (CWD), so other MCPs sharing the
                    workspace can see generated files.
        filename_prefix: Filename stem. Defaults to a slug of the prompt.
        output_format: "png", "jpeg", or "webp".
        output_compression: 0-100 for jpeg/webp.
        timeout: Max seconds to wait for the job to complete.

    Returns a dict with `paths`, `count`, `job_id`, `status`, `model`, timing,
    and the submitted prompt/seed/dimensions for reproducibility.
    """
    if not prompt or not prompt.strip():
        return {"error": "prompt cannot be empty"}

    # Resolve input images
    def _maybe_resolve(v: str | None) -> str | None:
        return _resolve_input_image(v) if v else None

    try:
        init_b64 = _maybe_resolve(init_image)
        mask_b64 = _maybe_resolve(mask_image)
        control_b64 = _maybe_resolve(control_image)
        refs_b64 = [_resolve_input_image(r) for r in (ref_images or [])]
    except ValueError as e:
        return {"error": f"input image error: {e}"}

    body = _build_request_body(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        distilled_guidance=distilled_guidance,
        seed=seed,
        batch_count=batch_count,
        clip_skip=clip_skip,
        strength=strength,
        sampler=sampler,
        scheduler=scheduler,
        init_image_b64=init_b64,
        mask_image_b64=mask_b64,
        control_image_b64=control_b64,
        ref_images_b64=refs_b64 or None,
        control_strength=control_strength,
        lora=lora,
        output_format=output_format,
        output_compression=output_compression,
    )

    # Resolve output dir eagerly so errors surface before spending GPU time
    try:
        out_dir = resolve_output_dir(output_dir)
    except OSError as e:
        return {"error": f"cannot prepare output dir: {e}"}

    # Submit
    try:
        submit = await _post_json("/sdcpp/v1/img_gen", body)
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:500]
        return {"error": f"submit failed (HTTP {e.response.status_code}): {detail}"}
    except httpx.RequestError as e:
        return {"error": f"cannot reach sd-server at {_sdcpp_host}: {e}"}

    job_id = submit.get("id")
    if not job_id:
        return {"error": f"submission response missing id: {submit}"}

    if ctx is not None:
        try:
            await ctx.info(f"submitted job {job_id}")
        except Exception:
            pass

    # Poll
    try:
        job = await _poll_job(job_id, timeout=timeout, ctx=ctx)
    except TimeoutError as e:
        return {"error": str(e), "job_id": job_id}
    except httpx.RequestError as e:
        return {"error": f"polling failed: {e}", "job_id": job_id}

    status = job.get("status")
    if status != "completed":
        err = job.get("error") or {}
        return {
            "error": err.get("message") or f"job {status}",
            "error_code": err.get("code"),
            "status": status,
            "job_id": job_id,
        }

    result = job.get("result") or {}
    images = result.get("images") or []
    if not images:
        return {"error": "completed job returned no images", "job_id": job_id}

    prefix = filename_prefix or _slugify(prompt)
    actual_format = result.get("output_format") or output_format
    paths = _write_images(images, out_dir, actual_format, prefix)

    duration = None
    if job.get("started") and job.get("completed"):
        duration = float(job["completed"]) - float(job["started"])

    return {
        "paths": [str(p) for p in paths],
        "count": len(paths),
        "job_id": job_id,
        "status": status,
        "output_dir": str(out_dir),
        "output_format": actual_format,
        "prompt": prompt,
        "seed": seed,
        "dimensions": (
            {"width": width, "height": height} if (width or height) else None
        ),
        "duration_seconds": duration,
    }


@mcp.tool(annotations=READONLY, timeout=15.0)
async def get_capabilities() -> dict[str, Any]:
    """Return the server's /sdcpp/v1/capabilities — model, defaults, samplers, schedulers, loras, limits, features."""
    try:
        return await _get_json("/sdcpp/v1/capabilities")
    except httpx.RequestError as e:
        return {"error": f"cannot reach sd-server at {_sdcpp_host}: {e}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}


@mcp.tool(annotations=READONLY, timeout=15.0)
async def get_current_model() -> dict[str, Any]:
    """Return the model currently loaded in sd-server."""
    caps = await get_capabilities()
    if "error" in caps:
        return caps
    return caps.get("model") or {"error": "capabilities response missing 'model'"}


@mcp.tool(annotations=READONLY, timeout=15.0)
async def list_loras() -> dict[str, Any]:
    """List LoRAs available to the current model (names and paths)."""
    caps = await get_capabilities()
    if "error" in caps:
        return caps
    loras = caps.get("loras") or []
    return {"count": len(loras), "loras": loras}


@mcp.tool(annotations=READONLY, timeout=15.0)
async def list_samplers() -> dict[str, Any]:
    """List sampler names supported by the current model."""
    caps = await get_capabilities()
    if "error" in caps:
        return caps
    s = caps.get("samplers") or []
    return {"count": len(s), "samplers": s}


@mcp.tool(annotations=READONLY, timeout=15.0)
async def list_schedulers() -> dict[str, Any]:
    """List scheduler names supported by the current model."""
    caps = await get_capabilities()
    if "error" in caps:
        return caps
    s = caps.get("schedulers") or []
    return {"count": len(s), "schedulers": s}


# ─────────────────────────────────────────────────────────────────────────────
# MCP resources
# ─────────────────────────────────────────────────────────────────────────────


@mcp.resource("resource://server/info")
def server_info() -> dict[str, Any]:
    return {
        "name": "sdcpppal",
        "version": __version__,
        "description": "your pal stable-diffusion.cpp - MCP wrapper for sd-server's native /sdcpp/v1 API",
        "sdcpp_host": _sdcpp_host,
        "default_output_dir": str(_default_output_dir) if _default_output_dir else None,
        "features": [
            "generate_image", "img2img", "inpaint", "control_net",
            "ref_images", "lora", "capabilities_discovery",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Config file
# ─────────────────────────────────────────────────────────────────────────────


def _load_config() -> dict:
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    config_path = Path(config_home) / "sdcpppal" / "config.toml"
    if not config_path.is_file():
        logger.debug("No config file at %s", config_path)
        return {}
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        logger.warning("Invalid TOML in %s: %s", config_path, e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    global _project_root, _sdcpp_host, _default_output_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="sdcpppal - your pal stable-diffusion.cpp")
    parser.add_argument(
        "--sdcpp-host",
        default=None,
        help=f"sd-server base URL (default from SDCPP_HOST env or {DEFAULT_SDCPP_HOST})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default output dir (default: $XDG_DATA_HOME/sdcpppal/outputs)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("sdcpppal").setLevel(logging.DEBUG)

    config = _load_config()

    if args.sdcpp_host:
        _sdcpp_host = args.sdcpp_host
    elif config.get("sdcpp_host"):
        _sdcpp_host = str(config["sdcpp_host"])

    if args.output_dir:
        _default_output_dir = Path(os.path.expandvars(os.path.expanduser(args.output_dir))).resolve()
    elif config.get("output_dir"):
        _default_output_dir = Path(
            os.path.expandvars(os.path.expanduser(str(config["output_dir"])))
        ).resolve()
    else:
        _default_output_dir = _xdg_default_output_dir()

    _default_output_dir.mkdir(parents=True, exist_ok=True)
    _project_root = Path.cwd().resolve()

    logger.info("sd-server host: %s", _sdcpp_host)
    logger.info("default output dir: %s", _default_output_dir)
    logger.info("project root (sandbox): %s", _project_root)

    mcp.run()


if __name__ == "__main__":
    main()

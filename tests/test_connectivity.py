#!/usr/bin/env python3
"""
Live smoke test against a running sd-server.

Requires an sd-server reachable at $SDCPP_HOST (default http://localhost:1234)
with a model loaded. Generates one tiny image with minimal steps, verifies
the file lands on disk, and prints meta.

Usage:
    uv run python tests/test_connectivity.py
    SDCPP_HOST=http://remote:1234 uv run python tests/test_connectivity.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Make the package importable when run as `python tests/test_connectivity.py`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sdcpppal import server as S  # noqa: E402


async def main() -> int:
    host = os.environ.get("SDCPP_HOST", S.DEFAULT_SDCPP_HOST)
    print(f"→ sd-server: {host}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp).resolve()
        S._project_root = tmpdir
        S._default_output_dir = tmpdir / "xdg_default"
        S._sdcpp_host = host
        S._client = None  # force rebuild

        # 1. Capabilities — verify the server responds at all.
        print("→ fetching capabilities...")
        caps = await S.get_capabilities()
        if "error" in caps:
            print(f"✗ capabilities failed: {caps['error']}")
            return 1
        model = caps.get("model", {})
        print(f"  model:       {model.get('name', '?')}")
        print(f"  samplers:    {len(caps.get('samplers') or [])}")
        print(f"  schedulers:  {len(caps.get('schedulers') or [])}")
        print(f"  max_dims:    {caps['limits']['max_width']}x{caps['limits']['max_height']}")

        # 2. Tiny txt2img — just prove the pipeline works. 256x256 @ 4 steps.
        print("\n→ generating 256x256, 4-step test image...")
        result = await S.generate_image(
            prompt="a red circle on a white background",
            width=256,
            height=256,
            steps=4,
            seed=1,
            batch_count=1,
            output_dir=str(tmpdir / "output"),
            filename_prefix="smoke",
            timeout=120.0,
        )

        if "error" in result:
            print(f"✗ generate_image failed: {result}")
            return 1

        paths = result.get("paths") or []
        if not paths:
            print(f"✗ no paths returned: {result}")
            return 1

        for p in paths:
            size = Path(p).stat().st_size
            print(f"  wrote: {p} ({size:,} bytes)")
            if size == 0:
                print("✗ written file is empty")
                return 1

        print(f"  job_id:   {result.get('job_id')}")
        print(f"  duration: {result.get('duration_seconds')}s")
        print(f"  status:   {result.get('status')}")

        print("\n✓ smoke test passed")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

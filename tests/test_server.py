"""Unit tests for sdcpppal — no sd-server required."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest

from sdcpppal import server as S


# ─────────────────────────────────────────────────────────────────────────────
# Path sandboxing
# ─────────────────────────────────────────────────────────────────────────────


class TestValidatePath:
    def setup_method(self, method):
        S._project_root = None

    def test_inside_cwd_relative(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a.png").write_bytes(b"x")
        p = S._validate_path("a.png")
        assert p == (tmp_path / "a.png").resolve()

    def test_traversal_dotdot_blocked(self, tmp_path, monkeypatch):
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError, match="outside project directory"):
            S._validate_path("../secret.png")

    def test_absolute_outside_blocked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="outside project directory"):
            S._validate_path("/etc/passwd")

    def test_symlink_to_absolute_outside_blocked(self, tmp_path, monkeypatch):
        outside = tmp_path.parent / "outside.png"
        outside.write_bytes(b"x")
        try:
            monkeypatch.chdir(tmp_path)
            link = tmp_path / "link.png"
            link.symlink_to(outside)
            with pytest.raises(ValueError):
                S._validate_path("link.png")
        finally:
            outside.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Output directory resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveOutputDir:
    def setup_method(self, method):
        S._project_root = None
        S._default_output_dir = None

    def test_none_returns_configured_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        configured = tmp_path / "xdg_home"
        S._default_output_dir = configured
        out = S.resolve_output_dir(None)
        assert out == configured.resolve()
        assert out.is_dir()

    def test_relative_resolves_under_workspace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        S._default_output_dir = tmp_path / "xdg"
        out = S.resolve_output_dir("./generated")
        assert out == (tmp_path / "generated").resolve()
        assert out.is_dir()

    def test_absolute_used_as_is(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        S._default_output_dir = tmp_path / "xdg"
        target = tmp_path / "absolute_here"
        out = S.resolve_output_dir(str(target))
        assert out == target.resolve()
        assert out.is_dir()

    def test_xdg_default_honors_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        out = S._xdg_default_output_dir()
        assert out == tmp_path / "sdcpppal" / "outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Input image resolution
# ─────────────────────────────────────────────────────────────────────────────


# 1x1 PNG
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c636001000000050001d3a6e8b80000000049454e44ae426082"
)


class TestResolveInputImage:
    def setup_method(self, method):
        S._project_root = None

    def test_path_to_image(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "pic.png").write_bytes(_TINY_PNG)
        out = S._resolve_input_image("pic.png")
        assert base64.standard_b64decode(out) == _TINY_PNG

    def test_data_url(self):
        b64 = base64.standard_b64encode(_TINY_PNG).decode("ascii")
        url = f"data:image/png;base64,{b64}"
        assert S._resolve_input_image(url) == b64

    def test_raw_base64_passthrough(self):
        b64 = base64.standard_b64encode(_TINY_PNG).decode("ascii")
        assert S._resolve_input_image(b64) == b64

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError):
            S._resolve_input_image("../outside.png")

    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            S._resolve_input_image("nope.png")

    def test_garbage_rejected(self):
        with pytest.raises(ValueError):
            S._resolve_input_image("!@#$%not-base64-not-a-path")

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            S._resolve_input_image("")


# ─────────────────────────────────────────────────────────────────────────────
# Request body construction
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildRequestBody:
    def test_minimal(self):
        body = S._build_request_body(prompt="a cat")
        assert body["prompt"] == "a cat"
        assert body["negative_prompt"] == ""
        assert body["output_format"] == "png"
        # Optional fields must NOT be present
        assert "sample_params" not in body
        assert "width" not in body
        assert "seed" not in body
        assert "strength" not in body

    def test_unset_sampler_fields_omitted(self):
        """Per api.md: scheduler, sample_method, eta, flow_shift, img_cfg must be omitted when unset."""
        body = S._build_request_body(prompt="p", steps=20)
        sp = body.get("sample_params", {})
        # Only sample_steps should be present — no scheduler, no sample_method
        assert sp == {"sample_steps": 20}, sp
        assert "scheduler" not in sp
        assert "sample_method" not in sp

    def test_sampler_and_scheduler_passed_through(self):
        body = S._build_request_body(
            prompt="p", sampler="euler_a", scheduler="karras", steps=28
        )
        assert body["sample_params"] == {
            "sample_method": "euler_a",
            "scheduler": "karras",
            "sample_steps": 28,
        }

    def test_cfg_nests_into_guidance(self):
        body = S._build_request_body(prompt="p", cfg=7.5, distilled_guidance=3.5)
        assert body["sample_params"]["guidance"] == {
            "txt_cfg": 7.5,
            "distilled_guidance": 3.5,
        }

    def test_img_cfg_not_synthesized(self):
        """We never insert img_cfg unless caller provided it — server-side default matters."""
        body = S._build_request_body(prompt="p", cfg=7.5)
        guidance = body["sample_params"]["guidance"]
        assert "img_cfg" not in guidance

    def test_init_image_fields(self):
        body = S._build_request_body(
            prompt="p",
            init_image_b64="AAAA",
            mask_image_b64="BBBB",
            strength=0.6,
        )
        assert body["init_image"] == "AAAA"
        assert body["mask_image"] == "BBBB"
        assert body["strength"] == 0.6

    def test_ref_images_passed_as_array(self):
        body = S._build_request_body(
            prompt="p", ref_images_b64=["A==", "B=="]
        )
        assert body["ref_images"] == ["A==", "B=="]

    def test_empty_ref_images_omitted(self):
        body = S._build_request_body(prompt="p", ref_images_b64=None)
        assert "ref_images" not in body

    def test_lora_passed_through(self):
        lora = [{"path": "foo.safetensors", "multiplier": 0.8}]
        body = S._build_request_body(prompt="p", lora=lora)
        assert body["lora"] == lora

    def test_dimensions_and_seed(self):
        body = S._build_request_body(
            prompt="p", width=768, height=512, seed=42, batch_count=2
        )
        assert body["width"] == 768
        assert body["height"] == 512
        assert body["seed"] == 42
        assert body["batch_count"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Slug + image writing
# ─────────────────────────────────────────────────────────────────────────────


class TestSlugify:
    def test_basic(self):
        assert S._slugify("a cat on a mat") == "a_cat_on_a_mat"

    def test_strips_special_chars(self):
        assert S._slugify("Hello! World?? (2026)") == "hello_world_2026"

    def test_max_len(self):
        s = S._slugify("x" * 500, max_len=10)
        assert len(s) <= 10

    def test_empty_fallback(self):
        assert S._slugify("   ") == "img"
        assert S._slugify("!@#$%") == "img"


class TestWriteImages:
    def test_writes_all_with_indexed_names(self, tmp_path):
        b64 = base64.standard_b64encode(_TINY_PNG).decode("ascii")
        images = [
            {"index": 0, "b64_json": b64},
            {"index": 1, "b64_json": b64},
        ]
        paths = S._write_images(images, tmp_path, "png", "myprefix")
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert all(p.read_bytes() == _TINY_PNG for p in paths)
        names = [p.name for p in paths]
        assert any("myprefix_" in n and "_00." in n for n in names)
        assert any("myprefix_" in n and "_01." in n for n in names)

    def test_jpeg_becomes_jpg_extension(self, tmp_path):
        b64 = base64.standard_b64encode(b"fakejpeg").decode("ascii")
        paths = S._write_images(
            [{"index": 0, "b64_json": b64}], tmp_path, "jpeg", "x"
        )
        assert paths[0].suffix == ".jpg"

    def test_skips_image_without_b64(self, tmp_path):
        paths = S._write_images(
            [{"index": 0, "b64_json": None}], tmp_path, "png", "x"
        )
        assert paths == []

    def test_collision_gets_suffix(self, tmp_path, monkeypatch):
        # Force a deterministic timestamp so the first and second call collide.
        fixed_ts = 1_700_000_000_000
        monkeypatch.setattr(S.time, "time", lambda: fixed_ts / 1000)
        b64 = base64.standard_b64encode(_TINY_PNG).decode("ascii")

        first = S._write_images(
            [{"index": 0, "b64_json": b64}], tmp_path, "png", "p"
        )
        second = S._write_images(
            [{"index": 0, "b64_json": b64}], tmp_path, "png", "p"
        )
        assert len(first) == 1 and len(second) == 1
        assert first[0] != second[0]
        assert first[0].exists() and second[0].exists()
        # Second name carries a disambiguator
        assert "_1." in second[0].name or "_2." in second[0].name


# ─────────────────────────────────────────────────────────────────────────────
# Job polling
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestPollJob:
    async def test_returns_on_completed(self, monkeypatch):
        calls = {"n": 0}

        async def fake_get(path):
            calls["n"] += 1
            if calls["n"] < 3:
                return {"id": "j1", "status": "generating"}
            return {
                "id": "j1",
                "status": "completed",
                "result": {"output_format": "png", "images": []},
            }

        monkeypatch.setattr(S, "_get_json", fake_get)
        result = await S._poll_job("j1", timeout=5.0, ctx=None, poll_interval=0.0)
        assert result["status"] == "completed"
        assert calls["n"] == 3

    async def test_returns_on_failed(self, monkeypatch):
        async def fake_get(path):
            return {"id": "j1", "status": "failed", "error": {"code": "x", "message": "y"}}

        monkeypatch.setattr(S, "_get_json", fake_get)
        result = await S._poll_job("j1", timeout=5.0, ctx=None, poll_interval=0.0)
        assert result["status"] == "failed"

    async def test_times_out(self, monkeypatch):
        async def fake_get(path):
            return {"id": "j1", "status": "generating"}

        monkeypatch.setattr(S, "_get_json", fake_get)
        with pytest.raises(TimeoutError):
            await S._poll_job("j1", timeout=0.1, ctx=None, poll_interval=0.01)

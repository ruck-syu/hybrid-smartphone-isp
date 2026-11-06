"""Unit tests for core ISP invariants.  Run: python -m pytest -q tests"""
import numpy as np
import pytest

from isp import classical as C
from isp.hybrid import estimate_nlf, run_tiled
from isp.rawio import RawImage, cam2srgb_from_xyz2cam


def mosaic(rgb, cfa="RGGB"):
    idx = {"R": 0, "G": 1, "B": 2}
    out = np.zeros(rgb.shape[:2], np.float32)
    for i, ch in enumerate(cfa):
        out[i // 2::2, i % 2::2] = rgb[i // 2::2, i % 2::2, idx[ch]]
    return out


@pytest.mark.parametrize("cfa", ["RGGB", "GRBG", "GBRG", "BGGR"])
@pytest.mark.parametrize("method", ["bilinear", "malvar", "ea"])
def test_demosaic_flat_colour_any_cfa(cfa, method):
    rgb = np.ones((32, 48, 3), np.float32) * np.array([0.2, 0.5, 0.8], np.float32)
    out = C.demosaic(mosaic(rgb, cfa), cfa, method)
    assert out.shape == rgb.shape
    np.testing.assert_allclose(out[4:-4, 4:-4], rgb[4:-4, 4:-4], atol=2e-3)


def test_pack_unpack_roundtrip():
    x = np.random.default_rng(0).random((16, 20)).astype(np.float32)
    np.testing.assert_array_equal(C.unpack_rggb(C.pack_rggb(x)), x)


def test_ccm_preserves_white():
    cm = np.array([0.66, -0.05, -0.12, -0.54, 1.44, 0.07, -0.18, 0.41, 0.48])
    M = cam2srgb_from_xyz2cam(cm)
    np.testing.assert_allclose(M @ np.ones(3), np.ones(3), atol=1e-5)


def test_srgb_oetf_inverse():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    np.testing.assert_allclose(C.srgb_eotf(C.srgb_oetf(x)), x, atol=1e-5)


def test_nlf_estimation_recovers_noise_model():
    rng = np.random.default_rng(1)
    a, b = 4e-4, 2e-6
    clean = np.tile(np.linspace(0.02, 0.8, 512, dtype=np.float32), (512, 1))
    clean = np.stack([clean] * 4, -1)
    noisy = clean + rng.standard_normal(clean.shape).astype(np.float32) * np.sqrt(a * clean + b)
    ea, eb = estimate_nlf(noisy)
    assert abs(ea - a) / a < 0.25


def test_tiled_equals_untiled_for_pointwise_op():
    fn = lambda x: np.repeat(np.repeat(x[:, :3], 2, 2), 2, 3)  # nearest upsample of first 3 channels
    x = np.random.default_rng(2).random((200, 260, 8)).astype(np.float32)
    full = run_tiled(fn, x, tile=1024)
    tiled = run_tiled(fn, x, tile=64, overlap=8)
    np.testing.assert_allclose(full, tiled, atol=1e-5)


def test_classical_pipeline_runs_without_metadata():
    rng = np.random.default_rng(3)
    raw = RawImage(bayer=rng.random((64, 64)).astype(np.float32) * 0.3, cfa="GRBG")
    out, info = C.ClassicalISP(C.ClassicalConfig(wb="shades_of_gray")).run(raw)
    assert out.shape == (64, 64, 3) and np.isfinite(out).all()
    assert 0 <= out.min() and out.max() <= 1

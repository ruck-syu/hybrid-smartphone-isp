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


@pytest.mark.parametrize("shape", [(9, 9), (9, 8), (17, 23)])
def test_hybrid_odd_sizes_reconstruct_edges(shape, tmp_path):
    """Odd mosaics are padded (not truncated): output has the input shape and a real last row/col."""
    from isp.hybrid import HybridISP

    class Fake:
        mode = "hybrid"
        path = "fake"

        def __call__(self, x):  # bilinear-ish: repeat WB'd R/G/B planes
            r, g, b = x[:, 0:1], 0.5 * (x[:, 1:2] + x[:, 2:3]), x[:, 3:4]
            up = lambda z: np.repeat(np.repeat(z, 2, 2), 2, 3)
            return np.concatenate([up(r), up(g), up(b)], 1)

    isp = HybridISP.__new__(HybridISP)
    isp.net, isp.tile = Fake(), 512
    rng = np.random.default_rng(0)
    raw = RawImage(bayer=rng.random(shape).astype(np.float32) * 0.5, cfa="GRBG",
                   wb_gains=np.ones(3, np.float32), cam2srgb=np.eye(3, dtype=np.float32), noise_profile=(1e-4, 1e-6))
    out, _ = isp.run(raw)
    assert out.shape == shape + (3,)
    assert np.isfinite(out).all()


def test_tile_starts_are_evenly_spaced():
    calls = []

    def fn(x):
        calls.append(x.shape)
        return np.zeros((1, 3, x.shape[2] * 2, x.shape[3] * 2), np.float32)
    run_tiled(fn, np.zeros((1024, 1024, 8), np.float32), tile=512, overlap=32)
    assert len(calls) == 9  # 3x3 evenly spaced tiles, not a near-duplicate last tile


def test_tiny_classical_input_does_not_crash():
    raw = RawImage(bayer=np.full((4, 4), 0.2, np.float32), cfa="RGGB")
    out, _ = C.ClassicalISP(C.ClassicalConfig(wb="grayworld")).run(raw)
    assert out.shape == (4, 4, 3)

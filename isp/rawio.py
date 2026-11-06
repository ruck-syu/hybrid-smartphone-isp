"""RAW input/output: SIDD .MAT files, DNG files (via rawpy) and a unified RawImage container.

All Bayer data is converted to a canonical form:
  * float32, black-level subtracted, normalised to [0, 1] by the white level
  * CFA pattern stored as a 4-char string in raster order of the 2x2 tile, e.g. "RGGB", "GRBG"
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

# XYZ(D65) -> linear sRGB
XYZ2SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                     [-0.9692660, 1.8760108, 0.0415560],
                     [0.0556434, -0.2040259, 1.0572252]], dtype=np.float64)

_CFA_CODES = "RGB"


@dataclasses.dataclass
class RawImage:
    bayer: np.ndarray                 # (H, W) float32 in [0, 1]
    cfa: str = "RGGB"                 # 2x2 pattern in raster order
    wb_gains: np.ndarray | None = None  # (3,) multiplicative gains (R, G, B), G == 1
    cam2srgb: np.ndarray | None = None  # (3, 3) white-balanced camera RGB -> linear sRGB
    iso: float | None = None
    exposure: float | None = None
    noise_profile: tuple[float, float] | None = None  # (shot, read) variance params: var = a*x + b
    orientation: int = 1
    make: str = ""
    model: str = ""
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def has_metadata(self) -> bool:
        return self.wb_gains is not None and self.cam2srgb is not None

    def to_json(self) -> dict:
        return {
            "cfa": self.cfa,
            "wb_gains": None if self.wb_gains is None else [float(x) for x in self.wb_gains],
            "cam2srgb": None if self.cam2srgb is None else np.asarray(self.cam2srgb).tolist(),
            "iso": self.iso, "exposure": self.exposure,
            "noise_profile": self.noise_profile, "orientation": self.orientation,
            "make": self.make, "model": self.model,
        }


def cam2srgb_from_xyz2cam(xyz2cam: np.ndarray) -> np.ndarray:
    """DNG ColorMatrix (XYZ->camera) to a white-preserving camera->sRGB matrix.

    Following the standard DNG / simple-camera-pipeline convention: combine with XYZ->sRGB,
    normalise rows so that a white-balanced neutral (1,1,1) maps to sRGB white, then invert.
    """
    xyz2cam = np.asarray(xyz2cam, dtype=np.float64).reshape(3, 3)
    srgb2cam = xyz2cam @ np.linalg.inv(XYZ2SRGB)
    srgb2cam = srgb2cam / srgb2cam.sum(axis=1, keepdims=True)
    return np.linalg.inv(srgb2cam).astype(np.float32)


def _cfa_from_codes(codes) -> str:
    codes = [int(c) for c in np.asarray(codes).ravel()[:4]]
    return "".join(_CFA_CODES[c] for c in codes)


def _illuminant_cct(code: int) -> float:
    # EXIF LightSource codes -> approximate CCT
    table = {17: 2856, 21: 6504, 20: 5503, 23: 5003, 22: 7504, 1: 5500, 2: 4200, 3: 2856,
             4: 5500, 9: 6000, 10: 6500, 11: 7000, 12: 6400, 13: 5000, 14: 4150, 15: 3450,
             18: 4874, 19: 6774, 24: 3200}
    return float(table.get(int(code), 5000))


def interpolate_color_matrix(cm1, cm2, ill1, ill2, wb_gains) -> np.ndarray:
    """Pick/interpolate between the two DNG colour matrices using a crude CCT estimate.

    The CCT of the scene is estimated from the as-shot neutral by finding which matrix maps it
    closer to its calibration illuminant; we interpolate in inverse-CCT like the DNG SDK.
    """
    cm1 = np.asarray(cm1, np.float64).reshape(3, 3)
    cm2 = np.asarray(cm2, np.float64).reshape(3, 3)
    t1, t2 = _illuminant_cct(ill1), _illuminant_cct(ill2)
    if abs(t1 - t2) < 1:
        return cm1
    neutral = 1.0 / np.asarray(wb_gains, np.float64)
    # Iterate: guess weight -> xyz of neutral -> CCT (McCamy) -> weight
    w = 0.5
    for _ in range(10):
        cm = w * cm1 + (1 - w) * cm2
        xyz = np.linalg.solve(cm, neutral)
        x, y = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
        n = (x - 0.3320) / (0.1858 - y)
        cct = float(np.clip(449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33, 2000, 12000))
        lo, hi = sorted([(t1, 1.0), (t2, 0.0)])
        if cct <= lo[0]:
            w_new = lo[1]
        elif cct >= hi[0]:
            w_new = hi[1]
        else:
            a = (1 / cct - 1 / hi[0]) / (1 / lo[0] - 1 / hi[0])
            w_new = a * lo[1] + (1 - a) * hi[1]
        if abs(w_new - w) < 1e-4:
            break
        w = w_new
    return w * cm1 + (1 - w) * cm2


def load_sidd_metadata(path: str | Path) -> dict:
    import scipy.io as sio
    md = sio.loadmat(str(path))["metadata"][0, 0]
    names = md.dtype.names

    def g(k, default=None):
        if k not in names:
            return default
        v = md[k]
        try:
            v = np.asarray(v).squeeze()
            return v if v.size else default
        except Exception:
            return default

    out = {"make": str(g("Make", "")), "model": str(g("Model", "")),
           "iso": float(np.ravel(g("ISOSpeedRatings", 0) or 0)[0]),
           "exposure": float(np.ravel(g("ExposureTime", 0) or 0)[0]),
           "orientation": int(g("Orientation", 1) or 1),
           "as_shot_neutral": np.asarray(g("AsShotNeutral"), np.float64).ravel(),
           "cm1": g("ColorMatrix1"), "cm2": g("ColorMatrix2"),
           "ill1": int(g("CalibrationIlluminant1", 21) or 21),
           "ill2": int(g("CalibrationIlluminant2", 17) or 17),
           "cfa": "RGGB", "noise_profile": None}
    for t in md["UnknownTags"].ravel():
        tag = int(np.asarray(t[0]).ravel()[0])
        val = np.asarray(t[2]).ravel()
        if tag == 33422:
            out["cfa"] = _cfa_from_codes(val)
        elif tag == 51041 and val.size >= 2:  # same policy as DNG files: mean of per-plane (S, O)
            out["noise_profile"] = (float(val[0::2].mean()), float(val[1::2].mean()))
    if out["cm1"] is None:
        out["cm1"] = out["cm2"]
    if out["cm2"] is None:
        out["cm2"] = out["cm1"]
    return out


def load_sidd_raw(raw_path: str | Path, meta_path: str | Path | None = None) -> RawImage:
    import h5py
    with h5py.File(str(raw_path), "r") as f:
        x = np.array(f["x"], dtype=np.float32).T  # MATLAB column-major -> (H, W)
    img = RawImage(bayer=np.clip(x, 0, 1))
    if meta_path is not None:
        m = load_sidd_metadata(meta_path)
        img.cfa = m["cfa"]
        neutral = m["as_shot_neutral"]
        if neutral.size == 3 and np.all(np.isfinite(neutral)) and np.all(neutral > 0):
            img.wb_gains = (neutral[1] / neutral).astype(np.float32)
        else:  # missing AsShotNeutral -> grey-world estimate from the mosaic
            from .classical import estimate_wb_grayworld
            img.wb_gains = estimate_wb_grayworld(img.bayer, m["cfa"])
        cm = interpolate_color_matrix(m["cm1"], m["cm2"], m["ill1"], m["ill2"], img.wb_gains)
        img.cam2srgb = cam2srgb_from_xyz2cam(cm)
        img.iso, img.exposure = m["iso"], m["exposure"]
        img.noise_profile = m["noise_profile"]
        img.orientation, img.make, img.model = m["orientation"], m["make"], m["model"]
        img.meta = {"cm1": np.asarray(m["cm1"]).tolist(), "cm2": np.asarray(m["cm2"]).tolist(),
                    "ill1": m["ill1"], "ill2": m["ill2"]}
    return img


def _rational(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v[0::2] / np.where(v[1::2] == 0, 1, v[1::2]) if v.size % 2 == 0 and v.size >= 2 else v


def read_dng_tags(path: str | Path) -> dict:
    """Colour/noise metadata straight from DNG IFD0 (LibRaw does not expose all of it)."""
    out = {}
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as t:
            tags = {tg.code: tg.value for tg in t.pages[0].tags.values()}
    except Exception:
        return out

    def rat(code, n):
        if code not in tags:
            return None
        v = tags[code]
        arr = np.asarray(v, dtype=np.float64).ravel()
        # tifffile returns rationals as flat (num, den, num, den, ...) tuples
        if arr.size == 2 * n:
            arr = _rational(arr)
        return arr if arr.size == n else None
    out["cm1"], out["cm2"] = rat(50721, 9), rat(50722, 9)
    out["ill1"], out["ill2"] = int(tags.get(50778, 21)), int(tags.get(50779, 17))
    out["as_shot_neutral"] = rat(50728, 3)
    if 51041 in tags:
        npf = np.asarray(tags[51041], dtype=np.float64).ravel()
        if npf.size >= 2:
            out["noise_profile"] = (float(npf[0::2].mean()), float(npf[1::2].mean()))
    if 274 in tags:
        out["orientation"] = int(tags[274])
    return out


def load_dng(path: str | Path) -> RawImage:
    """Load any RAW format supported by LibRaw (DNG, CR2, NEF, ARW...)."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        if r.raw_pattern is None or r.num_colors != 3 or np.asarray(r.raw_pattern).shape != (2, 2):
            raise ValueError("Only 2x2 Bayer sensors are supported (X-Trans, Foveon, monochrome and "
                             "linear/demosaiced DNGs are not).")
        raw = r.raw_image_visible.astype(np.float32)
        # colour index map of the *visible* area (accounts for odd top/left sensor margins)
        colors = np.asarray(r.raw_colors_visible)
        desc = r.color_desc.decode()
        cfa = "".join(desc[colors[i // 2, i % 2]] for i in range(4))
        if sorted(cfa) != ["B", "G", "G", "R"]:
            raise ValueError(f"Unsupported CFA layout {cfa!r}.")
        blacks = np.array(r.black_level_per_channel, np.float32)
        black = blacks[colors] if len(blacks) >= 4 else np.full_like(raw, float(blacks.mean()))
        white = float(r.white_level)
        bayer = np.clip((raw - black) / max(white - black.max(), 1.0), 0, 1)
        wb = np.array(r.camera_whitebalance[:3], np.float64)
        wb_gains = None
        if wb.min() > 0:
            wb_gains = (wb / wb[1]).astype(np.float32)
        xyz2cam = np.array(r.rgb_xyz_matrix[:3], np.float64)
        cam2srgb = cam2srgb_from_xyz2cam(xyz2cam) if np.abs(xyz2cam).sum() > 0 else None
        img = RawImage(bayer=bayer.astype(np.float32), cfa=cfa.replace("g", "G"),
                       wb_gains=wb_gains, cam2srgb=cam2srgb)
        try:
            img.orientation = {0: 1, 3: 3, 5: 8, 6: 6}.get(r.sizes.flip, 1)
        except Exception:
            pass
    tags = read_dng_tags(path)
    if tags.get("as_shot_neutral") is not None and np.all(tags["as_shot_neutral"] > 0):
        n = tags["as_shot_neutral"]
        img.wb_gains = (n[1] / n).astype(np.float32)
    if tags.get("cm1") is not None and img.wb_gains is not None:
        cm2 = tags["cm2"] if tags.get("cm2") is not None else tags["cm1"]
        cm = interpolate_color_matrix(tags["cm1"], cm2, tags["ill1"], tags["ill2"], img.wb_gains)
        img.cam2srgb = cam2srgb_from_xyz2cam(cm)
    if tags.get("noise_profile") is not None:
        img.noise_profile = tags["noise_profile"]
    if "orientation" in tags:
        img.orientation = tags["orientation"]
    # EXIF ISO (best-effort)
    try:
        import exifread  # optional
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
        iso = tags.get("EXIF ISOSpeedRatings")
        if iso:
            img.iso = float(str(iso))
    except Exception:
        pass
    return img


def load_any(path: str | Path, meta_path: str | Path | None = None) -> RawImage:
    p = Path(path)
    if p.suffix.lower() == ".mat":
        return load_sidd_raw(p, meta_path)
    if p.suffix.lower() == ".npz":
        d = np.load(p, allow_pickle=True)
        meta = json.loads(str(d["meta"])) if "meta" in d else {}
        img = RawImage(bayer=d["bayer"].astype(np.float32), cfa=meta.get("cfa", "RGGB"))
        if meta.get("wb_gains") is not None:
            img.wb_gains = np.array(meta["wb_gains"], np.float32)
        if meta.get("cam2srgb") is not None:
            img.cam2srgb = np.array(meta["cam2srgb"], np.float32)
        img.iso = meta.get("iso")
        np_ = meta.get("noise_profile")
        img.noise_profile = tuple(np_) if np_ else None
        return img
    return load_dng(p)


def save_npz(path: str | Path, img: RawImage, crop: tuple[int, int, int, int] | None = None):
    b = img.bayer
    if crop is not None:
        y, x, h, w = crop
        b = b[y:y + h, x:x + w]
    np.savez_compressed(path, bayer=(b * 65535).round().astype(np.uint16).astype(np.float32) / 65535,
                        meta=json.dumps(img.to_json()))

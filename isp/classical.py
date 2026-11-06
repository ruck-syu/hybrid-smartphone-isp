"""Classical (hand-engineered) ISP stages, all numpy/OpenCV, all operating on float32.

Stage order of the conventional baseline (`ClassicalISP`):
    Bayer [0,1] -> (noise estimation) -> white balance (on CFA) -> demosaic -> CCM
    -> denoise (edge preserving, noise adaptive, in YCbCr) -> tone map -> gamma (sRGB OETF)
    -> sharpen (unsharp mask on luma) -> clip / 8-bit
"""
from __future__ import annotations

import dataclasses
import time

import cv2
import numpy as np

from .rawio import RawImage

# --------------------------------------------------------------------------------------------
# Bayer helpers
# --------------------------------------------------------------------------------------------

_SHIFT_TO_RGGB = {"RGGB": (0, 0), "GRBG": (0, 1), "GBRG": (1, 0), "BGGR": (1, 1)}


def to_rggb(bayer: np.ndarray, cfa: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad by (dy, dx) so the mosaic starts with R at (0,0). Returns padded image + shift.

    Reflect padding by one pixel copies a same-colour pixel (period 2), so the CFA stays valid.
    """
    dy, dx = _SHIFT_TO_RGGB[cfa]
    if dy or dx:
        bayer = np.pad(bayer, ((dy, dy), (dx, dx)), mode="reflect")
    return bayer, (dy, dx)


def from_rggb(img: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    dy, dx = shift
    h, w = img.shape[:2]
    return img[dy:h - dy if dy else h, dx:w - dx if dx else w]


def pack_rggb(bayer_rggb: np.ndarray) -> np.ndarray:
    """(H, W) RGGB mosaic -> (H/2, W/2, 4) planes [R, Gr, Gb, B]."""
    return np.stack([bayer_rggb[0::2, 0::2], bayer_rggb[0::2, 1::2],
                     bayer_rggb[1::2, 0::2], bayer_rggb[1::2, 1::2]], axis=-1)


def unpack_rggb(planes: np.ndarray) -> np.ndarray:
    h, w, _ = planes.shape
    out = np.empty((h * 2, w * 2), planes.dtype)
    out[0::2, 0::2], out[0::2, 1::2] = planes[..., 0], planes[..., 1]
    out[1::2, 0::2], out[1::2, 1::2] = planes[..., 2], planes[..., 3]
    return out


def cfa_masks(h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w, 3), np.float32)
    m[0::2, 0::2, 0] = 1
    m[0::2, 1::2, 1] = 1
    m[1::2, 0::2, 1] = 1
    m[1::2, 1::2, 2] = 1
    return m


# --------------------------------------------------------------------------------------------
# Demosaicing
# --------------------------------------------------------------------------------------------

def demosaic_bilinear(rggb: np.ndarray) -> np.ndarray:
    h, w = rggb.shape
    m = cfa_masks(h, w)
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4
    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4
    out = np.empty((h, w, 3), np.float32)
    for c, k in ((0, k_rb), (1, k_g), (2, k_rb)):
        out[..., c] = cv2.filter2D(rggb * m[..., c], -1, k, borderType=cv2.BORDER_REFLECT)
    return out


def demosaic_malvar(rggb: np.ndarray) -> np.ndarray:
    """Malvar-He-Cutler (2004) high-quality linear demosaicing (gradient-corrected bilinear)."""
    x = rggb.astype(np.float32)
    h, w = x.shape
    f = lambda k: cv2.filter2D(x, -1, np.array(k, np.float32) / 8, borderType=cv2.BORDER_REFLECT)
    g_at_rb = f([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]])
    rb_at_g_rrow = f([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]])
    rb_at_g_rcol = f([[0, 0, -1, 0, 0], [0, -1, 4, -1, 0], [.5, 0, 5, 0, .5], [0, -1, 4, -1, 0], [0, 0, -1, 0, 0]])
    rb_at_br = f([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]])
    R = np.zeros((h, w), np.float32); G = np.zeros_like(R); B = np.zeros_like(R)
    # R sites (0,0)
    R[0::2, 0::2] = x[0::2, 0::2]; G[0::2, 0::2] = g_at_rb[0::2, 0::2]; B[0::2, 0::2] = rb_at_br[0::2, 0::2]
    # B sites (1,1)
    B[1::2, 1::2] = x[1::2, 1::2]; G[1::2, 1::2] = g_at_rb[1::2, 1::2]; R[1::2, 1::2] = rb_at_br[1::2, 1::2]
    # G in R row (0,1): R horizontal neighbours, B vertical
    G[0::2, 1::2] = x[0::2, 1::2]; R[0::2, 1::2] = rb_at_g_rrow[0::2, 1::2]; B[0::2, 1::2] = rb_at_g_rcol[0::2, 1::2]
    # G in B row (1,0): B horizontal, R vertical
    G[1::2, 0::2] = x[1::2, 0::2]; B[1::2, 0::2] = rb_at_g_rrow[1::2, 0::2]; R[1::2, 0::2] = rb_at_g_rcol[1::2, 0::2]
    return np.clip(np.stack([R, G, B], -1), 0, None)


def demosaic_opencv(rggb: np.ndarray, method: str = "EA") -> np.ndarray:
    """OpenCV edge-aware (EA) or VNG demosaicing. Uses 16-bit internally."""
    x16 = np.clip(rggb * 65535, 0, 65535).astype(np.uint16)
    # OpenCV names Bayer patterns by the 2nd row, 2nd/3rd column: RGGB raster == "BayerBG".
    code = {"EA": cv2.COLOR_BayerBG2RGB_EA, "VNG": cv2.COLOR_BayerBG2RGB_VNG,
            "BILINEAR": cv2.COLOR_BayerBG2RGB}[method]
    if method == "VNG":
        x8 = (x16 >> 8).astype(np.uint8)  # VNG only supports 8-bit
        return cv2.cvtColor(x8, code).astype(np.float32) / 255
    return cv2.cvtColor(x16, code).astype(np.float32) / 65535


DEMOSAIC = {"bilinear": demosaic_bilinear, "malvar": demosaic_malvar,
            "ea": lambda x: demosaic_opencv(x, "EA"), "vng": lambda x: demosaic_opencv(x, "VNG")}


def demosaic(bayer: np.ndarray, cfa: str, method: str = "malvar") -> np.ndarray:
    rggb, shift = to_rggb(bayer, cfa)
    return from_rggb(DEMOSAIC[method](rggb), shift)


# --------------------------------------------------------------------------------------------
# White balance
# --------------------------------------------------------------------------------------------

def apply_wb_bayer(bayer: np.ndarray, cfa: str, gains: np.ndarray) -> np.ndarray:
    out = bayer.copy()
    idx = {"R": 0, "G": 1, "B": 2}
    for i, ch in enumerate(cfa):
        out[i // 2::2, i % 2::2] *= gains[idx[ch]]
    return out


def bayer_channel_means(bayer: np.ndarray, cfa: str, mask_sat: float = 0.98) -> np.ndarray:
    sums, cnts = np.zeros(3), np.zeros(3)
    idx = {"R": 0, "G": 1, "B": 2}
    for i, ch in enumerate(cfa):
        p = bayer[i // 2::2, i % 2::2]
        valid = p < mask_sat
        sums[idx[ch]] += p[valid].sum()
        cnts[idx[ch]] += valid.sum()
    return sums / np.maximum(cnts, 1)


def estimate_wb_grayworld(bayer: np.ndarray, cfa: str) -> np.ndarray:
    m = bayer_channel_means(bayer, cfa)
    return (m[1] / np.maximum(m, 1e-8)).astype(np.float32)


def estimate_wb_shades_of_gray(rgb_lin: np.ndarray, p: float = 6.0) -> np.ndarray:
    """Shades-of-grey (Finlayson 2004) AWB on a demosaiced linear camera image (Minkowski p-norm)."""
    x = rgb_lin.reshape(-1, 3)
    lum = x.mean(1)
    valid = (x.max(1) < 0.98) & (lum > np.percentile(lum, 2))
    x = x[valid] if valid.sum() > 100 else x
    e = np.power(np.mean(np.power(np.maximum(x, 0), p), 0), 1 / p)
    return (e[1] / np.maximum(e, 1e-8)).astype(np.float32)


def estimate_wb_gray_edge(rgb_lin: np.ndarray, sigma: float = 2.0, p: float = 6.0) -> np.ndarray:
    """Grey-edge AWB (van de Weijer 2007): the average edge difference is achromatic."""
    blur = cv2.GaussianBlur(rgb_lin, (0, 0), sigma)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2).reshape(-1, 3)
    e = np.power(np.mean(np.power(mag, p), 0), 1 / p)
    return (e[1] / np.maximum(e, 1e-8)).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------------------------

# Generic camera->sRGB matrix (average of SIDD cameras) used when no metadata is available.
GENERIC_CAM2SRGB = np.array([[1.75, -0.60, -0.15], [-0.22, 1.60, -0.38], [0.02, -0.55, 1.53]], np.float32)


def apply_ccm(rgb: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    return (rgb.reshape(-1, 3) @ np.asarray(ccm, np.float32).T).reshape(h, w, 3)


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def srgb_eotf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def gamma_simple(x: np.ndarray, g: float = 2.2) -> np.ndarray:
    return np.power(np.clip(x, 0, 1), 1 / g).astype(np.float32)


def rgb_to_ycbcr(x: np.ndarray) -> np.ndarray:
    m = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], np.float32)
    return x @ m.T


def ycbcr_to_rgb(x: np.ndarray) -> np.ndarray:
    m = np.array([[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], np.float32)
    return x @ m.T


# --------------------------------------------------------------------------------------------
# Tone mapping
# --------------------------------------------------------------------------------------------

def auto_exposure_gain(rgb_lin: np.ndarray, target: float = 0.18, max_gain: float = 16.0) -> float:
    """Digital gain bringing the (robust) log-average luminance to `target`."""
    lum = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    lum = lum[::4, ::4]
    key = float(np.exp(np.mean(np.log(np.clip(lum, 1e-4, 1)))))
    p99 = float(np.percentile(lum, 99.5))
    gain = target / max(key, 1e-4)
    gain = min(gain, 0.98 / max(p99, 1e-4) * 1.5, max_gain)  # avoid blowing highlights
    return max(gain, 1.0)


def tone_reinhard(x: np.ndarray, white: float = 4.0) -> np.ndarray:
    """Extended Reinhard global operator applied on luminance, preserving chromaticity."""
    lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    lum_t = lum * (1 + lum / white ** 2) / (1 + lum)
    return x * (lum_t / np.maximum(lum, 1e-6))[..., None]


def tone_filmic_s(x: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Mild S-curve applied in display space (contrast boost). x in [0,1]."""
    s = x * x * (3 - 2 * x)
    return (1 - strength) * x + strength * s


# --------------------------------------------------------------------------------------------
# Denoising (edge preserving, noise adaptive)
# --------------------------------------------------------------------------------------------

def estimate_noise_sigma(bayer: np.ndarray, cfa: str = "RGGB") -> float:
    """Robust noise estimate (Donoho MAD of the finest Haar diagonal subband) on the green plane."""
    rggb, _ = to_rggb(bayer, cfa)
    g = rggb[0::2, 1::2]
    h, w = (g.shape[0] // 2) * 2, (g.shape[1] // 2) * 2
    g = g[:h, :w]
    hh = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    if hh.size == 0:
        return 0.0
    return float(np.median(np.abs(hh)) / 0.6745)


def noise_level_at(signal: float, profile: tuple[float, float] | None, fallback_sigma: float) -> float:
    if profile is None:
        return fallback_sigma
    a, b = profile
    return float(np.sqrt(max(a * signal + b, 1e-12)))


def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter (grey guide). I, p: (H, W) float32."""
    box = lambda z: cv2.boxFilter(z, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    mI, mp = box(I), box(p)
    cov = box(I * p) - mI * mp
    var = box(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def denoise_bilateral_ycc(rgb: np.ndarray, sigma: float, luma_strength: float = 1.0,
                          chroma_strength: float = 2.0) -> np.ndarray:
    """Noise-adaptive edge-preserving denoise: bilateral on luma, guided filter on chroma."""
    if sigma <= 1e-5:
        return rgb
    ycc = rgb_to_ycbcr(rgb)
    y = ycc[..., 0].astype(np.float32)
    s_r = float(np.clip(3.0 * sigma * luma_strength, 1e-3, 0.5))
    d = int(np.clip(round(3 + sigma * 200), 3, 9))
    y_d = cv2.bilateralFilter(y, d, s_r, d / 2.0)
    r = int(np.clip(round(2 + sigma * 300 * chroma_strength), 2, 12))
    eps = (2.0 * sigma * chroma_strength) ** 2
    cb = guided_filter(y_d, ycc[..., 1].astype(np.float32), r, eps)
    cr = guided_filter(y_d, ycc[..., 2].astype(np.float32), r, eps)
    return ycbcr_to_rgb(np.stack([y_d, cb, cr], -1))


def denoise_bayer(bayer: np.ndarray, cfa: str, profile, strength: float = 1.0) -> np.ndarray:
    """Bayer-domain noise reduction (before demosaic / clipping, as in hardware ISPs).

    Each colour plane is filtered with a variance-stabilised bilateral: the generalised Anscombe
    transform makes the Poisson-Gaussian noise ~unit variance, so one range sigma fits all levels."""
    a, b = profile
    rggb, shift = to_rggb(bayer, cfa)
    H, W = (rggb.shape[0] // 2) * 2, (rggb.shape[1] // 2) * 2
    p = pack_rggb(rggb[:H, :W])
    a = max(a, 1e-8)
    vst = lambda x: (2 / a) * np.sqrt(np.maximum(a * x + 3 / 8 * a * a + b, 0))
    ivst = lambda y: np.maximum(((a * y / 2) ** 2 - 3 / 8 * a * a - b) / a, 0)
    out = np.empty_like(p)
    for c in range(4):
        y = vst(p[..., c]).astype(np.float32)
        out[..., c] = ivst(cv2.bilateralFilter(y, 5, 2.0 * strength, 2.0))
    rec = rggb.copy()
    rec[:H, :W] = unpack_rggb(out.astype(np.float32))
    return from_rggb(rec, shift)


def denoise_nlm(rgb_disp: np.ndarray, sigma: float) -> np.ndarray:
    """OpenCV fast non-local means on display-referred 8-bit (slower, higher quality)."""
    x8 = np.clip(rgb_disp * 255, 0, 255).astype(np.uint8)
    h = float(np.clip(sigma * 255 * 1.2, 1, 30))
    out = cv2.fastNlMeansDenoisingColored(x8, None, h, h * 1.5, 5, 15)
    return out.astype(np.float32) / 255


# --------------------------------------------------------------------------------------------
# Sharpening
# --------------------------------------------------------------------------------------------

def sharpen_unsharp(rgb: np.ndarray, amount: float = 0.5, sigma: float = 1.0, threshold: float = 0.01) -> np.ndarray:
    """Unsharp mask on luma only, with a noise threshold (coring) so flat regions are not boosted."""
    if amount <= 0:
        return rgb
    ycc = rgb_to_ycbcr(rgb)
    y = ycc[..., 0]
    detail = y - cv2.GaussianBlur(y, (0, 0), sigma)
    detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
    ycc[..., 0] = y + amount * detail
    return ycbcr_to_rgb(ycc)


# --------------------------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------------------------

def apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 3:
        return img[::-1, ::-1]
    if orientation == 6:
        return np.rot90(img, -1)
    if orientation == 8:
        return np.rot90(img, 1)
    return img


# --------------------------------------------------------------------------------------------
# Full conventional pipeline
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class ClassicalConfig:
    demosaic: str = "malvar"
    wb: str = "metadata"          # metadata | grayworld | shades_of_gray | gray_edge | none
    ccm: str = "metadata"         # metadata | generic | none
    denoise: str = "bilateral"    # bilateral | nlm | none
    raw_denoise: bool = True      # Bayer-domain NR (variance-stabilised bilateral) before demosaic
    denoise_strength: float = 1.0  # tuned on SIDD training scenes (with Bayer-domain NR enabled)
    adaptive: bool = True         # noise-adaptive parameters (from NLF/estimated sigma)
    tone: str = "none"            # none | reinhard | auto
    gamma: str = "srgb"           # srgb | 2.2
    sharpen: float = 0.0          # tuning found sharpening lowers fidelity on SIDD GT
    exposure_gain: float = 1.0
    apply_orientation: bool = False


class ClassicalISP:
    """Conventional ISP baseline. `run` returns the final sRGB image and per-stage intermediates."""

    def __init__(self, cfg: ClassicalConfig | None = None):
        self.cfg = cfg or ClassicalConfig()

    def run(self, raw: RawImage, keep_intermediates: bool = False) -> tuple[np.ndarray, dict]:
        c = self.cfg
        inter, timings = {}, {}
        t = time.perf_counter()

        def tick(name, img=None):
            nonlocal t
            now = time.perf_counter()
            timings[name] = (now - t) * 1000
            t = now
            if keep_intermediates and img is not None:
                inter[name] = img

        bayer = raw.bayer
        sigma = estimate_noise_sigma(bayer, raw.cfa)
        mean_sig = float(np.mean(bayer))
        sigma_nlf = noise_level_at(mean_sig, raw.noise_profile, sigma)
        tick("noise_estimation")

        # White balance on the CFA (before demosaic: avoids colour zippering at edges)
        if c.wb == "metadata" and raw.wb_gains is not None:
            gains = raw.wb_gains
        elif c.wb in ("grayworld", "metadata"):
            gains = estimate_wb_grayworld(bayer, raw.cfa)
        elif c.wb in ("shades_of_gray", "gray_edge"):
            rough = demosaic(bayer[: (bayer.shape[0] // 2) * 2, : (bayer.shape[1] // 2) * 2], raw.cfa, "bilinear")
            rough = cv2.resize(rough, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            gains = estimate_wb_shades_of_gray(rough) if c.wb == "shades_of_gray" else estimate_wb_gray_edge(rough)
        else:
            gains = np.ones(3, np.float32)
        if c.raw_denoise and c.denoise != "none":
            prof = raw.noise_profile if (raw.noise_profile is not None and c.adaptive) else (
                (sigma ** 2 / max(mean_sig, 1e-3), 1e-6) if c.adaptive else (2e-4, 1e-6))
            bayer = denoise_bayer(bayer, raw.cfa, prof, strength=1.0)
            tick("raw_denoise", bayer)
        x = apply_wb_bayer(bayer, raw.cfa, gains * c.exposure_gain)
        x = np.minimum(x, 1.0)  # saturated photosites -> neutral white (avoids magenta highlights)
        tick("white_balance", x)

        rgb = demosaic(x, raw.cfa, c.demosaic)
        tick("demosaic", rgb)

        if c.ccm == "metadata" and raw.cam2srgb is not None:
            ccm = raw.cam2srgb
        elif c.ccm in ("generic", "metadata"):
            ccm = GENERIC_CAM2SRGB
        else:
            ccm = np.eye(3, dtype=np.float32)
        rgb = np.clip(apply_ccm(rgb, ccm), 0, None)
        tick("ccm", rgb)

        if c.tone in ("auto", "reinhard"):
            g = auto_exposure_gain(rgb) if c.tone == "auto" else 1.0
            rgb = tone_reinhard(rgb * g, white=4.0) if c.tone == "reinhard" else np.clip(rgb * g, 0, 1)
        rgb = np.clip(rgb, 0, 1)
        tick("tone_map", rgb)

        disp = srgb_oetf(rgb) if c.gamma == "srgb" else gamma_simple(rgb, 2.2)
        tick("gamma", disp)

        # Denoise in display space (perceptually uniform noise); sigma propagated through gamma.
        # d(gamma)/dx at the mean signal level gives the display-space noise std.
        if c.denoise != "none":
            lvl = max(mean_sig * float(np.mean(gains)), 1e-4)
            slope = (1.055 / 2.4) * lvl ** (1 / 2.4 - 1) if c.gamma == "srgb" else (1 / 2.2) * lvl ** (1 / 2.2 - 1)
            s_disp = (sigma_nlf if c.adaptive else 0.004) * float(np.mean(gains)) * slope * c.denoise_strength
            s_disp = float(np.clip(s_disp, 0.0, 0.25))
            disp = denoise_nlm(disp, s_disp) if c.denoise == "nlm" else denoise_bilateral_ycc(disp, s_disp)
            inter_sigma = s_disp
        else:
            inter_sigma = 0.0
        tick("denoise", disp)

        amount = c.sharpen * (1.0 / (1.0 + 30 * inter_sigma) if c.adaptive else 1.0)
        disp = sharpen_unsharp(disp, amount=amount, sigma=1.0, threshold=max(inter_sigma, 0.004))
        disp = np.clip(disp, 0, 1)
        tick("sharpen", disp)

        if c.apply_orientation:
            disp = apply_orientation(disp, raw.orientation)
        info = {"timings_ms": timings, "wb_gains": [float(v) for v in gains], "sigma_raw": sigma,
                "sigma_nlf": sigma_nlf, "sigma_display": inter_sigma, "intermediates": inter}
        return disp.astype(np.float32), info

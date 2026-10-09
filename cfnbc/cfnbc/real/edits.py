"""Task-preserving counterfactual edits of real camera frames (no simulator), with masks and checks.

Without a simulator we can't re-render a recorded state, so counterfactuals are *edits of the recorded frame*,
restricted to pixels that can't carry task information:
  table mask   = pixels that match the episode's static background (temporal median: the camera is fixed, the arms
                 move) where that background looks like mat (low saturation, not black/white); the cup is transparent,
                 so mat seen through it counts as table (a different mat would show through it too)
  keep-out     = everything else (arms, grippers, cup rim/lid highlights, frame borders), dilated by KEEP_OUT_PX
  swept region = union of keep-out masks over the episode: where the arms and task objects ever are

Families (levels L1, L3 used by default; ASSUMPTION, modelled on the sim nuisances):
  appearance  recolour table pixels, texture-preserving (colour x local luminance)
  lighting    global gain + a soft elliptical shadow on table pixels (global change exempt from the mask check)
  distractor  4 blue cubes pasted on free table pixels at a clearance from the swept region (closer = higher level;
              clearance relative to the episode's free space: the arms sweep most of the table in this task)
  support     a grey patch on table pixels centred on the task area (seen through the transparent cup)

Every edit reports the fraction of changed pixels that fall inside the keep-out mask (must be 0 except lighting) and,
for distractors, their clearance to the swept region. Edited frames are JPEG-encoded (q95) and decoded again; the
clean frame goes through the same codec so drift doesn't measure compression.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

KEEP_OUT_PX = 9
APPEARANCE = {0: (235, 235, 228), 1: (158, 102, 64), 2: (77, 153, 115), 3: (217, 56, 199)}   # RGB, as in sim
LIGHTING = {0: (0.85, 0.15), 1: (0.70, 0.30), 2: (0.55, 0.45), 3: (0.42, 0.60)}             # (gain, shadow depth)
DISTRACTOR_CLEARANCE_FRAC = {0: 0.90, 1: 0.65, 2: 0.40, 3: 0.15}   # of the episode's max free-table clearance
DISTRACTOR_HALF = 16
MIN_KEEP_OUT_AREA = 300   # px; smaller 'moving' blobs are compression noise / flicker, not objects
SUPPORT = {0: (70, 0.85), 1: (90, 0.75), 2: (110, 0.65), 3: (130, 0.55)}                    # (half size px, gain)
JPEG_QUALITY = 95

CONDITIONS = {f"{fam}_L{l}": (fam, l) for fam in ("appearance", "lighting", "distractor", "support") for l in range(4)}
CONDITIONS["combo_app3_dis3_lit3_sup3"] = ("combo", 3)
DEFAULT_CONDITIONS = [f"{f}_L{l}" for f in ("appearance", "lighting", "distractor", "support") for l in (1, 3)] + \
                     ["combo_app3_dis3_lit3_sup3"]


def mat_like(bg: np.ndarray) -> np.ndarray:
    """Background pixels that look like the table mat: low saturation, neither black (arm bases) nor white."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] < 55) & (hsv[..., 2] > 25) & (hsv[..., 2] < 205)


def table_mask(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Table pixels in this frame: match the episode's static background (the camera doesn't move, the arms do) and
    the background there is mat. Arms, grippers and anything moved by them differ from the background."""
    mat = mat_like(bg)
    # auto-exposure compensation: match the frame's mat brightness to the background before differencing
    gain = float(np.median(bg[mat])) / max(float(np.median(img[mat])), 1.0)
    diff = np.abs(img.astype(np.float32) * gain - bg.astype(np.float32)).mean(-1)
    m = (diff < 30) & mat
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=8)
    small = np.isin(lab, np.nonzero(stats[:, cv2.CC_STAT_AREA] < MIN_KEEP_OUT_AREA)[0])
    return m | small


def keep_out(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    ko = (~table_mask(img, bg)).astype(np.uint8)
    return cv2.dilate(ko, np.ones((2 * KEEP_OUT_PX + 1,) * 2, np.uint8)).astype(bool)


@dataclass
class EpisodeContext:
    bg: np.ndarray           # (H, W, 3) per-episode temporal median = empty-table background
    swept: np.ndarray        # (H, W) bool: union of keep-out over the episode
    task_center: tuple       # (x, y) centroid of the swept region inside the table area
    distractor_xy: dict      # level -> list of (x, y) centres

    @staticmethod
    def build(frames: np.ndarray, rng_seed: int = 0) -> "EpisodeContext":
        bg = np.median(frames, axis=0).astype(np.uint8)
        swept = np.zeros(frames.shape[1:3], bool)
        for f in frames:
            swept |= keep_out(f, bg)
        h, w = swept.shape
        ys, xs = np.nonzero(swept[int(0.15 * h):int(0.85 * h), int(0.1 * w):int(0.9 * w)])
        center = (int(xs.mean() + 0.1 * w), int(ys.mean() + 0.15 * h)) if len(xs) else (w // 2, h // 2)
        free = (~swept).astype(np.uint8)
        free[:int(0.1 * h)] = free[int(0.95 * h):] = 0
        free[:, :int(0.05 * w)] = free[:, int(0.95 * w):] = 0
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        rng = np.random.default_rng(rng_seed)
        dxy = {}
        dmax = float(dist.max())
        for lvl, frac in DISTRACTOR_CLEARANCE_FRAC.items():
            c = max(frac * dmax, DISTRACTOR_HALF + 6)   # never closer than 6 px from the swept region
            band = np.argwhere((dist > c) & (dist < c + max(6.0, 0.15 * dmax)))
            if len(band) < 4:
                band = np.argwhere(dist > c)
            # spread 4 picks around the task centre: one per quadrant when possible
            picks = []
            for qx, qy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                sel = band[((band[:, 1] - center[0]) * qx > 0) & ((band[:, 0] - center[1]) * qy > 0)]
                sel = sel if len(sel) else band
                y, x = sel[rng.integers(len(sel))]
                picks.append((int(x), int(y)))
            dxy[lvl] = picks
        return EpisodeContext(bg, swept, center, dxy)


def _jpeg(img: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


def encode_jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()


def decode_jpeg(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]


def _appearance(img, table, lvl):
    out = img.astype(np.float32)
    lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ref = max(float(lum[table].mean()), 1.0)
    col = np.asarray(APPEARANCE[lvl], np.float32)
    out[table] = np.clip(col[None, :] * (lum[table] / ref)[:, None], 0, 255)
    return out


def _lighting(img, table, lvl, ctx):
    gain, depth = LIGHTING[lvl]
    out = img.astype(np.float32) * gain
    h, w = table.shape
    yy, xx = np.mgrid[:h, :w]
    cx, cy = ctx.task_center
    shadow = np.exp(-(((xx - cx - 0.18 * w) / (0.30 * w)) ** 2 + ((yy - cy + 0.10 * h) / (0.22 * h)) ** 2))
    out[table] *= (1 - depth * shadow[table])[:, None]
    return np.clip(out, 0, 255)


def _distractor(img, table, lvl, ctx):
    out = img.astype(np.float32)
    s = DISTRACTOR_HALF
    for x, y in ctx.distractor_xy[lvl]:
        y0, y1, x0, x1 = max(0, y - s), min(img.shape[0], y + s), max(0, x - s), min(img.shape[1], x + s)
        patch = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        patch[:] = (30, 150, 255)
        shade = np.linspace(1.05, 0.8, y1 - y0)[:, None, None]
        region = table[y0:y1, x0:x1]
        out[y0:y1, x0:x1][region] = (patch * shade)[region]
    return np.clip(out, 0, 255)


def _support(img, table, lvl, ctx):
    out = img.astype(np.float32)
    half, gain = SUPPORT[lvl]
    cx, cy = ctx.task_center
    m = np.zeros_like(table)
    m[max(0, cy - half // 2):cy + half // 2, max(0, cx - half):cx + half] = True
    sel = m & table
    out[sel] = out[sel] * gain + (1 - gain) * 150
    return np.clip(out, 0, 255)


def frame_masks(img: np.ndarray, ctx: EpisodeContext) -> tuple[np.ndarray, np.ndarray]:
    """(editable table pixels, keep-out) for one frame; compute once and reuse across conditions."""
    tm = table_mask(img, ctx.bg)
    ko = cv2.dilate((~tm).astype(np.uint8), np.ones((2 * KEEP_OUT_PX + 1,) * 2, np.uint8)).astype(bool)
    return tm & ~ko, ko


def apply(img: np.ndarray, condition: str, ctx: EpisodeContext, masks=None, roundtrip: bool = True) -> tuple[np.ndarray, dict]:
    """Return (edited frame, checks); after a JPEG round trip unless roundtrip=False. img: RGB uint8 (H, W, 3)."""
    fam, lvl = CONDITIONS[condition]
    table, ko = masks if masks is not None else frame_masks(img, ctx)   # table = editable pixels (outside the margin)
    if fam == "combo":
        out = _appearance(img, table, lvl)
        out = _support(out.astype(np.uint8), table, lvl, ctx)
        out = _distractor(out.astype(np.uint8), table, lvl, ctx)
        out = _lighting(out.astype(np.uint8), table, lvl, ctx)
    else:
        out = {"appearance": lambda: _appearance(img, table, lvl), "lighting": lambda: _lighting(img, table, lvl, ctx),
               "distractor": lambda: _distractor(img, table, lvl, ctx), "support": lambda: _support(img, table, lvl, ctx)}[fam]()
    out = out.astype(np.uint8)
    changed = np.any(np.abs(out.astype(int) - img.astype(int)) > 2, axis=-1)
    local = changed if fam != "lighting" else np.zeros_like(changed)
    if fam in ("lighting", "combo"):  # global gain is a legitimate whole-image change; check the local edits only
        local = np.any(np.abs(_appearance(img, table, lvl).astype(int) - img.astype(int)) > 2, -1) if fam == "combo" else local
    checks = dict(changed_frac=float(changed.mean()), keepout_violation_frac=float((local & ko).sum() / max(local.sum(), 1)))
    if fam in ("distractor", "combo"):
        dist = cv2.distanceTransform((~ctx.swept).astype(np.uint8), cv2.DIST_L2, 5)
        checks["distractor_clearance_px"] = float(min(dist[y, x] for x, y in ctx.distractor_xy[lvl]) - DISTRACTOR_HALF)
    return (_jpeg(out) if roundtrip else out), checks


def clean(img: np.ndarray) -> np.ndarray:
    return _jpeg(img)

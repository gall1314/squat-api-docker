# -*- coding: utf-8 -*-
# dips_analysis.py — Dips analysis with signal-based rep counting
# Architecture: 2-pass (analysis → render) with landmark snapshots
# Rep counting: combined elbow+shoulder signal with peak/valley detection

import os
import sys
import cv2
import math
import json
import time
import subprocess
import numpy as np
from collections import deque
from PIL import ImageFont, ImageDraw, Image

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None

# ===================== STYLE / FONTS =====================
BAR_BG_ALPHA         = 0.55
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)
DEPTH_RING_BG        = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"

_REF_H = 480.0
_REF_REPS_FONT_SIZE        = 28
_REF_FEEDBACK_FONT_SIZE    = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE   = 18

_FONT_CACHE = {}

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        pass
    for fb in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(fb, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def _get_font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = _load_font(path, size)
    return _FONT_CACHE[key]

def _scaled_font_size(ref_size, canvas_h):
    return max(10, int(round(ref_size * (canvas_h / _REF_H))))

# ===================== HELPERS =====================
def _ang(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cos = float(np.clip(np.dot(ba, bc) / den, -1, 1))
    return float(np.degrees(np.arccos(cos)))

def _half_floor10(x):
    return max(0.0, min(10.0, math.floor(x * 2.0) / 2.0))

def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    if abs(q - round(q)) < 1e-9:
        return str(int(round(q)))
    return f"{q:.1f}"

def score_label(s):
    s = float(s)
    if s >= 9.0: return "Excellent"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

# ===================== OVERLAY — frame-resolution, efficient =====================
def _wrap_two_lines(draw, text, font, max_width):
    words = (text or "").split()
    if not words:
        return [""]
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) == 2:
            break
    if cur and len(lines) < 2:
        lines.append(cur)
    if len(lines) >= 2 and draw.textlength(lines[-1], font=font) > max_width:
        last = lines[-1] + "\u2026"
        while draw.textlength(last, font=font) > max_width and len(last) > 1:
            last = last[:-2] + "\u2026"
        lines[-1] = last
    return lines

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Overlay at frame resolution — no 1080p canvas, much faster."""
    h, w, _ = frame.shape
    _REPS_F = _get_font(FONT_PATH, _scaled_font_size(_REF_REPS_FONT_SIZE, h))
    _FB_F   = _get_font(FONT_PATH, _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, h))
    _DL_F   = _get_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, h))
    _DP_F   = _get_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, h))

    pct = float(np.clip(depth_pct, 0, 1))
    bg_alpha = int(round(255 * BAR_BG_ALPHA))
    ref_h = max(int(h * 0.06), int(_REPS_F.size * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick = max(3, int(radius * DONUT_THICKNESS_FRAC))
    cx = w - 12 - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)

    ov = np.zeros((h, w, 4), dtype=np.uint8)
    _tmp_img = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(_tmp_img)
    txt = f"Reps: {int(reps)}"
    tw = tmp_draw.textlength(txt, font=_REPS_F)
    cv2.rectangle(ov, (0, 0), (int(tw + 20), int(_REPS_F.size + 12)), (0, 0, 0, bg_alpha), -1)

    cv2.circle(ov, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)
    sa, ea = -90, -90 + int(360 * pct)
    if ea != sa:
        cv2.ellipse(ov, (cx, cy), (radius, radius), 0, sa, ea,
                    (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)

    fb_y0 = fb_pad_x = fb_pad_y = line_gap = line_h = 0
    fb_lines = []
    if feedback:
        safe = max(6, int(h * 0.02))
        fb_pad_x = 12
        fb_pad_y = 8
        line_gap = 4
        fb_lines = _wrap_two_lines(tmp_draw, feedback, _FB_F, int(w - 2 * fb_pad_x - 20))
        line_h = _FB_F.size + 6
        block_h = 2 * fb_pad_y + len(fb_lines) * line_h + (len(fb_lines) - 1) * line_gap
        fb_y0 = max(0, h - safe - block_h)
        cv2.rectangle(ov, (0, fb_y0), (w, h - safe), (0, 0, 0, bg_alpha), -1)

    pil = Image.fromarray(ov, mode="RGBA")
    draw = ImageDraw.Draw(pil)
    draw.text((10, 5), txt, font=_REPS_F, fill=(255, 255, 255, 255))

    gap = max(2, int(radius * 0.10))
    by = cy - (_DL_F.size + gap + _DP_F.size) // 2
    lbl = "DEPTH"
    pt = f"{int(pct * 100)}%"
    draw.text((cx - int(draw.textlength(lbl, font=_DL_F) // 2), by),
              lbl, font=_DL_F, fill=(255, 255, 255, 255))
    draw.text((cx - int(draw.textlength(pt, font=_DP_F) // 2), by + _DL_F.size + gap),
              pt, font=_DP_F, fill=(255, 255, 255, 255))

    if feedback and fb_lines:
        ty = fb_y0 + fb_pad_y
        for ln in fb_lines:
            tx = max(fb_pad_x, (w - int(draw.textlength(ln, font=_FB_F))) // 2)
            draw.text((tx, ty), ln, font=_FB_F, fill=(255, 255, 255, 255))
            ty += line_h + line_gap

    ov_arr = np.array(pil)
    alpha = ov_arr[:, :, 3:4].astype(np.float32) / 255.0
    out_f = frame.astype(np.float32) * (1 - alpha) + ov_arr[:, :, [2, 1, 0]].astype(np.float32) * alpha
    return out_f.astype(np.uint8)

# ===================== BODY-ONLY SKELETON =====================
_FACE_LMS = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS = tuple()

def _init_skeleton_data():
    global _FACE_LMS, _BODY_CONNECTIONS, _BODY_POINTS
    if _BODY_CONNECTIONS or mp_pose is None:
        return
    _FACE_LMS = {
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value,
        mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
        mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
    }
    _BODY_CONNECTIONS = tuple(
        (a, b) for (a, b) in mp_pose.POSE_CONNECTIONS
        if a not in _FACE_LMS and b not in _FACE_LMS
    )
    _BODY_POINTS = tuple(sorted({i for c in _BODY_CONNECTIONS for i in c}))

def _dyn_thickness(h):
    return max(2, int(round(h * 0.003))), max(3, int(round(h * 0.005)))

def draw_body_only(frame, lms, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    line, dot = _dyn_thickness(h)
    for a, b in _BODY_CONNECTIONS:
        pa, pb = lms[a], lms[b]
        cv2.line(frame,
                 (int(pa.x * w), int(pa.y * h)),
                 (int(pb.x * w), int(pb.y * h)),
                 color, line, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = lms[i]
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), dot, color, -1, cv2.LINE_AA)
    return frame

# ===================== SNAPSHOT LANDMARKS =====================
def _snapshot_lm(lm):
    """Freeze mediapipe landmarks into plain objects we can keep in memory."""
    if lm is None:
        return None
    class _P:
        __slots__ = ("x", "y", "z", "visibility")
        def __init__(self, src):
            self.x = src.x; self.y = src.y
            self.z = src.z; self.visibility = src.visibility
    return [_P(p) for p in lm]

# ===================== VIDEO ROTATION =====================
def _get_rotation_angle(video_path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream_tags=rotate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10)
        s = r.stdout.strip()
        if s:
            return int(s)
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_streams", "-of", "json", video_path],
            capture_output=True, text=True, timeout=10)
        for st in json.loads(r.stdout).get("streams", []):
            if st.get("codec_type") == "video":
                if "rotate" in st.get("tags", {}):
                    return int(st["tags"]["rotate"])
                for sd in st.get("side_data_list", []):
                    if "rotation" in sd:
                        a = int(float(sd["rotation"]))
                        return 360 + a if a < 0 else a
    except Exception:
        pass
    return 0

def _apply_rotation(frame, rotation):
    if rotation == 90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

# ===================== SIGNAL / REP COUNTING =====================
# Landmark indices (set after mp_pose init)
if mp_pose is not None:
    _LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    _RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    _LE  = mp_pose.PoseLandmark.LEFT_ELBOW.value
    _RE  = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    _LW  = mp_pose.PoseLandmark.LEFT_WRIST.value
    _RW  = mp_pose.PoseLandmark.RIGHT_WRIST.value
    _LH  = mp_pose.PoseLandmark.LEFT_HIP.value
    _RH  = mp_pose.PoseLandmark.RIGHT_HIP.value
    _LA  = mp_pose.PoseLandmark.LEFT_ANKLE.value
    _RA  = mp_pose.PoseLandmark.RIGHT_ANKLE.value

# Smoothing
EMA_ALPHA = 0.35

# Rep counting (signal-based)
MIN_SWING       = 0.18   # minimum combined-signal swing for a real rep
MIN_REP_SEC     = 0.7    # minimum seconds between consecutive rep bottoms
MIN_REP_AMP     = 0.15   # minimum bottom->top amplitude for a valid rep
MAX_REP_SEC     = 5.0    # maximum seconds for a single rep (reject slow drift)
VIS_RATIO       = 1.5    # prefer one side if it's this much more visible

# "On dips bars" detection
# When on dips bars: ankle hangs below shoulder (large body_spread)
# When standing/walking: body_spread is small (ankles near body)
# body_spread = ankle_y - shoulder_y (normalized image coords)
ON_DIPS_SPREAD_THR = 0.08     # body_spread must exceed this to be "on bars"
ON_DIPS_MIN_FRAMES = 5         # need this many consecutive "on dips" frames to enter state

# Form feedback cues
FB_CUE_DEEPER   = "Go deeper (elbows to 90\u00b0)"
FB_CUE_LEAN     = "Reduce forward lean (keep torso upright)"
FB_CUE_LOCKOUT  = "Fully lockout elbows at top"
FB_CUE_ELBOWS_IN = "Keep elbows closer to body"

DEPTH_MIN_ANGLE   = 100.0
LOCKOUT_MIN_ANGLE = 140.0
TORSO_MAX_LEAN    = 30.0
ELBOW_FLARE_MAX   = 50.0

FB_WEIGHTS = {
    FB_CUE_DEEPER: 1.0,
    FB_CUE_LOCKOUT: 0.8,
    FB_CUE_LEAN: 0.7,
    FB_CUE_ELBOWS_IN: 0.6,
}
FORM_TIP_PRIORITY = [FB_CUE_DEEPER, FB_CUE_LOCKOUT, FB_CUE_ELBOWS_IN, FB_CUE_LEAN]

def _compute_elbow(lm):
    """Visibility-weighted elbow angle. Returns (elbow_used, eL, eR)."""
    eL = _ang((lm[_LSH].x, lm[_LSH].y), (lm[_LE].x, lm[_LE].y), (lm[_LW].x, lm[_LW].y))
    eR = _ang((lm[_RSH].x, lm[_RSH].y), (lm[_RE].x, lm[_RE].y), (lm[_RW].x, lm[_RW].y))
    vL = min(lm[_LSH].visibility, lm[_LE].visibility, lm[_LW].visibility)
    vR = min(lm[_RSH].visibility, lm[_RE].visibility, lm[_RW].visibility)
    if vL > vR * VIS_RATIO:
        elbow = eL
    elif vR > vL * VIS_RATIO:
        elbow = eR
    else:
        elbow = (eL + eR) / 2.0
    # Sanity check: reject impossibly small angles from noise
    if elbow < 30:
        elbow = max(eL, eR)
    return elbow, eL, eR

def _calc_torso_lean(lm):
    mid_sh = ((lm[_LSH].x + lm[_RSH].x) / 2.0, (lm[_LSH].y + lm[_RSH].y) / 2.0)
    mid_hp = ((lm[_LH].x + lm[_RH].x) / 2.0, (lm[_LH].y + lm[_RH].y) / 2.0)
    dx = mid_sh[0] - mid_hp[0]
    dy = mid_sh[1] - mid_hp[1]
    return abs(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9)))

def _calc_elbow_flare(lm):
    mid_sh_x = (lm[_LSH].x + lm[_RSH].x) / 2.0
    lx = abs(lm[_LE].x - mid_sh_x)
    ly = abs(lm[_LE].y - lm[_LSH].y)
    la = math.degrees(math.atan2(lx, ly + 1e-9))
    rx = abs(lm[_RE].x - mid_sh_x)
    ry = abs(lm[_RE].y - lm[_RSH].y)
    ra = math.degrees(math.atan2(rx, ry + 1e-9))
    return max(la, ra)

# =====================================================================
# PASS 1 — analysis (mediapipe), saves per-frame data
# =====================================================================
def _analysis_pass(video_path, rotation, frame_skip, scale, fps_in, model_complexity):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    frame_data = {}       # frame_idx -> {lm, reps, fb, prog}
    signal_points = []    # list of dicts for rep detection
    last_valid_lm = None

    ema_elbow = None
    ema_sh = None

    frame_idx = 0
    t0 = time.time()

    with mp_pose.Pose(model_complexity=model_complexity,
                      min_detection_confidence=0.4,
                      min_tracking_confidence=0.4) as pose:
        _init_skeleton_data()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if time.time() - t0 > 180:
                print("[DIPS] Pass1 timeout", file=sys.stderr, flush=True)
                break

            frame_idx += 1
            if frame_idx % max(1, frame_skip) != 0:
                continue

            frame = _apply_rotation(frame, rotation)
            work = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame

            res = pose.process(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))

            if not res.pose_landmarks:
                frame_data[frame_idx] = {
                    "lm": last_valid_lm, "reps": 0, "fb": None, "prog": 0.0
                }
                continue

            lm = res.pose_landmarks.landmark
            snap = _snapshot_lm(lm)
            last_valid_lm = snap

            # Compute signal
            sh_y = (lm[_LSH].y + lm[_RSH].y) / 2.0
            elbow, eL, eR = _compute_elbow(lm)

            if ema_elbow is None:
                ema_elbow = elbow
            else:
                ema_elbow = EMA_ALPHA * elbow + (1 - EMA_ALPHA) * ema_elbow
            if ema_sh is None:
                ema_sh = sh_y
            else:
                ema_sh = EMA_ALPHA * sh_y + (1 - EMA_ALPHA) * ema_sh

            # Form metrics (per-frame)
            torso_lean = _calc_torso_lean(lm)
            flare = _calc_elbow_flare(lm)

            # "On dips" detection: body_spread = ankle_y - shoulder_y
            # When on dips bars, feet hang below body (large spread)
            # When standing/walking, spread is small
            ankle_y = (lm[_LA].y + lm[_RA].y) / 2.0
            hip_y = (lm[_LH].y + lm[_RH].y) / 2.0
            body_spread = ankle_y - sh_y
            vis_ankle = min(lm[_LA].visibility, lm[_RA].visibility)
            vis_hip = min(lm[_LH].visibility, lm[_RH].visibility)

            signal_points.append({
                "f": frame_idx,
                "t": frame_idx / fps_in,
                "elbow": ema_elbow,
                "sh_y": ema_sh,
                "sh_y_raw": sh_y,
                "ankle_y": ankle_y,
                "hip_y": hip_y,
                "raw_elbow_min": min(eL, eR),
                "raw_elbow_max": max(eL, eR),
                "torso_lean": torso_lean,
                "flare": flare,
                "body_spread": body_spread,
                "vis_ankle": vis_ankle,
                "vis_hip": vis_hip,
            })

            frame_data[frame_idx] = {
                "lm": snap, "reps": 0, "fb": None, "prog": 0.0
            }

    cap.release()
    print(f"[DIPS] Pass1 done: {len(signal_points)} pose frames "
          f"out of {frame_idx} total frames", file=sys.stderr, flush=True)

    if len(signal_points) < 5:
        return None, None

    # ============================================================
    # Detect "on dips bars" state per frame using body_spread
    # Requires a streak of high-spread frames to enter state
    # (prevents single-frame spikes from triggering)
    # ============================================================
    # First pass: mark each frame as "above threshold" or not
    for s in signal_points:
        # Must have visible ankle AND high body spread
        has_visible_ankle = s["vis_ankle"] >= 0.2
        s["_spread_ok"] = has_visible_ankle and s["body_spread"] >= ON_DIPS_SPREAD_THR

    # Second pass: compute streak-based on_dips state
    streak = 0
    off_streak = 0
    on_dips_state = False
    for s in signal_points:
        if s["_spread_ok"]:
            streak += 1
            off_streak = 0
            if streak >= ON_DIPS_MIN_FRAMES:
                on_dips_state = True
        else:
            off_streak += 1
            streak = 0
            # Need sustained "off" to exit (hysteresis: 10 frames)
            if off_streak >= 10:
                on_dips_state = False
        s["on_dips"] = on_dips_state

    on_dips_frames = sum(1 for s in signal_points if s["on_dips"])
    print(f"[DIPS] On-dips state: {on_dips_frames}/{len(signal_points)} frames",
          file=sys.stderr, flush=True)

    # ============================================================
    # Build combined signal (60% elbow + 40% shoulder)
    # ============================================================
    sh_vals = [s["sh_y"] for s in signal_points]
    sh_min = min(sh_vals); sh_max = max(sh_vals)
    sh_range = max(sh_max - sh_min, 0.01)

    for s in signal_points:
        e_norm = float(np.clip((s["elbow"] - 60.0) / (180.0 - 60.0), 0, 1))
        s_norm = 1.0 - float(np.clip((s["sh_y"] - sh_min) / sh_range, 0, 1))
        s["combined"] = 0.6 * e_norm + 0.4 * s_norm

    combined_vals = [s["combined"] for s in signal_points]

    # ============================================================
    # Peak/valley detection
    # ============================================================
    direction = 0
    last_extreme = combined_vals[0]
    last_extreme_idx = 0
    swings = []

    for i in range(1, len(combined_vals)):
        v = combined_vals[i]
        if direction == 0:
            if v - last_extreme > MIN_SWING:
                direction = 1
                last_extreme_idx = 0
            elif last_extreme - v > MIN_SWING:
                direction = -1
                last_extreme_idx = 0
        elif direction == 1:  # signal going up (toward top of rep)
            if v > last_extreme:
                last_extreme = v
                last_extreme_idx = i
            elif (last_extreme - v) > MIN_SWING:
                swings.append((last_extreme_idx, signal_points[last_extreme_idx]["t"],
                               last_extreme, "top"))
                direction = -1
                last_extreme = v
                last_extreme_idx = i
        elif direction == -1:  # signal going down (toward bottom of rep)
            if v < last_extreme:
                last_extreme = v
                last_extreme_idx = i
            elif (v - last_extreme) > MIN_SWING:
                swings.append((last_extreme_idx, signal_points[last_extreme_idx]["t"],
                               last_extreme, "bottom"))
                direction = 1
                last_extreme = v
                last_extreme_idx = i

    # ============================================================
    # Count reps: bottom -> top = one complete rep (count on ascent)
    # With multiple filters to reject false positives
    # ============================================================

    # Mount detection: only applies if the person was NOT on dips bars
    # at the start of the video (low body_spread in first few frames).
    # If person is already on bars from the start, don't skip anything.
    starts_off_bars = False
    if len(signal_points) >= 3:
        first_spreads = [s["body_spread"] for s in signal_points[:5]]
        avg_first_spread = sum(first_spreads) / len(first_spreads)
        starts_off_bars = avg_first_spread < ON_DIPS_SPREAD_THR

    if starts_off_bars and swings and swings[0][3] == "bottom":
        print(f"[DIPS] Skipping first swing (mount detected at t={swings[0][1]:.2f}s, "
              f"avg_first_spread={avg_first_spread:.3f})",
              file=sys.stderr, flush=True)
        swings = swings[1:]

    raw_rep_events = []
    last_bottom_time = -999

    for i in range(len(swings) - 1):
        if swings[i][3] == "bottom" and swings[i + 1][3] == "top":
            b_idx = swings[i][0]
            t_idx = swings[i + 1][0]
            b_t = swings[i][1]
            t_t = swings[i + 1][1]
            b_val = swings[i][2]
            t_val = swings[i + 1][2]
            amp = t_val - b_val
            duration = t_t - b_t

            # Dismount detection: body spread drops after this top
            # (person getting off bars). Check raw spread, not delayed on_dips.
            lookahead_end = min(t_idx + 20, len(signal_points))
            after_top_frames = signal_points[t_idx + 1:lookahead_end]
            if after_top_frames:
                min_spread_after = min(s["body_spread"] for s in after_top_frames)
                if min_spread_after < ON_DIPS_SPREAD_THR * 0.5:
                    print(f"[DIPS] Skipping dismount rep at t={b_t:.2f}-{t_t:.2f}s "
                          f"(spread_after={min_spread_after:.3f})",
                          file=sys.stderr, flush=True)
                    continue

            # Filter 1: minimum time since last rep bottom
            if (b_t - last_bottom_time) < MIN_REP_SEC:
                continue

            # Filter 2: minimum amplitude
            if amp < MIN_REP_AMP:
                continue

            # Filter 3: maximum duration
            if duration > MAX_REP_SEC:
                continue

            b_frame = signal_points[b_idx]["f"]
            t_frame = signal_points[t_idx]["f"]
            raw_rep_events.append((b_idx, t_idx, b_frame, t_frame, b_t, t_t, amp))
            last_bottom_time = b_t

    # Filter 4: require that the bottom actually had a valid dip position
    # (elbow was bent below threshold)
    # Filter 5: person must be "on dips bars" (body spread indicates hanging)
    # Filter 6: noise detection — reject cycles where elbow readings are
    #           mostly impossibly low (<50°), indicating pose tracking failure
    #           (real dip bottoms are 60-100°, never below 50°)
    rep_events = []
    filtered_not_on_dips = 0
    filtered_elbow = 0
    filtered_noise = 0
    for evt in raw_rep_events:
        b_idx, t_idx, b_frame, t_frame, b_t, t_t, amp = evt
        cycle = signal_points[b_idx:t_idx + 1] if t_idx > b_idx else [signal_points[b_idx]]

        # Filter 4: elbow bent below 130° at some point
        min_elbow_in_cycle = min(c["raw_elbow_min"] for c in cycle)
        if min_elbow_in_cycle > 130.0:
            filtered_elbow += 1
            continue

        # Filter 5: majority of cycle frames must be in "on dips" state
        on_dips_in_cycle = sum(1 for c in cycle if c.get("on_dips", False))
        on_dips_ratio = on_dips_in_cycle / max(1, len(cycle))
        if on_dips_ratio < 0.7:
            filtered_not_on_dips += 1
            continue

        # Filter 6: noise detection — if too many frames have BOTH elbows
        # at impossibly low angles, the pose tracker is failing on both
        # sides. If only ONE side fails (raw_elbow_min low but raw_elbow_max
        # is normal), the other side still has real data and the smoothed
        # elbow is reliable. Real dip bottoms: 60-100°.
        noise_frames = sum(1 for c in cycle if c["raw_elbow_max"] < 50.0)
        noise_ratio = noise_frames / max(1, len(cycle))
        if noise_ratio > 0.30:
            filtered_noise += 1
            continue

        rep_events.append((b_idx, t_idx, b_frame, t_frame, b_t, t_t))

    rep_count = len(rep_events)
    print(f"[DIPS] Detected {rep_count} reps "
          f"(raw={len(raw_rep_events)}, "
          f"filtered: elbow={filtered_elbow}, "
          f"not_on_dips={filtered_not_on_dips}, "
          f"noise={filtered_noise})",
          file=sys.stderr, flush=True)

    # ============================================================
    # Evaluate form per rep + build feedback
    # ============================================================
    session_feedback = set()
    depth_fails = lockout_fails = lean_fails = flare_fails = 0
    rep_reports = []
    good_reps = bad_reps = 0

    for idx, (b_idx, t_idx, b_frame, t_frame, b_t, t_t) in enumerate(rep_events):
        cycle = signal_points[b_idx:t_idx + 1] if t_idx > b_idx else [signal_points[b_idx]]

        min_elbow_bottom = min(c["raw_elbow_min"] for c in cycle)
        top_start = max(0, t_idx - 2)
        top_end = min(len(signal_points), t_idx + 3)
        max_elbow_top = max(signal_points[k]["raw_elbow_max"] for k in range(top_start, top_end))

        max_lean = max(c["torso_lean"] for c in cycle)
        max_flare = max(c["flare"] for c in cycle)

        rep_tips = []
        if min_elbow_bottom > DEPTH_MIN_ANGLE:
            depth_fails += 1
            rep_tips.append(FB_CUE_DEEPER)
        if max_elbow_top < LOCKOUT_MIN_ANGLE:
            lockout_fails += 1
            rep_tips.append(FB_CUE_LOCKOUT)
        if max_lean > TORSO_MAX_LEAN:
            lean_fails += 1
            rep_tips.append(FB_CUE_LEAN)
        if max_flare > ELBOW_FLARE_MAX:
            flare_fails += 1
            rep_tips.append(FB_CUE_ELBOWS_IN)

        rep_score = 10.0 if not rep_tips else 9.5
        if rep_tips:
            bad_reps += 1
        else:
            good_reps += 1

        rep_reports.append({
            "rep_index": idx + 1,
            "score": float(rep_score),
            "good": not rep_tips,
            "bottom_elbow": float(min_elbow_bottom),
            "top_elbow": float(max_elbow_top),
            "max_lean": float(max_lean),
            "max_flare": float(max_flare),
            "bottom_time": float(b_t),
            "top_time": float(t_t),
        })

    # Session feedback: needs at least 2 fails to report
    if depth_fails >= 2: session_feedback.add(FB_CUE_DEEPER)
    if lockout_fails >= 2: session_feedback.add(FB_CUE_LOCKOUT)
    if lean_fails >= 2: session_feedback.add(FB_CUE_LEAN)
    if flare_fails >= 2: session_feedback.add(FB_CUE_ELBOWS_IN)

    # ============================================================
    # Fill frame_data with per-frame reps count, feedback, depth
    # ============================================================
    combo_min = min(combined_vals)
    combo_max = max(combined_vals)
    combo_range = max(combo_max - combo_min, 0.01)

    sig_by_frame = {s["f"]: s for s in signal_points}

    # Determine primary tip (if any)
    primary_tip = None
    if session_feedback:
        for cue in FORM_TIP_PRIORITY:
            if cue in session_feedback:
                primary_tip = cue
                break

    # Sorted top frames for progressive rep counting in video
    sorted_top_frames = sorted([tf for (_, _, _, tf, _, _) in rep_events])

    current_rep_count = 0
    next_top_idx = 0
    for fi in sorted(frame_data.keys()):
        while next_top_idx < len(sorted_top_frames) and fi >= sorted_top_frames[next_top_idx]:
            current_rep_count = next_top_idx + 1
            next_top_idx += 1

        depth = 0.0
        if fi in sig_by_frame:
            s = sig_by_frame[fi]
            depth = 1.0 - float(np.clip((s["combined"] - combo_min) / combo_range, 0, 1))

        frame_data[fi]["reps"] = current_rep_count
        frame_data[fi]["prog"] = depth
        if primary_tip and current_rep_count >= 2:
            frame_data[fi]["fb"] = primary_tip

    # ============================================================
    # Final scores + feedback list
    # ============================================================
    if rep_count == 0:
        technique_score = 0.0
    else:
        if session_feedback:
            penalty = sum(FB_WEIGHTS.get(m, 0.5) for m in session_feedback)
            penalty = max(0.5, penalty)
        else:
            penalty = 0.0
        technique_score = _half_floor10(max(0.0, 10.0 - penalty))

    fb_list = [cue for cue in FORM_TIP_PRIORITY if cue in session_feedback]
    if not fb_list and technique_score >= 10.0 - 1e-6:
        fb_list = ["Great form! Keep it up \U0001f4aa"]

    analysis = {
        "squat_count": int(rep_count),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": fb_list,
        "tips": [],
        "reps": rep_reports,
    }
    if primary_tip is not None:
        analysis["form_tip"] = primary_tip

    return analysis, frame_data

# =====================================================================
# PASS 2 — video rendering, NO mediapipe (uses snapshots from pass 1)
# =====================================================================
def _render_pass(video_path, rotation, frame_skip, output_path,
                 out_w, out_h, fps_in, frame_data):
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, effective_fps, (out_w, out_h))
    frame_idx = written = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frame = _apply_rotation(frame, rotation)
        # Resize to output dimensions (fast path if already correct)
        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        fd = frame_data.get(frame_idx)

        if fd is None:
            out.write(draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0))
        else:
            f = frame.copy()
            if fd["lm"] is not None:
                f = draw_body_only(f, fd["lm"])
            out.write(draw_overlay(f, reps=fd["reps"],
                                   feedback=fd["fb"], depth_pct=fd["prog"]))
        written += 1

    cap.release()
    out.release()
    print(f"[DIPS] Pass2 done frames_written={written}", file=sys.stderr, flush=True)

# =====================================================================
# PUBLIC ENTRY POINT
# =====================================================================
def _ret_err(msg, feedback_path):
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    return {
        "squat_count": 0, "technique_score": 0.0,
        "technique_score_display": display_half_str(0.0),
        "technique_label": score_label(0.0),
        "good_reps": 0, "bad_reps": 0,
        "feedback": [msg], "tips": [],
        "reps": [], "video_path": "", "feedback_path": feedback_path
    }

def run_dips_analysis(video_path,
                      frame_skip=3,
                      scale=0.4,
                      output_path="dips_analyzed.mp4",
                      feedback_path="dips_feedback.txt",
                      preserve_quality=False,
                      encode_crf=None,
                      return_video=True,
                      fast_mode=None):
    print(f"[DIPS] Start fast_mode={fast_mode} return_video={return_video}",
          file=sys.stderr, flush=True)

    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    # Use the SAME pose detection settings in both fast and slow modes
    # to guarantee consistent rep counts. The only difference between modes
    # is whether to render the output video.
    # (The full model at scale=0.4 produces more accurate counts than the
    #  lite model at scale=0.35 — use it everywhere.)
    model_complexity = 1
    # Keep the user-supplied scale (default 0.4), don't downscale further

    if fast_mode:
        return_video = False

    if preserve_quality:
        scale = 1.0
        frame_skip = 1
        encode_crf = 18 if encode_crf is None else encode_crf
    else:
        encode_crf = 23 if encode_crf is None else encode_crf

    print(f"[DIPS] Settings: model_complexity={model_complexity} "
          f"scale={scale} frame_skip={frame_skip}", file=sys.stderr, flush=True)

    create_video = bool(return_video) and bool(output_path)

    # Get rotation
    rotation = _get_rotation_angle(video_path)
    print(f"[DIPS] rotation={rotation}deg", file=sys.stderr, flush=True)

    # Probe video dimensions
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _ret_err("Could not open video", feedback_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames < 10:
        return _ret_err(f"Invalid video - only {total_frames} frames", feedback_path)

    out_w, out_h = (orig_h, orig_w) if rotation in (90, 270) else (orig_w, orig_h)

    # Cap output resolution for fast rendering + encoding
    # Full-res output (e.g. 720x1280) creates huge intermediate mp4v files
    # and makes the ffmpeg re-encode step extremely slow
    MAX_OUTPUT_LONG_SIDE = 540
    long_side = max(out_w, out_h)
    if long_side > MAX_OUTPUT_LONG_SIDE:
        output_scale = MAX_OUTPUT_LONG_SIDE / float(long_side)
        out_w = int(round(out_w * output_scale))
        out_h = int(round(out_h * output_scale))
        # Ensure even dimensions (required by yuv420p)
        out_w = out_w - (out_w % 2)
        out_h = out_h - (out_h % 2)

    print(f"[DIPS] {total_frames}fr @ {fps_in:.1f}fps out={out_w}x{out_h}",
          file=sys.stderr, flush=True)

    # PASS 1 — analysis
    result = _analysis_pass(video_path, rotation, frame_skip, scale, fps_in, model_complexity)
    if result is None or result[0] is None:
        return _ret_err("Not enough pose data", feedback_path)
    analysis, frame_data = result

    # Write feedback file
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {analysis['squat_count']}\n")
            f.write(f"Technique Score: {analysis['technique_score_display']} / 10  "
                    f"({analysis['technique_label']})\n")
            if analysis.get("feedback"):
                f.write("Feedback:\n")
                for ln in analysis["feedback"]:
                    f.write(f"- {ln}\n")
    except Exception as e:
        print(f"[DIPS] Warning feedback file: {e}", file=sys.stderr, flush=True)

    # PASS 2 — video rendering (NO mediapipe — uses snapshots)
    final_video_path = ""
    if create_video:
        print("[DIPS] Pass2 rendering...", file=sys.stderr, flush=True)
        _render_pass(video_path, rotation, frame_skip, output_path,
                     out_w, out_h, fps_in, frame_data)

        encoded = output_path.replace(".mp4", "_encoded.mp4")
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", output_path,
                 "-c:v", "libx264", "-preset", "ultrafast",
                 "-crf", str(int(encode_crf)),
                 "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                 "-metadata:s:v:0", "rotate=0",
                 encoded],
                capture_output=True, timeout=120)
            print(f"[DIPS] ffmpeg rc={proc.returncode}", file=sys.stderr, flush=True)
            if proc.returncode != 0:
                print(f"[DIPS] ffmpeg stderr: {proc.stderr.decode(errors='replace')[-500:]}",
                      file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[DIPS] ffmpeg exception: {e}", file=sys.stderr, flush=True)

        encoded_ok = os.path.exists(encoded) and os.path.getsize(encoded) > 1000
        if encoded_ok:
            try:
                os.remove(output_path)
            except Exception:
                pass
            final_video_path = encoded
        elif os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            final_video_path = output_path

        print(f"[DIPS] video='{final_video_path}' "
              f"exists={bool(final_video_path and os.path.exists(final_video_path))}",
              file=sys.stderr, flush=True)

    analysis["video_path"] = str(final_video_path)
    analysis["feedback_path"] = str(feedback_path)
    return analysis

def run_analysis(*args, **kwargs):
    return run_dips_analysis(*args, **kwargs)

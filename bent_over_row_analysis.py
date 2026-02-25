# -*- coding: utf-8 -*-
"""
bent_over_row_analysis.py - FIXED VERSION
Bent-Over Barbell Row — analysis pipeline matching the Squat standard:
- Same overlay look & return schema (incl. 'squat_count' key)
- Rep counting with frame skipping + soft-start gating
- Live donut progress, single feedback line, body-only skeleton
- Faststart-encoded analyzed video output
- FIXED: proper_reps now counts correctly (not overly strict)
- FIXED: form_tip now returns best tip instead of null
- FIXED: Overlay sizes now scale proportionally with frame height (matches pullup)
- FIXED: Skeleton drawing uses MediaPipe body-only approach (matches pullup/good morning)
- FIXED: Donut fills when bar is at BOTTOM (arms extended = max depth), not when pulling
- FIXED: Overlay uses frame-resolution cache (no 1080p upscale per frame)
- FIXED: Rotation metadata applied to pixels before processing/writing
- FIXED: Depth logic inverted — straight arms (bar down) = full donut
"""

import os
import cv2
import math
import time
import json
import uuid
import re
import shutil
import subprocess
from collections import deque
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None

# ================== THEME (MATCH SQUAT) ==================
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
BAR_BG_ALPHA = 0.55
DONUT_RADIUS_SCALE = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR = (40, 200, 80)
DEPTH_RING_BG = (70, 70, 70)
_REF_H = 480.0
_REF_REPS_FONT_SIZE = 28
_REF_FEEDBACK_FONT_SIZE = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE = 18

# ================== ANALYSIS PARAMS ==================
SCALE = 0.4
FRAME_SKIP = 2
CONF_DET = 0.5
CONF_TRK = 0.5
MODEL_COMPLEXITY = 1
EMA_ALPHA = 0.2
HIP_VEL_THRESH = 2.0
GLOBAL_MOTION_LOCK_FRAMES = 4
ELBOW_EXTENDED_MIN = 160.0
ELBOW_PULL_START = 150.0
ELBOW_TOP_MAX = 80.0
MIN_FRAMES_BETWEEN_REPS = 6
DONUT_FLOOR = 0.0
DONUT_CEIL = 100.0
TORSO_IDEAL_MIN = 15.0
TORSO_IDEAL_MAX = 35.0
TORSO_HIGH_WARNING = 50.0
TORSO_DRIFT_MAX = 25.0
KNEE_EXTENSION_THRESHOLD = 12.0
KNEE_EXTENSION_TIP_THRESHOLD = 9.0
MIN_REP_SAMPLES = 5
MIN_DEPTH_PCT_FOR_STABILITY = 60.0
MIN_SCORE = 4.0
MAX_SCORE = 10.0
ROM_PENALTY_STRONG = 3.0
ROM_PENALTY_LIGHT = 1.5
MOMENTUM_PENALTY = 2.0
TORSO_PENALTY = 1.0
LEGDRIVE_PENALTY = 0.8

# ================== ROTATION ==================
def _read_source_rotation_degrees(video_path):
    """Read rotation metadata (0/90/180/270). OpenCV ignores this metadata."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        if proc.returncode == 0 and proc.stdout:
            for line in proc.stdout.splitlines():
                m = re.search(r"-?\d+(?:\.\d+)?", line.strip())
                if not m:
                    continue
                try:
                    rot = int(round(float(m.group(0)))) % 360
                    return min((0, 90, 180, 270), key=lambda x: abs(x - rot))
                except Exception:
                    continue
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", video_path],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        m = re.search(r"rotate\s*:\s*(-?\d+)", proc.stderr or "")
        if m:
            rot = int(m.group(1)) % 360
            return min((0, 90, 180, 270), key=lambda x: abs(x - rot))
    except Exception:
        pass
    return 0

def _rotate_frame(frame, rotation_degrees):
    if rotation_degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

# ================== OVERLAY CACHE ==================
from PIL import ImageFont, ImageDraw, Image

_OVERLAY_CACHE = {}

def _load_font(p, s):
    try:
        return ImageFont.truetype(p, s)
    except Exception:
        pass
    for fallback in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(fallback, s)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=s)
    except TypeError:
        return ImageFont.load_default()

def _scaled_font_size(ref_size, canvas_h):
    return max(10, int(round(ref_size * (canvas_h / _REF_H))))

def _get_overlay_cache(w, h):
    key = (int(w), int(h))
    cached = _OVERLAY_CACHE.get(key)
    if cached is not None:
        return cached

    fw, fh = int(w), int(h)
    reps_font_size        = _scaled_font_size(_REF_REPS_FONT_SIZE, fh)
    feedback_font_size    = _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, fh)
    depth_label_font_size = _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, fh)
    depth_pct_font_size   = _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, fh)

    reps_font        = _load_font(FONT_PATH, reps_font_size)
    feedback_font    = _load_font(FONT_PATH, feedback_font_size)
    depth_label_font = _load_font(FONT_PATH, depth_label_font_size)
    depth_pct_font   = _load_font(FONT_PATH, depth_pct_font_size)

    _tmp   = Image.new("RGBA", (1, 1))
    _tdraw = ImageDraw.Draw(_tmp)

    sample_txt = "Reps: 00"
    pad_x = max(6, int(fw * 0.013))
    pad_y = max(4, int(fh * 0.010))
    tw    = _tdraw.textlength(sample_txt, font=reps_font)
    rep_box_w = int(tw + 2 * pad_x)
    rep_box_h = int(reps_font.size + 2 * pad_y)

    ref_h_donut = max(int(fh * 0.06), int(reps_font_size * 1.6))
    radius = int(ref_h_donut * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = max(8, int(fw * 0.016))
    cx = fw - margin - radius
    cy = max(ref_h_donut + radius // 8, radius + thick // 2 + 2)

    safe_margin = max(4, int(fh * 0.012))
    fb_pad_x    = max(8, int(fw * 0.016))
    fb_pad_y    = max(4, int(fh * 0.010))
    line_gap    = max(2, int(fh * 0.006))
    max_text_w  = fw - 2 * fb_pad_x - max(8, int(fw * 0.015))
    line_h      = feedback_font.size + max(4, int(fh * 0.008))
    block_h     = 2 * fb_pad_y + 2 * line_h + line_gap
    fb_y0       = max(0, fh - safe_margin - block_h)
    fb_y1       = fh - safe_margin

    bg_alpha_val = int(round(255 * BAR_BG_ALPHA))

    rep_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.rectangle(rep_bg, (0, 0), (rep_box_w, rep_box_h), (0, 0, 0, bg_alpha_val), -1)

    fb_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.rectangle(fb_bg, (0, fb_y0), (fw, fb_y1), (0, 0, 0, bg_alpha_val), -1)

    donut_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.circle(donut_bg, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)

    label_gap     = max(2, int(radius * 0.10))
    label_block_h = depth_label_font.size + label_gap + depth_pct_font.size
    label_by      = cy - label_block_h // 2

    cache = {
        "fw": fw, "fh": fh,
        "reps_font": reps_font, "feedback_font": feedback_font,
        "depth_label_font": depth_label_font, "depth_pct_font": depth_pct_font,
        "radius": radius, "thick": thick, "cx": cx, "cy": cy,
        "rep_box_h": rep_box_h,
        "rep_txt_x": pad_x, "rep_txt_y": pad_y - 1,
        "pad_x": pad_x, "pad_y": pad_y,
        "fb_pad_x": fb_pad_x, "fb_pad_y": fb_pad_y,
        "fb_y0": fb_y0, "fb_y1": fb_y1,
        "line_h": line_h, "line_gap": line_gap,
        "max_text_w": max_text_w,
        "label_by": label_by, "label_gap": label_gap,
        "rep_bg_pil":   Image.fromarray(rep_bg,   mode="RGBA"),
        "fb_bg_pil":    Image.fromarray(fb_bg,    mode="RGBA"),
        "donut_bg_pil": Image.fromarray(donut_bg, mode="RGBA"),
    }
    _OVERLAY_CACHE[key] = cache
    return cache

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
        last = lines[-1] + "…"
        while draw.textlength(last, font=font) > max_width and len(last) > 1:
            last = last[:-2] + "…"
        lines[-1] = last
    return lines

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """
    Fast overlay at frame resolution with cached static layers.
    depth_pct: 0.0 = bar at top (max pull / arms bent),
               1.0 = bar at bottom (arms fully extended = maximum depth).
    Donut fills clockwise as bar goes DOWN toward full extension.
    """
    h, w, _ = frame.shape
    c = _get_overlay_cache(w, h)
    pct = float(np.clip(depth_pct, 0, 1))
    fw, fh = c["fw"], c["fh"]
    cx, cy, radius, thick = c["cx"], c["cy"], c["radius"], c["thick"]

    canvas = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    canvas.alpha_composite(c["rep_bg_pil"])
    if feedback:
        canvas.alpha_composite(c["fb_bg_pil"])
    canvas.alpha_composite(c["donut_bg_pil"])

    if pct > 0:
        arc_np = np.zeros((fh, fw, 4), dtype=np.uint8)
        start_ang = -90
        end_ang = start_ang + int(360 * pct)  # clockwise fill
        cv2.ellipse(arc_np, (cx, cy), (radius, radius), 0,
                    start_ang, end_ang,
                    (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)
        canvas.alpha_composite(Image.fromarray(arc_np, mode="RGBA"))

    draw = ImageDraw.Draw(canvas)

    txt = f"Reps: {int(reps)}"
    draw.text((c["rep_txt_x"], c["rep_txt_y"]), txt,
              font=c["reps_font"], fill=(255, 255, 255, 255))

    label   = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label,   font=c["depth_label_font"])
    pw = draw.textlength(pct_txt, font=c["depth_pct_font"])
    draw.text((cx - int(lw // 2), c["label_by"]),
              label, font=c["depth_label_font"], fill=(255, 255, 255, 255))
    draw.text((cx - int(pw // 2), c["label_by"] + c["depth_label_font"].size + c["label_gap"]),
              pct_txt, font=c["depth_pct_font"], fill=(255, 255, 255, 255))

    if feedback:
        fb_lines = _wrap_two_lines(draw, feedback, c["feedback_font"], c["max_text_w"])
        ty = c["fb_y0"] + c["fb_pad_y"]
        for ln in fb_lines:
            tw2 = draw.textlength(ln, font=c["feedback_font"])
            tx  = max(c["fb_pad_x"], (fw - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=c["feedback_font"], fill=(255, 255, 255, 255))
            ty += c["line_h"] + c["line_gap"]

    canvas_np   = np.array(canvas)
    alpha       = canvas_np[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = canvas_np[:, :, [2, 1, 0]].astype(np.float32)
    result = frame.astype(np.float32) * (1.0 - alpha) + overlay_bgr * alpha
    return result.astype(np.uint8)

# ================== SKELETON ==================
mp_pose_module   = mp.solutions.pose if mp is not None else None
_FACE_LMS        = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS     = tuple()

def _init_skeleton_data():
    global _FACE_LMS, _BODY_CONNECTIONS, _BODY_POINTS
    if mp_pose_module and not _BODY_CONNECTIONS:
        _FACE_LMS = {
            mp_pose_module.PoseLandmark.NOSE.value,
            mp_pose_module.PoseLandmark.LEFT_EYE_INNER.value,
            mp_pose_module.PoseLandmark.LEFT_EYE.value,
            mp_pose_module.PoseLandmark.LEFT_EYE_OUTER.value,
            mp_pose_module.PoseLandmark.RIGHT_EYE_INNER.value,
            mp_pose_module.PoseLandmark.RIGHT_EYE.value,
            mp_pose_module.PoseLandmark.RIGHT_EYE_OUTER.value,
            mp_pose_module.PoseLandmark.LEFT_EAR.value,
            mp_pose_module.PoseLandmark.RIGHT_EAR.value,
            mp_pose_module.PoseLandmark.MOUTH_LEFT.value,
            mp_pose_module.PoseLandmark.MOUTH_RIGHT.value,
        }
        _BODY_CONNECTIONS = tuple(
            (a, b) for (a, b) in mp_pose_module.POSE_CONNECTIONS
            if a not in _FACE_LMS and b not in _FACE_LMS
        )
        _BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

def _dyn_thickness(h):
    return max(2, int(round(h * 0.002))), max(3, int(round(h * 0.004)))

def draw_body_only(frame, lms, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    line, dot = _dyn_thickness(h)
    for a, b in _BODY_CONNECTIONS:
        pa, pb = lms[a], lms[b]
        ax, ay = int(pa.x * w), int(pa.y * h)
        bx, by = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, line, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = lms[i]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), dot, color, -1, cv2.LINE_AA)
    return frame

# ================== UTILS ==================
def angle(a, b, c):
    if any(v is None for v in (a, b, c)):
        return None
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    ba_len = math.hypot(ba[0], ba[1])
    bc_len = math.hypot(bc[0], bc[1])
    if ba_len == 0 or bc_len == 0:
        return None
    cosang = (ba[0]*bc[0] + ba[1]*bc[1]) / (ba_len*bc_len + 1e-9)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def angle_between_vectors(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    v1_len = np.linalg.norm(v1)
    v2_len = np.linalg.norm(v2)
    if v1_len == 0 or v2_len == 0:
        return None
    cosang = np.dot(v1, v2) / (v1_len * v2_len)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def ema(prev, val, alpha=EMA_ALPHA):
    if prev is None:
        return val
    return prev + alpha * (val - prev)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def round_to_half(x):
    return round(x * 2) / 2.0

def technique_label(score):
    if score >= 9.5: return "Excellent"
    if score >= 8.5: return "Very good"
    if score >= 7.5: return "Good"
    if score >= 6.5: return "Fair"
    return "Needs work"

# ================== DEPTH CALCULATION ==================
def row_depth_pct(elbow_angle, ext_ref, top_ref):
    """
    Returns 0.0..1.0 where:
      0.0 = arms fully bent / max pull (bar at top, touching torso)
      1.0 = arms fully extended (bar hanging down = maximum DEPTH)

    Logic: LARGER elbow angle = more extended = deeper = more depth.
      ext_ref = largest elbow angle seen (arms straight, bar at bottom)
      top_ref = smallest elbow angle seen (max pull, bar at torso)

    This means the donut fills as the bar DESCENDS (arms extend),
    and empties as the athlete pulls up — correct for a DEPTH metric.
    """
    if elbow_angle is None or ext_ref is None or top_ref is None:
        return 0.0
    span = max(10.0, ext_ref - top_ref)
    # Larger elbow_angle (toward ext_ref) → pct closer to 1.0
    pct = (elbow_angle - top_ref) / span
    return float(clamp(pct, 0.0, 1.0))

def row_depth_pct_from_min(elbow_min, ext_ref, top_ref):
    """
    For rep summary: use the MAXIMUM elbow angle of the rep
    (the most extended point = deepest descent) to score depth.
    Note: pass elbow_MAX here, not elbow_min.
    """
    return row_depth_pct(elbow_min, ext_ref, top_ref)

# ================== CORE ANALYSIS ==================
def get_landmarks_xy(results, frame_shape):
    h, w = frame_shape[:2]
    lm = [None]*33
    if results and results.pose_landmarks:
        pts = results.pose_landmarks.landmark
        for i in range(33):
            if pts[i].visibility > 0.4:
                lm[i] = (int(pts[i].x*w), int(pts[i].y*h))
    return lm

def analyze_rep(rep_metrics):
    feedback  = []
    penalty   = 0.0
    is_proper = True

    depth_pct = rep_metrics.get("depth_pct") or 0.0
    allow_stability_cues = depth_pct >= MIN_DEPTH_PCT_FOR_STABILITY / 100.0

    torso_mean       = rep_metrics.get("torso_mean")
    torso_is_upright = torso_mean is not None and torso_mean > TORSO_IDEAL_MAX

    # ROM check
    if rep_metrics["min_elbow_angle"] is None or rep_metrics["min_elbow_angle"] > ELBOW_TOP_MAX + 10:
        penalty += ROM_PENALTY_STRONG
        feedback.append(("severe", "Pull higher — reach your torso"))
        is_proper = False
    elif rep_metrics["min_elbow_angle"] > ELBOW_TOP_MAX + 5 and not torso_is_upright:
        penalty += ROM_PENALTY_LIGHT
        feedback.append(("medium", "Try to pull a bit higher"))
    elif rep_metrics["min_elbow_angle"] > ELBOW_TOP_MAX and not torso_is_upright:
        penalty += ROM_PENALTY_LIGHT * 0.5
        feedback.append(("tip", "Aim for stronger elbow flexion at the top"))

    # Torso angle check
    if allow_stability_cues and torso_mean is not None:
        if torso_mean > TORSO_HIGH_WARNING:
            penalty += MOMENTUM_PENALTY
            feedback.append(("severe", "Lower your torso — get more horizontal"))
            is_proper = False
        elif torso_mean > TORSO_IDEAL_MAX:
            penalty += TORSO_PENALTY
            feedback.append(("medium", "Try to lower your torso closer to parallel"))
        elif torso_mean < TORSO_IDEAL_MIN:
            feedback.append(("tip", "You can raise your torso slightly"))

    # Momentum / torso drift
    if allow_stability_cues:
        if rep_metrics["torso_drift"] is not None and rep_metrics["torso_drift"] > TORSO_DRIFT_MAX * 1.5:
            penalty += MOMENTUM_PENALTY * 0.8
            feedback.append(("severe", "Avoid using momentum — keep your torso still"))
            is_proper = False
        elif rep_metrics["torso_drift"] is not None and rep_metrics["torso_drift"] > TORSO_DRIFT_MAX:
            penalty += TORSO_PENALTY * 0.6
            feedback.append(("medium", "Keep your torso angle more consistent"))
        elif rep_metrics["torso_drift"] is not None and rep_metrics["torso_drift"] > 0.7 * TORSO_DRIFT_MAX:
            feedback.append(("tip", "Try to minimize torso movement"))

    # Leg drive
    if allow_stability_cues:
        knee_ext = rep_metrics.get("knee_extension")
        if knee_ext is not None and knee_ext > KNEE_EXTENSION_THRESHOLD:
            penalty += LEGDRIVE_PENALTY
            feedback.append(("medium", "Minimize leg drive — keep knees steady"))
        elif knee_ext is not None and knee_ext > KNEE_EXTENSION_TIP_THRESHOLD:
            feedback.append(("tip", "Try to keep your knee angle constant throughout"))

    overlay_msg = None
    for sev in ("severe", "medium", "tip"):
        for (s, msg) in feedback:
            if s == sev:
                overlay_msg = msg
                break
        if overlay_msg:
            break

    session_msgs = []
    for sev in ("severe", "medium"):
        for (s, msg) in feedback:
            if s == sev:
                session_msgs.append(msg)
    tips = [msg for (s, msg) in feedback if s == "tip"]

    return penalty, overlay_msg, session_msgs, tips, is_proper

def _robust_range(values, min_samples=MIN_REP_SAMPLES):
    if not values or len(values) < min_samples:
        return None
    low  = np.percentile(values, 10)
    high = np.percentile(values, 90)
    return float(high - low)

def _safe_mean(values, min_samples=MIN_REP_SAMPLES):
    if not values or len(values) < min_samples:
        return None
    return float(sum(values) / len(values))

def faststart_remux(in_path):
    base, ext = os.path.splitext(in_path)
    out_path = f"{base}_encoded.mp4"
    try:
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-c:v", "libx264", "-preset", "fast",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-metadata:s:v:0", "rotate=0",
            out_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path
    except Exception:
        return in_path

def run_row_analysis(input_video_path, output_dir="./media",
                     user_session_id=None, return_video=True, fast_mode=None):
    if mp is None:
        return {"success": False, "error": "mediapipe not installed"}
    if fast_mode is True:
        return_video = False

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {"success": False, "error": "cannot open video"}

    source_rotation = _read_source_rotation_degrees(input_video_path)
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w    = int(width  * SCALE)
    out_h    = int(height * SCALE)

    uid          = uuid.uuid4().hex[:8]
    out_basename = f"barbell_row_{uid}_analyzed.mp4"
    out_path     = os.path.join(output_dir, out_basename)
    fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
    writer       = cv2.VideoWriter(out_path, fourcc, src_fps / max(1, FRAME_SKIP),
                                   (out_w, out_h)) if return_video else None

    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=False,
        min_detection_confidence=CONF_DET,
        min_tracking_confidence=CONF_TRK,
    )
    _init_skeleton_data()

    reps      = 0
    good_reps = 0
    bad_reps  = 0
    last_rep_frame      = -9999
    frame_idx           = -1
    hip_y_prev          = None
    hip_vel_ema         = None
    global_motion_cooldown = 0

    in_pull    = False
    rep_start  = None
    elbow_min  = None
    elbow_max  = None
    torso_vals = []
    knee_vals  = []

    ext_ref = None   # largest elbow angle seen  (arms extended, bar down)
    top_ref = None   # smallest elbow angle seen (max pull, bar at torso)

    live_depth_pct  = 0.0
    session_feedback = []
    session_tips     = []
    rep_details      = []
    overlay_msg      = ""
    start_time       = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
                continue

            if source_rotation != 0:
                frame = _rotate_frame(frame, source_rotation)
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            lm      = get_landmarks_xy(results, frame.shape)

            def pick(idx_r, idx_l):
                return lm[idx_r] if lm[idx_r] is not None else lm[idx_l]

            shoulder  = pick(12, 11)
            elbow_pt  = pick(14, 13)
            wrist     = pick(16, 15)
            hip       = pick(24, 23)
            knee      = pick(26, 25)
            ankle     = pick(28, 27)

            elbow_ang    = angle(shoulder, elbow_pt, wrist)
            torso_vector = (shoulder[0]-hip[0], shoulder[1]-hip[1]) if shoulder and hip else None
            torso_ang    = angle_between_vectors(torso_vector, (1.0, 0.0)) if torso_vector else None
            knee_ang     = angle(hip, knee, ankle)

            # Update references
            if elbow_ang is not None:
                if ext_ref is None or elbow_ang > ext_ref:
                    ext_ref = elbow_ang   # largest = arms straight
                if top_ref is None or elbow_ang < top_ref:
                    top_ref = elbow_ang   # smallest = max pull

            # Global motion gating
            hip_y = hip[1] if hip is not None else None
            if hip_y is not None:
                hip_vel     = 0.0 if hip_y_prev is None else abs(hip_y - hip_y_prev)
                hip_y_prev  = hip_y
                hip_vel_ema = ema(hip_vel_ema, hip_vel, EMA_ALPHA)
                if hip_vel_ema is not None and hip_vel_ema > HIP_VEL_THRESH:
                    global_motion_cooldown = GLOBAL_MOTION_LOCK_FRAMES
                else:
                    global_motion_cooldown = max(0, global_motion_cooldown - 1)

            # ------------------------------------------------------------------
            # Live depth donut — CORRECTED LOGIC:
            #   large elbow angle (arms extended, bar DOWN) = high depth = full donut
            #   small elbow angle (arms bent, bar UP at torso) = low depth = empty donut
            # ------------------------------------------------------------------
            live_depth_pct = row_depth_pct(elbow_ang, ext_ref, top_ref)

            # Rep state machine
            if not in_pull:
                if (elbow_ang is not None
                        and elbow_ang < ELBOW_PULL_START
                        and hip is not None
                        and global_motion_cooldown == 0
                        and (frame_idx - last_rep_frame) > MIN_FRAMES_BETWEEN_REPS):
                    in_pull    = True
                    rep_start  = frame_idx
                    elbow_min  = elbow_ang
                    elbow_max  = elbow_ang
                    torso_vals = [torso_ang] if torso_ang is not None else []
                    knee_vals  = [knee_ang]  if knee_ang  is not None else []
            else:
                if elbow_ang is not None:
                    elbow_min = min(elbow_min, elbow_ang) if elbow_min is not None else elbow_ang
                    elbow_max = max(elbow_max, elbow_ang) if elbow_max is not None else elbow_ang
                if torso_ang is not None:
                    torso_vals.append(torso_ang)
                if knee_ang is not None:
                    knee_vals.append(knee_ang)

                if (elbow_ang is not None
                        and elbow_ang >= ELBOW_EXTENDED_MIN
                        and global_motion_cooldown == 0):
                    reps          += 1
                    last_rep_frame = frame_idx

                    torso_drift    = _robust_range(torso_vals)
                    torso_mean     = _safe_mean(torso_vals)
                    knee_extension = None
                    if knee_vals and len(knee_vals) >= MIN_REP_SAMPLES:
                        n3         = len(knee_vals) // 3
                        knee_start = knee_vals[:n3]
                        knee_mid   = knee_vals[n3:]
                        if knee_start and knee_mid:
                            knee_extension = (sum(knee_mid) / len(knee_mid)
                                              - sum(knee_start) / len(knee_start))

                    # Use elbow_MAX for depth (deepest descent = most extended)
                    rep_depth_pct = row_depth_pct_from_min(elbow_max, ext_ref, top_ref)

                    rep_m = {
                        "start_frame":     rep_start,
                        "end_frame":       frame_idx,
                        "min_elbow_angle": elbow_min,
                        "max_elbow_angle": elbow_max,
                        "torso_drift":     torso_drift,
                        "knee_extension":  knee_extension,
                        "torso_mean":      torso_mean,
                        "depth_pct":       rep_depth_pct,
                    }
                    pen, ov_msg, sess_msgs, tips, is_proper = analyze_rep(rep_m)
                    rep_score = clamp(round_to_half(MAX_SCORE - pen), MIN_SCORE, MAX_SCORE)

                    if rep_score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1

                    overlay_msg = ov_msg or overlay_msg
                    for m in sess_msgs:
                        if m not in session_feedback:
                            session_feedback.append(m)
                    for t in tips:
                        if t not in session_tips:
                            session_tips.append(t)
                    rep_details.append(rep_m)

                    in_pull   = False
                    rep_start = None
                    elbow_min = elbow_max = None
                    torso_vals = []
                    knee_vals  = []

            if return_video and writer is not None:
                if results and results.pose_landmarks:
                    draw_body_only(frame, results.pose_landmarks.landmark)
                frame = draw_overlay(frame, reps, overlay_msg, live_depth_pct)
                writer.write(frame)

    finally:
        try:
            pose.close()
        except Exception:
            pass
        cap.release()
        if writer is not None:
            writer.release()

    # Session score
    score = MAX_SCORE
    for rep_m in rep_details:
        pen, _, _, _, _ = analyze_rep(rep_m)
        score -= pen / max(1, len(rep_details))
    score = clamp(round_to_half(score), MIN_SCORE, MAX_SCORE)
    label = technique_label(score)

    # Form tip
    form_tip = None
    if session_feedback:
        if any("torso" in fb.lower() or "horizontal" in fb.lower() for fb in session_feedback):
            form_tip = "Focus on hip hinge — push your hips back to get more horizontal"
        elif any("pull" in fb.lower() or "higher" in fb.lower() for fb in session_feedback):
            form_tip = "Think 'elbows to ceiling' — drive them high and back"
        elif any("momentum" in fb.lower() for fb in session_feedback):
            form_tip = "Control the descent — lower the bar slowly over 2-3 seconds"
        else:
            form_tip = session_feedback[0]
    elif session_tips:
        form_tip = "Squeeze your shoulder blades together at the top"
    else:
        form_tip = (
            "Great form! Keep it up 💪"
            if (good_reps == reps and reps > 0)
            else (
                "Try slowing down the lowering phase for better control"
                if reps > 0
                else "Complete a full rep to get feedback"
            )
        )

    out_fast      = faststart_remux(out_path) if return_video else ""
    duration_s    = time.time() - start_time

    result = {
        "success":                  True,
        "squat_count":              reps,
        "row_count":                reps,
        "good_reps":                int(good_reps),
        "bad_reps":                 int(bad_reps),
        "technique_score":          float(score),
        "technique_score_display":  f"{score:.1f}",
        "technique_label":          label,
        "feedback":                 session_feedback,
        "tips":                     session_tips,
        "form_tip":                 form_tip,
        "rep_details":              rep_details,
        "analyzed_video_path":      out_fast if return_video else "",
        "fps":                      src_fps / max(1, FRAME_SKIP),
        "duration_s":               duration_s,
    }
    return result


if __name__ == "__main__":
    test_in = "sample_row.mp4"
    if not os.path.exists(test_in):
        print(json.dumps({"success": False, "error": f"Missing {test_in} for test"},
                         ensure_ascii=False))
    else:
        res = run_row_analysis(test_in)
        print(json.dumps(res, indent=2, ensure_ascii=False))

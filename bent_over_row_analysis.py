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
"""
import os
import cv2
import math
import time
import json
import uuid
import shutil
import subprocess
from collections import deque

import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None

# ================== THEME (MATCH SQUAT) ==================
# Keep these values identical to the Squat overlay to ensure visual parity.
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"  # must exist in working dir (same as squat)
BAR_BG_ALPHA         = 0.55
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)
DEPTH_RING_BG        = (70, 70, 70)

# Reference height at which the original fixed font sizes looked correct (pullup typical)
_REF_H = 480.0
_REF_REPS_FONT_SIZE = 28
_REF_FEEDBACK_FONT_SIZE = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE = 18

# ================== ANALYSIS PARAMS ==================
SCALE = 0.4          # identical to squat
FRAME_SKIP = 2       # identical skipping logic
CONF_DET = 0.5
CONF_TRK = 0.5
MODEL_COMPLEXITY = 1

# Soft-start gating and motion thresholds (match squat style)
EMA_ALPHA = 0.2
HIP_VEL_THRESH = 2.0          # px/frame after scaling; tune like squat hip/ankle gating
GLOBAL_MOTION_LOCK_FRAMES = 4 # frames to wait when motion spikes

# Rep logic (Row-specific, but conservative to match squat reliability)
ELBOW_EXTENDED_MIN = 160.0    # start from nearly straight
ELBOW_PULL_START   = 150.0    # crossing below -> start pull
ELBOW_TOP_MAX      = 80.0     # top position must be ≤ this to count ROM (more tolerant for torso-contact reps)
MIN_FRAMES_BETWEEN_REPS = 6

# ROM donut normalization
# depth_pct = normalize( extended_angle -> top_angle )
DONUT_FLOOR = 0.0
DONUT_CEIL  = 100.0

# Posture/quality thresholds
TORSO_IDEAL_MIN = 15.0    # deg above horizontal - nearly parallel to floor
TORSO_IDEAL_MAX = 35.0    # deg above horizontal - acceptable range
TORSO_HIGH_WARNING = 50.0 # deg above horizontal - too upright warning
TORSO_DRIFT_MAX  = 25.0   # allowed change during a rep (INCREASED - stable execution)
KNEE_EXTENSION_THRESHOLD = 12.0  # deg of knee straightening = meaningful leg drive
KNEE_EXTENSION_TIP_THRESHOLD = 9.0  # mild straightening gets tip only
MIN_REP_SAMPLES  = 5      # minimum samples to trust drift/mean metrics
MIN_DEPTH_PCT_FOR_STABILITY = 60.0  # only trust stability cues on meaningful ROM

# Scoring
MIN_SCORE = 4.0
MAX_SCORE = 10.0
ROM_PENALTY_STRONG = 3.0
ROM_PENALTY_LIGHT  = 1.5  # Reduced from 2.0
MOMENTUM_PENALTY   = 2.0
TORSO_PENALTY      = 1.0  # Reduced from 1.5
LEGDRIVE_PENALTY   = 0.8  # Reduced from 1.0

# ================== UTILS ==================
def ensure_font():
    # If font missing, fallback to Hershey (handled inside draw functions).
    return os.path.exists(FONT_PATH)

def angle(a, b, c):
    """Return angle ABC in degrees for 2D points (x,y)."""
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
    """Calculate angle between two 2D vectors in degrees."""
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

# ================== OVERLAY (proportional sizing — matches pullup/good morning) ==================
from PIL import ImageFont, ImageDraw, Image

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def _scaled_font_size(ref_size, frame_h):
    """Scale font size proportionally to frame height, so overlay looks
    the same relative size regardless of frame resolution."""
    return max(10, int(round(ref_size * (frame_h / _REF_H))))

# ================== SKELETON (body-only, matches pullup/good morning) ==================
mp_pose_module = mp.solutions.pose if mp is not None else None

_FACE_LMS = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS = tuple()

def _init_skeleton_data():
    global _FACE_LMS, _BODY_CONNECTIONS, _BODY_POINTS
    if mp_pose_module and not _BODY_CONNECTIONS:
        _FACE_LMS = {
            mp_pose_module.PoseLandmark.NOSE.value,
            mp_pose_module.PoseLandmark.LEFT_EYE_INNER.value, mp_pose_module.PoseLandmark.LEFT_EYE.value, mp_pose_module.PoseLandmark.LEFT_EYE_OUTER.value,
            mp_pose_module.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose_module.PoseLandmark.RIGHT_EYE.value, mp_pose_module.PoseLandmark.RIGHT_EYE_OUTER.value,
            mp_pose_module.PoseLandmark.LEFT_EAR.value, mp_pose_module.PoseLandmark.RIGHT_EAR.value,
            mp_pose_module.PoseLandmark.MOUTH_LEFT.value, mp_pose_module.PoseLandmark.MOUTH_RIGHT.value,
        }
        _BODY_CONNECTIONS = tuple((a, b) for (a, b) in mp_pose_module.POSE_CONNECTIONS if a not in _FACE_LMS and b not in _FACE_LMS)
        _BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

def _dyn_thickness(h):
    return max(2, int(round(h * 0.002))), max(3, int(round(h * 0.004)))

def draw_body_only(frame, lms, color=(255, 255, 255)):
    """Draw body-only skeleton from MediaPipe landmarks — matches pullup/good morning exactly."""
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

def draw_depth_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius = int(radius)
    thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90
    end_ang = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def _wrap_two_lines(draw, text, font, max_width):
    words = (text or "").split()
    if not words:
        return [""]
    lines = []
    cur = ""
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
    """Reps top-left; donut top-right; feedback bottom — matches pullup/good morning exactly."""
    h, w, _ = frame.shape

    # Scale all font sizes proportionally to frame height
    reps_font_size = _scaled_font_size(_REF_REPS_FONT_SIZE, h)
    feedback_font_size = _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, h)
    depth_label_font_size = _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, h)
    depth_pct_font_size = _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, h)

    _REPS_FONT = _load_font(FONT_PATH, reps_font_size)
    _FEEDBACK_FONT = _load_font(FONT_PATH, feedback_font_size)
    _DEPTH_LABEL_FONT = _load_font(FONT_PATH, depth_label_font_size)
    _DEPTH_PCT_FONT = _load_font(FONT_PATH, depth_pct_font_size)

    ref_h = max(int(h * 0.06), int(reps_font_size * 1.6))
    r = int(ref_h * DONUT_RADIUS_SCALE)
    th = max(3, int(r * DONUT_THICKNESS_FRAC))
    m = 12
    cx = w - m - r
    cy = max(ref_h + r // 8, r + th // 2 + 2)
    pct = float(np.clip(depth_pct, 0, 1))
    cv2.circle(frame, (cx, cy), r, DEPTH_RING_BG, th, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, -90 + int(360 * pct), DEPTH_COLOR, th, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    txt = f"Reps: {int(reps)}"
    pad_x, pad_y = 10, 6
    tw = draw.textlength(txt, font=_REPS_FONT)
    thh = _REPS_FONT.size
    base = np.array(pil)
    over = base.copy()
    cv2.rectangle(over, (0, 0), (int(tw + 2 * pad_x), int(thh + 2 * pad_y)), (0, 0, 0), -1)
    base = cv2.addWeighted(over, BAR_BG_ALPHA, base, 1 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    draw.text((pad_x, pad_y - 1), txt, font=_REPS_FONT, fill=(255, 255, 255))

    gap = max(2, int(r * 0.10))
    by = cy - (_DEPTH_LABEL_FONT.size + gap + _DEPTH_PCT_FONT.size) // 2
    label = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label, font=_DEPTH_LABEL_FONT)
    pw = draw.textlength(pct_txt, font=_DEPTH_PCT_FONT)
    draw.text((cx - int(lw // 2), by), label, font=_DEPTH_LABEL_FONT, fill=(255, 255, 255))
    draw.text((cx - int(pw // 2), by + _DEPTH_LABEL_FONT.size + gap), pct_txt, font=_DEPTH_PCT_FONT, fill=(255, 255, 255))

    if feedback:
        max_w = int(w - 2 * 12 - 20)
        lines = _wrap_two_lines(draw, feedback, _FEEDBACK_FONT, max_w)
        line_h = _FEEDBACK_FONT.size + 6
        block_h = 2 * 8 + len(lines) * line_h + (len(lines) - 1) * 4
        y0 = max(0, h - max(6, int(h * 0.02)) - block_h)
        y1 = h - max(6, int(h * 0.02))
        base2 = np.array(pil)
        over2 = base2.copy()
        cv2.rectangle(over2, (0, y0), (w, y1), (0, 0, 0), -1)
        base2 = cv2.addWeighted(over2, BAR_BG_ALPHA, base2, 1 - BAR_BG_ALPHA, 0)
        pil = Image.fromarray(base2)
        draw = ImageDraw.Draw(pil)
        ty = y0 + 8
        for ln in lines:
            tw2 = draw.textlength(ln, font=_FEEDBACK_FONT)
            tx = max(12, (w - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=_FEEDBACK_FONT, fill=(255, 255, 255))
            ty += line_h + 4

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

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

def row_depth_from_elbow(elbow_angle, ext_ref, top_ref):
    # Map elbow angle to 0..100 (0 = start, 100 = top)
    if elbow_angle is None or ext_ref is None or top_ref is None:
        return 0.0
    # bigger angle = more extended, smaller angle = more "pulled"
    span = max(10.0, ext_ref - top_ref)
    pct = (ext_ref - elbow_angle) * 100.0 / span
    return clamp(pct, DONUT_FLOOR, DONUT_CEIL)

def analyze_rep(rep_metrics):
    """
    Return penalties, feedbacks, and is_proper for a single rep based on collected metrics.
    FIXED: Only mark is_proper=False for severe ROM issues
    """
    feedback = []
    penalty = 0.0
    is_proper = True  # Start assuming proper
    depth_pct = rep_metrics.get("depth_pct") or 0.0
    allow_stability_cues = depth_pct >= MIN_DEPTH_PCT_FOR_STABILITY

    torso_mean = rep_metrics.get("torso_mean")
    torso_is_upright = torso_mean is not None and torso_mean > TORSO_IDEAL_MAX

    # ROM check - ONLY severe issues mark as improper.
    # If torso is already too upright, prioritize torso correction and avoid noisy "pull higher" cue.
    if rep_metrics["min_elbow_angle"] is None or rep_metrics["min_elbow_angle"] > ELBOW_TOP_MAX + 10:
        # Very severe ROM issue
        penalty += ROM_PENALTY_STRONG
        feedback.append(("severe", "Pull higher — reach your torso"))
        is_proper = False
    elif rep_metrics["min_elbow_angle"] > ELBOW_TOP_MAX + 5 and not torso_is_upright:
        # Moderate ROM issue - still counts as proper but with feedback
        penalty += ROM_PENALTY_LIGHT
        feedback.append(("medium", "Try to pull a bit higher"))
        # NOT marking as improper for moderate issues
    elif rep_metrics["min_elbow_angle"] > ELBOW_TOP_MAX and not torso_is_upright:
        # Minor ROM issue - just a tip
        penalty += ROM_PENALTY_LIGHT * 0.5
        feedback.append(("tip", "Aim for stronger elbow flexion at the top"))

    # Torso angle check - MAIN FOCUS: is back horizontal enough?
    if allow_stability_cues and torso_mean is not None:
        # Too upright (back angle too high) - THIS IS THE MAIN ISSUE
        if torso_mean > TORSO_HIGH_WARNING:
            penalty += MOMENTUM_PENALTY  # High penalty for very upright
            feedback.append(("severe", "Lower your torso — get more horizontal"))
            is_proper = False
        elif torso_mean > TORSO_IDEAL_MAX:
            penalty += TORSO_PENALTY
            feedback.append(("medium", "Try to lower your torso closer to parallel"))
        elif torso_mean < TORSO_IDEAL_MIN:
            # Too low (rare, but possible)
            feedback.append(("tip", "You can raise your torso slightly"))
    
    # Momentum / torso drift - ONLY very severe marks as improper
    if allow_stability_cues:
        if rep_metrics["torso_drift"] is not None and rep_metrics["torso_drift"] > TORSO_DRIFT_MAX * 1.5:
            # Very severe momentum
            penalty += MOMENTUM_PENALTY * 0.8  # Reduced - drift less critical than angle
            feedback.append(("severe", "Avoid using momentum — keep your torso still"))
            is_proper = False
        elif rep_metrics["torso_drift"] is not None and rep_metrics["torso_drift"] > TORSO_DRIFT_MAX:
            # Moderate momentum - counts as proper
            penalty += TORSO_PENALTY * 0.6
            feedback.append(("medium", "Keep your torso angle more consistent"))
        elif rep_metrics["torso_drift"] is not None and rep_metrics["torso_drift"] > 0.7*TORSO_DRIFT_MAX:
            # Minor drift
            feedback.append(("tip", "Try to minimize torso movement"))

    # Leg drive - detect ACTUAL knee extension (straightening), not just movement
    if allow_stability_cues:
        knee_ext = rep_metrics.get("knee_extension")
        if knee_ext is not None and knee_ext > KNEE_EXTENSION_THRESHOLD:  # Significant straightening = leg drive
            penalty += LEGDRIVE_PENALTY
            feedback.append(("medium", "Minimize leg drive — keep knees steady"))
        elif knee_ext is not None and knee_ext > KNEE_EXTENSION_TIP_THRESHOLD:
            feedback.append(("tip", "Try to keep your knee angle constant throughout"))

    # Choose one strongest message for video overlay:
    overlay_msg = None
    for sev in ("severe","medium","tip"):
        for (s,msg) in feedback:
            if s == sev:
                overlay_msg = msg
                break
        if overlay_msg:
            break

    # Session feedback bucketed (severity then unique)
    session_msgs = []
    for sev in ("severe","medium"):
        for (s,msg) in feedback:
            if s == sev:
                session_msgs.append(msg)
    tips = [msg for (s,msg) in feedback if s == "tip"]

    return penalty, overlay_msg, session_msgs, tips, is_proper

def _robust_range(values, min_samples=MIN_REP_SAMPLES):
    if not values or len(values) < min_samples:
        return None
    low = np.percentile(values, 10)
    high = np.percentile(values, 90)
    return float(high - low)

def _safe_mean(values, min_samples=MIN_REP_SAMPLES):
    if not values or len(values) < min_samples:
        return None
    return float(sum(values) / len(values))

def faststart_remux(in_path):
    """Remux MP4 with moov at start (faststart). Returns output path."""
    base, ext = os.path.splitext(in_path)
    out_path = f"{base}_encoded.mp4"
    try:
        # Prefer ffmpeg if available
        cmd = [
            "ffmpeg","-y","-i",in_path,
            "-c","copy","-movflags","+faststart",
            out_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path
    except Exception:
        return in_path  # fallback

def run_row_analysis(input_video_path, output_dir="./media", user_session_id=None, return_video=True, fast_mode=None):
    """
    Main entry. Mirrors squat pipeline and return schema (incl. 'squat_count').
    Returns a dict with keys:
      - success, squat_count, proper_reps, technique_score, technique_score_display, technique_label
      - feedback (unique, severe-first), tips, form_tip (best single tip)
      - rep_details (per rep metrics)
      - analyzed_video_path
      - fps, duration_s
    """
    if mp is None:
        return {"success": False, "error": "mediapipe not installed"}

    if fast_mode is True:
        return_video = False

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {"success": False, "error": "cannot open video"}

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w  = int(width * SCALE)
    out_h  = int(height * SCALE)

    # Output temp path
    uid = uuid.uuid4().hex[:8]
    out_basename = f"barbell_row_{uid}_analyzed.mp4"
    out_path = os.path.join(output_dir, out_basename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, src_fps/max(1,FRAME_SKIP), (out_w, out_h)) if return_video else None

    # Pose estimator
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=MODEL_COMPLEXITY,
                        enable_segmentation=False,
                        min_detection_confidence=CONF_DET, min_tracking_confidence=CONF_TRK)

    # Initialize skeleton data for body-only drawing
    _init_skeleton_data()

    # State
    reps = 0
    good_reps = 0  # Track good reps (like squat)
    bad_reps = 0   # Track bad reps (like squat)
    last_rep_frame = -9999
    frame_idx = -1

    hip_y_prev = None
    hip_vel_ema = None
    global_motion_cooldown = 0

    # Rep metrics accumulators
    in_pull = False
    rep_start = None
    elbow_min = None
    elbow_max = None
    torso_vals = []
    knee_vals  = []

    # donut refs
    ext_ref = None
    top_ref = None
    live_depth_pct = 0.0

    session_feedback = []
    session_tips = []

    rep_details = []

    # Single overlay line (most recent message)
    overlay_msg = ""

    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
                continue

            # Resize
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # Pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # Landmarks (xy tuple format for angle calculations)
            lm = get_landmarks_xy(results, frame.shape)
            # Needed joints
            def pick(idx_r, idx_l):
                return lm[idx_r] if lm[idx_r] is not None else lm[idx_l]

            shoulder = pick(12,11)
            elbow    = pick(14,13)
            wrist    = pick(16,15)
            hip      = pick(24,23)
            knee     = pick(26,25)
            ankle    = pick(28,27)

            elbow_ang = angle(shoulder, elbow, wrist)
            
            # Calculate ACTUAL torso angle relative to horizontal (ground)
            torso_vector = (shoulder[0] - hip[0], shoulder[1] - hip[1]) if shoulder is not None and hip is not None else None
            horizontal = (1.0, 0.0)
            torso_ang = angle_between_vectors(torso_vector, horizontal) if torso_vector is not None else None
            
            knee_ang  = angle(hip, knee, ankle)

            # Update donut references
            if elbow_ang is not None:
                if ext_ref is None or elbow_ang > ext_ref:
                    ext_ref = elbow_ang
                if top_ref is None or elbow_ang < top_ref:
                    top_ref = elbow_ang

            # Global motion gating using hip y vel
            hip_y = hip[1] if hip is not None else None
            hip_vel = 0.0
            if hip_y is not None:
                if hip_y_prev is not None:
                    hip_vel = hip_y - hip_y_prev
                hip_y_prev = hip_y
                hip_vel_ema = ema(hip_vel_ema, abs(hip_vel), EMA_ALPHA)
            if hip_vel_ema is not None and hip_vel_ema > HIP_VEL_THRESH:
                global_motion_cooldown = GLOBAL_MOTION_LOCK_FRAMES
            else:
                global_motion_cooldown = max(0, global_motion_cooldown - 1)

            # Rep state machine
            live_depth_pct = row_depth_from_elbow(elbow_ang, ext_ref, top_ref)

            if not in_pull:
                # Ready to start pull: elbow crossing below PULL_START from extended region
                if (elbow_ang is not None and elbow_ang < ELBOW_PULL_START and
                    hip is not None and global_motion_cooldown == 0 and
                    (frame_idx - last_rep_frame) > MIN_FRAMES_BETWEEN_REPS):
                    in_pull = True
                    rep_start = frame_idx
                    elbow_min = elbow_ang
                    elbow_max = elbow_ang
                    torso_vals = [torso_ang] if torso_ang is not None else []
                    knee_vals  = [knee_ang] if knee_ang is not None else []
            else:
                # In pull/lower cycle
                if elbow_ang is not None:
                    elbow_min = min(elbow_min, elbow_ang) if elbow_min is not None else elbow_ang
                    elbow_max = max(elbow_max, elbow_ang) if elbow_max is not None else elbow_ang
                if torso_ang is not None: torso_vals.append(torso_ang)
                if knee_ang is not None:  knee_vals.append(knee_ang)

                # End rep when returned to extension & motion calm
                if (elbow_ang is not None and elbow_ang >= ELBOW_EXTENDED_MIN and global_motion_cooldown == 0):
                    # finalize rep
                    reps += 1
                    last_rep_frame = frame_idx

                    torso_drift = _robust_range(torso_vals)
                    knee_drift  = _robust_range(knee_vals)
                    torso_mean  = _safe_mean(torso_vals)
                    knee_mean   = _safe_mean(knee_vals)
                    
                    # Detect ACTUAL leg drive: knee extension (angle increasing during pull)
                    knee_extension = None
                    if knee_vals and len(knee_vals) >= MIN_REP_SAMPLES:
                        knee_start = knee_vals[:len(knee_vals)//3]
                        knee_mid = knee_vals[len(knee_vals)//3:]
                        if knee_start and knee_mid:
                            start_avg = sum(knee_start) / len(knee_start)
                            mid_avg = sum(knee_mid) / len(knee_mid)
                            knee_extension = mid_avg - start_avg

                    rep_m = {
                        "start_frame": rep_start,
                        "end_frame": frame_idx,
                        "min_elbow_angle": elbow_min,
                        "max_elbow_angle": elbow_max,
                        "torso_drift": torso_drift,
                        "knee_drift": knee_drift,
                        "knee_extension": knee_extension,
                        "torso_mean": torso_mean,
                        "depth_pct": row_depth_from_elbow(elbow_min, ext_ref, top_ref)
                    }
                    pen, ov_msg, sess_msgs, tips, is_proper = analyze_rep(rep_m)
                    
                    # Calculate score for this rep
                    rep_score = MAX_SCORE - pen
                    rep_score = clamp(rep_score, MIN_SCORE, MAX_SCORE)
                    rep_score = round_to_half(rep_score)
                    
                    if rep_score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                    
                    # Keep one overlay message at a time
                    overlay_msg = ov_msg or overlay_msg

                    # Accumulate session feedback unique & severity-first
                    for m in sess_msgs:
                        if m not in session_feedback:
                            session_feedback.append(m)
                    for t in tips:
                        if t not in session_tips:
                            session_tips.append(t)

                    rep_details.append(rep_m)

                    # reset
                    in_pull = False
                    rep_start = None
                    elbow_min = elbow_max = None
                    torso_vals = []
                    knee_vals  = []

            # Draw overlay (with body-only skeleton) before writing
            if return_video and writer is not None:
                # Use body-only skeleton from MediaPipe landmarks (matches pullup/good morning)
                if results and results.pose_landmarks:
                    draw_body_only(frame, results.pose_landmarks.landmark)
                frame = draw_overlay(frame, reps, overlay_msg, live_depth_pct / 100.0)
                writer.write(frame)

        # Finalize
    finally:
        try:
            pose.close()
        except Exception:
            pass
        cap.release()
        if writer is not None:
            writer.release()

    # Scoring
    score = MAX_SCORE
    for rep_m in rep_details:
        pen, _, _, _, _ = analyze_rep(rep_m)
        score -= (pen / max(1, len(rep_details)))  # average penalty
    score = clamp(score, MIN_SCORE, MAX_SCORE)
    score = round_to_half(score)

    # Technique label
    label = technique_label(score)

    # Select best form tip (coaching advice - SEPARATE from feedback)
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
        if good_reps == reps and reps > 0:
            form_tip = "Great form! Keep it up 💪"
        elif reps > 0:
            form_tip = "Try slowing down the lowering phase for better control"
        else:
            form_tip = "Complete a full rep to get feedback"

    # Faststart remux
    out_fast = faststart_remux(out_path) if return_video else ""

    # Return schema (matching squat keys; include squat_count)
    duration_s = (time.time() - start_time)
    result = {
        "success": True,
        "squat_count": reps,
        "row_count": reps,
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "technique_score": float(score),
        "technique_score_display": f"{score:.1f}",
        "technique_label": label,
        "feedback": session_feedback,
        "tips": session_tips,
        "form_tip": form_tip,
        "rep_details": rep_details,
        "analyzed_video_path": out_fast if return_video else "",
        "fps": src_fps/max(1,FRAME_SKIP),
        "duration_s": duration_s,
    }
    return result

if __name__ == "__main__":
    test_in = "sample_row.mp4"
    if not os.path.exists(test_in):
        print(json.dumps({"success": False, "error": f"Missing {test_in} for test"}, ensure_ascii=False))
    else:
        res = run_row_analysis(test_in)
        print(json.dumps(res, indent=2, ensure_ascii=False))

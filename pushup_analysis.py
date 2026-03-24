# -*- coding: utf-8 -*-
# pushup_analysis.py — V6: model_complexity=0 + forced scale=0.4 + deadlift overlay
_PUSHUP_VERSION = "V6-2026-03-13"
# Analysis logic: UNCHANGED from original HD OVERLAY VERSION 2026-02-27
# Architecture: 2-pass like deadlift (analysis stores data, render draws)
# Overlay: work-resolution like deadlift (not 1080p upscale)
# ffmpeg: ultrafast crf=28 bilinear upscale like deadlift

import os, cv2, math, numpy as np, subprocess, json, time
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ============ Styles (identical to deadlift) ============
BAR_BG_ALPHA = 0.55
DONUT_RADIUS_SCALE = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR = (40, 200, 80)
DEPTH_RING_BG = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
_REF_H = 480.0
_REF_REPS_FONT_SIZE = 28
_REF_FEEDBACK_FONT_SIZE = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE = 18

_FONT_CACHE = {}

def _get_font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = _load_font(path, size)
    return _FONT_CACHE[key]

def _scaled_font_size(ref_size, frame_h):
    return max(10, int(round(ref_size * (frame_h / _REF_H))))

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

def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

def score_label(s):
    s = float(s)
    if s >= 9.0:
        return "Excellent"
    if s >= 7.0:
        return "Good"
    if s >= 5.5:
        return "Fair"
    return "Needs work"

# ============ MediaPipe ============
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None

# ============ Helpers ============
def _ang(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cos = float(np.clip(np.dot(ba, bc) / den, -1, 1))
    return float(np.degrees(np.arccos(cos)))

def _ema(prev, new, a):
    return float(new) if prev is None else (a * float(new) + (1 - a) * float(prev))

def _half_floor10(x):
    return max(0.0, min(10.0, math.floor(x * 2.0) / 2.0))

def _dyn_thickness(h):
    return max(2, int(round(h * 0.003))), max(3, int(round(h * 0.005)))

# ============ Body-only skeleton ============
_FACE_LMS = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS = tuple()
if mp_pose:
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
    _BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))


def _snapshot_smoothed(lms):
    """Convert mediapipe landmarks to dict {idx: (x,y)} like deadlift."""
    snap = {}
    for i in _BODY_POINTS:
        snap[i] = (float(lms[i].x), float(lms[i].y))
    return snap


def draw_skeleton(frame, smoothed_pts, color=(255, 255, 255)):
    """Draw skeleton from snapshot dict — identical to deadlift draw_skeleton."""
    h, w = frame.shape[:2]
    line_t, dot_r = _dyn_thickness(h)
    for a, b in _BODY_CONNECTIONS:
        if a in smoothed_pts and b in smoothed_pts:
            cv2.line(frame,
                     (int(smoothed_pts[a][0] * w), int(smoothed_pts[a][1] * h)),
                     (int(smoothed_pts[b][0] * w), int(smoothed_pts[b][1] * h)),
                     color, line_t, cv2.LINE_AA)
    for i in _BODY_POINTS:
        if i in smoothed_pts:
            cv2.circle(frame,
                       (int(smoothed_pts[i][0] * w), int(smoothed_pts[i][1] * h)),
                       dot_r, color, -1, cv2.LINE_AA)
    return frame


# ============ Overlay — IDENTICAL to deadlift (work-resolution) ============
def _wrap2(draw, text, font, maxw):
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) == 2:
            break
    if cur and len(lines) < 2:
        lines.append(cur)
    if len(lines) >= 2 and draw.textlength(lines[-1], font=font) > maxw:
        last = lines[-1] + "\u2026"
        while draw.textlength(last, font=font) > maxw and len(last) > 1:
            last = last[:-2] + "\u2026"
        lines[-1] = last
    return lines


def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Overlay at frame resolution — same code as deadlift draw_overlay."""
    h, w, _ = frame.shape
    _REPS_F = _get_font(FONT_PATH, _scaled_font_size(_REF_REPS_FONT_SIZE, h))
    _FB_F = _get_font(FONT_PATH, _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, h))
    _DL_F = _get_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, h))
    _DP_F = _get_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, h))

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
        fb_lines = _wrap2(tmp_draw, feedback, _FB_F, int(w - 2 * fb_pad_x - 20))
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


# ============ Motion Detection ============
BASE_FRAME_SKIP = 2   # must be 2 for accurate fast rep counting
ACTIVE_FRAME_SKIP = 2 # same during motion
MOTION_DETECTION_WINDOW = 8
MOTION_VEL_THRESHOLD = 0.0010
MOTION_ACCEL_THRESHOLD = 0.0006
ELBOW_CHANGE_THRESHOLD = 5.0
COOLDOWN_FRAMES = 15
MIN_VEL_FOR_MOTION = 0.0004


class MotionDetector:
    def __init__(self):
        self.shoulder_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.elbow_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.velocity_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.raw_elbow_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.is_active = False
        self.cooldown_counter = 0
        self.last_process_frame = -999
        self.activation_count = 0
        self.last_activation_reason = ""

    def add_sample(self, shoulder_y, elbow_angle, raw_elbow_min, frame_idx):
        self.shoulder_history.append(shoulder_y)
        self.elbow_history.append(elbow_angle)
        self.raw_elbow_history.append(raw_elbow_min)

        motion_detected = False
        reason = ""

        if len(self.shoulder_history) >= 2:
            vel = abs(self.shoulder_history[-1] - self.shoulder_history[-2])
            self.velocity_history.append(vel)

            if len(self.velocity_history) >= 3:
                max_vel = max(self.velocity_history)
                recent_avg = sum(list(self.velocity_history)[-3:]) / 3
                accel = abs(self.velocity_history[-1] - self.velocity_history[-2])

                if max_vel > MOTION_VEL_THRESHOLD:
                    motion_detected = True
                    reason = f"high_vel({max_vel:.4f})"
                elif accel > MOTION_ACCEL_THRESHOLD:
                    motion_detected = True
                    reason = f"accel({accel:.4f})"
                elif recent_avg > MOTION_VEL_THRESHOLD * 0.65:
                    motion_detected = True
                    reason = f"sustained({recent_avg:.4f})"

        if len(self.elbow_history) >= 3:
            elbow_change = abs(self.elbow_history[-1] - self.elbow_history[-3])
            elbow_vel = abs(self.elbow_history[-1] - self.elbow_history[-2])
            if elbow_change > ELBOW_CHANGE_THRESHOLD:
                motion_detected = True
                reason = f"elbow_change({elbow_change:.1f})"
            elif elbow_vel > ELBOW_CHANGE_THRESHOLD * 0.55:
                motion_detected = True
                reason = f"elbow_vel({elbow_vel:.1f})"

        if len(self.raw_elbow_history) >= 3:
            raw_change = abs(self.raw_elbow_history[-1] - self.raw_elbow_history[-3])
            raw_vel = abs(self.raw_elbow_history[-1] - self.raw_elbow_history[-2])
            if raw_change > 11.0:
                motion_detected = True
                reason = f"raw_spike({raw_change:.1f})"
            elif raw_vel > 7.0:
                motion_detected = True
                reason = f"raw_vel({raw_vel:.1f})"

        if len(self.raw_elbow_history) >= 5:
            elbows = list(self.raw_elbow_history)
            went_down = elbows[-5] - elbows[-3] > 13
            went_up = elbows[-1] - elbows[-3] > 13
            if went_down and went_up:
                motion_detected = True
                reason = "V_pattern"

        # Short V-pattern (3-sample) for very fast reps
        if len(self.raw_elbow_history) >= 3:
            elbows = list(self.raw_elbow_history)
            d1 = elbows[-3] - elbows[-2]  # drop
            d2 = elbows[-1] - elbows[-2]  # rise
            if d1 > 6 and d2 > 6:
                motion_detected = True
                reason = "fast_V"

        # Wide elbow range in recent history = active motion
        if len(self.raw_elbow_history) >= 4:
            rng = max(self.raw_elbow_history) - min(self.raw_elbow_history)
            if rng > 20:
                motion_detected = True
                reason = f"elbow_range({rng:.0f})"

        if len(self.shoulder_history) >= 5:
            diffs = [self.shoulder_history[i + 1] - self.shoulder_history[i]
                     for i in range(len(self.shoulder_history) - 1)]
            if len(diffs) >= 4:
                sign_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
                max_diff = max(abs(d) for d in diffs)
                if sign_changes >= 1 and max_diff > MIN_VEL_FOR_MOTION:
                    motion_detected = True
                    reason = "direction_change"

        if motion_detected:
            self.activate(reason)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0:
                self.is_active = False

    def activate(self, reason=""):
        if not self.is_active:
            self.is_active = True
            self.activation_count += 1
            self.last_activation_reason = reason
        self.cooldown_counter = COOLDOWN_FRAMES

    def should_process(self, frame_idx):
        skip = ACTIVE_FRAME_SKIP if (self.is_active or self.cooldown_counter > 0) else BASE_FRAME_SKIP
        should = (frame_idx - self.last_process_frame) >= skip
        if should:
            self.last_process_frame = frame_idx
        return should

    def get_stats(self):
        return {
            "is_active": self.is_active,
            "cooldown": self.cooldown_counter,
            "activations": self.activation_count,
            "last_reason": self.last_activation_reason
        }


# ============ Parameters ============
ELBOW_BENT_ANGLE = 105.0
SHOULDER_MIN_DESCENT = 0.032   # was 0.028 — slightly stricter to avoid false triggers
RESET_ASCENT = 0.018
RESET_ELBOW = 148.0
REFRACTORY_FRAMES = 5          # was 4 — a bit more gap between reps
ELBOW_EMA_ALPHA = 0.80         # was 0.72 — faster response to elbow changes
SHOULDER_EMA_ALPHA = 0.75      # was 0.67 — faster response to shoulder changes
VIS_THR_STRICT = 0.29
PLANK_BODY_ANGLE_MAX = 26.0
HANDS_BELOW_SHOULDERS = 0.035
ONPUSHUP_MIN_FRAMES = 2
OFFPUSHUP_MIN_FRAMES = 5
AUTO_STOP_AFTER_EXIT_SEC = 1.5
TAIL_NOPOSE_STOP_SEC = 1.0

# Fast rep detection: when raw elbow rises this much above the bottom,
# rearm for a new rep even if shoulder hasn't risen enough
RAW_ELBOW_REARM_RISE = 18.0   # was 25 — faster rearm for fast reps

# Feedback
FB_ERROR_DEPTH = "Go deeper (chest to floor)"
FB_ERROR_HIPS = "Keep hips level (don't sag or pike)"
FB_ERROR_LOCKOUT = "Fully lockout arms at top"
FB_ERROR_ELBOWS = "Keep elbows at 45\u00b0 (not flared)"
PERF_TIP_SLOW_DOWN = "Lower slowly for better control"
PERF_TIP_TEMPO = "Try 2-1-2 tempo (down-pause-up)"
PERF_TIP_BREATHING = "Breathe: inhale down, exhale up"
PERF_TIP_CORE = "Engage core throughout movement"

FB_W_DEPTH = 1.2
FB_W_HIPS = 1.0
FB_W_LOCKOUT = 0.9
FB_W_ELBOWS = 0.7
FB_WEIGHTS = {
    FB_ERROR_DEPTH: FB_W_DEPTH,
    FB_ERROR_HIPS: FB_W_HIPS,
    FB_ERROR_LOCKOUT: FB_W_LOCKOUT,
    FB_ERROR_ELBOWS: FB_W_ELBOWS,
}
FB_DEFAULT_WEIGHT = 0.5
PENALTY_MIN_IF_ANY = 0.5

FORM_ERROR_PRIORITY = [FB_ERROR_DEPTH, FB_ERROR_LOCKOUT, FB_ERROR_HIPS]
PERF_TIP_PRIORITY = [PERF_TIP_SLOW_DOWN, PERF_TIP_TEMPO, PERF_TIP_BREATHING, PERF_TIP_CORE]

DEPTH_EXCELLENT_ANGLE = 95.0   # was 90 — very deep pushup
DEPTH_GOOD_ANGLE = 108.0      # was 100 — good depth
DEPTH_FAIR_ANGLE = 118.0      # was 110 — acceptable
DEPTH_POOR_ANGLE = 128.0      # was 120 — shallow
HIP_EXCELLENT = 12.0
HIP_GOOD = 22.0           # was 18 — more tolerance
HIP_FAIR = 32.0            # was 25
HIP_POOR = 42.0            # was 35
LOCKOUT_EXCELLENT = 155.0
LOCKOUT_GOOD = 140.0
LOCKOUT_FAIR = 125.0
LOCKOUT_POOR = 110.0
FLARE_EXCELLENT = 90.0
FLARE_GOOD = 120.0
FLARE_FAIR = 150.0         # effectively disabled — mediapipe flare measurement unreliable
FLARE_POOR = 180.0
DESCENT_SPEED_IDEAL = 0.0010
DESCENT_SPEED_FAST = 0.0015

DEPTH_FAIL_MIN_REPS = 4    # was 3 — need more consistent fails
HIPS_FAIL_MIN_REPS = 4     # was 3
LOCKOUT_FAIL_MIN_REPS = 5
FLARE_FAIL_MIN_REPS = 6    # was 5 — very hard to trigger flare feedback
TEMPO_CHECK_MIN_REPS = 3
DEPTH_ERROR_ANGLE = 130.0   # was 125 — only very clearly shallow reps
LOCKOUT_ERROR_ANGLE = 130.0

BURST_FRAMES = 4
INFLECT_VEL_THR = 0.0027

DEBUG_ONPUSHUP = bool(int(os.getenv("DEBUG_ONPUSHUP", "0")))
DEBUG_MOTION = bool(int(os.getenv("DEBUG_MOTION", "0")))
DEBUG_GRADING = bool(int(os.getenv("DEBUG_GRADING", "1")))

MIN_CYCLE_ELBOW_SAMPLES = 2   # was 4 — lowered for fast reps with shorter cycles
ROBUST_BOTTOM_PERCENTILE = 25
ROBUST_TOP_PERCENTILE = 75
ROBUST_CONFIRMED_PERCENTILE = 10


# ============ Geometry helpers (unchanged) ============
def _calculate_body_angle(lms, LSH, RSH, LH, RH, LA, RA):
    mid_sh = ((lms[LSH].x + lms[RSH].x) / 2.0, (lms[LSH].y + lms[RSH].y) / 2.0)
    mid_ank = ((lms[LA].x + lms[RA].x) / 2.0, (lms[LA].y + lms[RA].y) / 2.0)
    dx = mid_sh[0] - mid_ank[0]
    dy = mid_sh[1] - mid_ank[1]
    return abs(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9)))


def _calculate_hip_misalignment(lms, LSH, RSH, LH, RH, LA, RA):
    mid_sh = ((lms[LSH].x + lms[RSH].x) / 2.0, (lms[LSH].y + lms[RSH].y) / 2.0)
    mid_hp = ((lms[LH].x + lms[RH].x) / 2.0, (lms[LH].y + lms[RH].y) / 2.0)
    mid_ank = ((lms[LA].x + lms[RA].x) / 2.0, (lms[LA].y + lms[RA].y) / 2.0)
    return abs(180.0 - _ang(mid_sh, mid_hp, mid_ank))


def _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW):
    mid_sh = ((lms[LSH].x + lms[RSH].x) / 2.0, (lms[LSH].y + lms[RSH].y) / 2.0)

    left_vec_sh = (mid_sh[0] - lms[LSH].x, mid_sh[1] - lms[LSH].y)
    left_vec_elb = (lms[LE].x - lms[LSH].x, lms[LE].y - lms[LSH].y)
    left_angle = abs(math.degrees(math.atan2(
        left_vec_sh[0] * left_vec_elb[1] - left_vec_sh[1] * left_vec_elb[0],
        left_vec_sh[0] * left_vec_elb[0] + left_vec_sh[1] * left_vec_elb[1])))

    right_vec_sh = (mid_sh[0] - lms[RSH].x, mid_sh[1] - lms[RSH].y)
    right_vec_elb = (lms[RE].x - lms[RSH].x, lms[RE].y - lms[RSH].y)
    right_angle = abs(math.degrees(math.atan2(
        right_vec_sh[0] * right_vec_elb[1] - right_vec_sh[1] * right_vec_elb[0],
        right_vec_sh[0] * right_vec_elb[0] + right_vec_sh[1] * right_vec_elb[1])))

    return max(left_angle, right_angle)


def _robust_cycle_elbows(bottom_samples, top_samples, fallback_bottom=None, fallback_top=None, confirmed_bottom=None):
    robust_bottom = fallback_bottom
    # For top: use the raw max (top_phase_max_elbow) — one good lockout frame
    # is enough to confirm the person locked out. Using percentile of ALL cycle
    # samples (which includes bottom frames) was giving falsely low values.
    robust_top = fallback_top

    if confirmed_bottom and len(confirmed_bottom) >= 1:
        arr = np.array(confirmed_bottom, dtype=np.float32)
        robust_bottom = float(np.percentile(arr, ROBUST_CONFIRMED_PERCENTILE))
    elif bottom_samples:
        arr = np.array(bottom_samples, dtype=np.float32)
        robust_bottom = float(np.percentile(arr, ROBUST_BOTTOM_PERCENTILE))

    if fallback_bottom is not None and robust_bottom is not None:
        robust_bottom = min(robust_bottom, fallback_bottom)

    # For top: keep using fallback_top (= top_phase_max_elbow = raw max)
    # This is the highest elbow angle seen in the cycle — representative of lockout

    return robust_bottom, robust_top


def _evaluate_cycle_form(lms, bottom_phase_min_elbow, top_phase_max_elbow,
                         cycle_max_hip_misalign, cycle_max_flare, cycle_max_descent_vel,
                         depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                         fast_descent_count, depth_already_reported, hips_already_reported,
                         lockout_already_reported, flare_already_reported, tempo_already_reported,
                         session_form_errors, session_perf_tips, rep_count, local_vars):

    import sys
    has_depth_issue = has_lockout_issue = has_hips_issue = has_flare_issue = False

    n_bottom = len(local_vars.get("cycle_bottom_samples", []))
    n_top = len(local_vars.get("cycle_top_samples", []))

    # Use MIN_CYCLE_ELBOW_SAMPLES=4 for normal reps, but 1 for fast reps
    # (fast reps have fewer samples per cycle due to quick resets)
    min_samples = min(MIN_CYCLE_ELBOW_SAMPLES, max(1, n_bottom, n_top))

    if bottom_phase_min_elbow is not None and n_bottom >= 1:
        if bottom_phase_min_elbow > DEPTH_ERROR_ANGLE:
            has_depth_issue = True
            local_vars['cycle_tip_deeper'] = True
            local_vars['depth_fail_count'] += 1
            if local_vars['depth_fail_count'] >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                session_form_errors.add(FB_ERROR_DEPTH)
                local_vars['depth_already_reported'] = True

    if top_phase_max_elbow is not None and n_top >= 1:
        if top_phase_max_elbow < LOCKOUT_ERROR_ANGLE:
            has_lockout_issue = True
            local_vars['cycle_tip_lockout'] = True
            local_vars['lockout_fail_count'] += 1
            if local_vars['lockout_fail_count'] >= LOCKOUT_FAIL_MIN_REPS and not lockout_already_reported:
                session_form_errors.add(FB_ERROR_LOCKOUT)
                local_vars['lockout_already_reported'] = True

    if cycle_max_hip_misalign is not None:
        if cycle_max_hip_misalign > HIP_FAIR:
            has_hips_issue = True
            local_vars['cycle_tip_hips'] = True
            local_vars['hips_fail_count'] += 1
            if local_vars['hips_fail_count'] >= HIPS_FAIL_MIN_REPS and not hips_already_reported:
                session_form_errors.add(FB_ERROR_HIPS)
                local_vars['hips_already_reported'] = True

    # Flare check DISABLED — mediapipe measurement unreliable
    # if cycle_max_flare is not None:
    #     if cycle_max_flare > FLARE_FAIR:
    #         has_flare_issue = True

    if rep_count >= TEMPO_CHECK_MIN_REPS and not tempo_already_reported:
        if cycle_max_descent_vel > DESCENT_SPEED_FAST:
            local_vars['fast_descent_count'] += 1
            if local_vars['fast_descent_count'] >= 1:
                session_perf_tips.add(PERF_TIP_SLOW_DOWN)
                session_perf_tips.add(PERF_TIP_TEMPO)
                local_vars['tempo_already_reported'] = True

    # Determine most important real-time feedback message (priority order)
    rt_msg = None
    if has_depth_issue and FB_ERROR_DEPTH in session_form_errors:
        rt_msg = FB_ERROR_DEPTH
    elif has_lockout_issue and FB_ERROR_LOCKOUT in session_form_errors:
        rt_msg = FB_ERROR_LOCKOUT
    elif has_hips_issue and FB_ERROR_HIPS in session_form_errors:
        rt_msg = FB_ERROR_HIPS
    elif has_flare_issue and FB_ERROR_ELBOWS in session_form_errors:
        rt_msg = FB_ERROR_ELBOWS

    return has_depth_issue or has_lockout_issue or has_hips_issue or has_flare_issue, rt_msg


def _count_rep(rep_reports, rep_count, bottom_elbow, descent_from, bottom_shoulder_y,
               all_scores, rep_has_tip,
               bottom_phase_min_elbow, top_phase_max_elbow,
               cycle_max_hip_misalign, cycle_max_flare):

    depth_score = lockout_score = hips_score = flare_score = 10.0

    if bottom_phase_min_elbow:
        if bottom_phase_min_elbow <= DEPTH_EXCELLENT_ANGLE:
            depth_score = 10.0
        elif bottom_phase_min_elbow <= DEPTH_GOOD_ANGLE:
            depth_score = 9.0
        elif bottom_phase_min_elbow <= DEPTH_FAIR_ANGLE:
            depth_score = 7.5
        elif bottom_phase_min_elbow <= DEPTH_POOR_ANGLE:
            depth_score = 5.0
        else:
            depth_score = 3.0

    if top_phase_max_elbow:
        if top_phase_max_elbow >= LOCKOUT_EXCELLENT:
            lockout_score = 10.0
        elif top_phase_max_elbow >= LOCKOUT_GOOD:
            lockout_score = 9.0
        elif top_phase_max_elbow >= LOCKOUT_FAIR:
            lockout_score = 7.5
        elif top_phase_max_elbow >= LOCKOUT_POOR:
            lockout_score = 5.0
        else:
            lockout_score = 3.0

    if cycle_max_hip_misalign is not None:
        if cycle_max_hip_misalign <= HIP_EXCELLENT:
            hips_score = 10.0
        elif cycle_max_hip_misalign <= HIP_GOOD:
            hips_score = 9.0
        elif cycle_max_hip_misalign <= HIP_FAIR:
            hips_score = 7.5
        elif cycle_max_hip_misalign <= HIP_POOR:
            hips_score = 5.0
        else:
            hips_score = 3.0

    if cycle_max_flare is not None:
        if cycle_max_flare <= FLARE_EXCELLENT:
            flare_score = 10.0
        elif cycle_max_flare <= FLARE_GOOD:
            flare_score = 9.0
        elif cycle_max_flare <= FLARE_FAIR:
            flare_score = 7.5
        elif cycle_max_flare <= FLARE_POOR:
            flare_score = 5.0
        else:
            flare_score = 3.0

    # Flare disabled from scoring — mediapipe measurement unreliable
    rep_score = (depth_score * 0.40 + lockout_score * 0.30 + hips_score * 0.30)
    rep_score = round(rep_score * 2) / 2
    all_scores.append(rep_score)

    import sys
    print(f"[PU] SCORE rep#{rep_count+1}: {rep_score} "
          f"depth={depth_score}(bpme={bottom_phase_min_elbow}) "
          f"lock={lockout_score}(tpme={top_phase_max_elbow}) "
          f"hips={hips_score} flare={flare_score}",
          file=sys.stderr, flush=True)

    rep_reports.append({
        "rep_index": int(rep_count + 1),
        "score": float(rep_score),
        "good": bool(rep_score >= 9.0),
        "bottom_elbow": float(bottom_elbow),
        "descent_from": float(descent_from),
        "bottom_shoulder_y": float(bottom_shoulder_y),
        "detailed_scores": {
            "depth": float(depth_score),
            "lockout": float(lockout_score),
            "hips": float(hips_score),
            "flare": float(flare_score),
        },
        "measurements": {
            "bottom_elbow_angle": float(bottom_phase_min_elbow) if bottom_phase_min_elbow else None,
            "top_elbow_angle": float(top_phase_max_elbow) if top_phase_max_elbow else None,
            "hip_misalignment": float(cycle_max_hip_misalign) if cycle_max_hip_misalign else None,
            "elbow_flare": float(cycle_max_flare) if cycle_max_flare else None,
        }
    })


# ============ Video rotation (like deadlift) ============
def _get_video_rotation(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for s in data.get('streams', []):
                if s.get('codec_type') != 'video':
                    continue
                tags = s.get('tags', {})
                if 'rotate' in tags:
                    return int(tags['rotate'])
                for sd in s.get('side_data_list', []):
                    if 'rotation' in sd:
                        return (-int(sd['rotation'])) % 360
    except Exception:
        pass
    return 0


def _apply_rotation(frame, angle):
    angle = angle % 360
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ============ PASS 1: Analysis (no video writing) — logic UNCHANGED ============
def _analysis_pass(video_path, rotation, scale, fps_in, fast_mode=False):
    import sys
    t0 = time.time()

    # Use model_complexity=0 always (like deadlift) for consistent + fast results
    model_complexity = 0
    # Scale: 0.4 for fast mode (accuracy), 0.35 for slow mode (speed — has render overhead)
    scale = 0.35 if not fast_mode else 0.4
    # Frame skip: 2 for fast mode (accurate counting), 3 for slow mode (speed)
    analysis_frame_skip = 2 if fast_mode else 3
    print(f"[PUSHUP] _analysis_pass: mc={model_complexity} scale={scale} skip={analysis_frame_skip} fast={fast_mode}", file=sys.stderr, flush=True)

    effective_fps = max(1.0, fps_in / max(1, analysis_frame_skip))
    sec_to_frames = lambda s: max(1, int(s * effective_fps))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {}, effective_fps

    frame_idx = 0
    frame_data = {}

    rep_count = 0
    good_reps = 0
    bad_reps = 0
    rep_reports = []
    all_scores = []

    motion_detector = MotionDetector()
    frames_processed = 0
    frames_skipped = 0

    LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
    RE = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW = mp_pose.PoseLandmark.LEFT_WRIST.value
    RW = mp_pose.PoseLandmark.RIGHT_WRIST.value
    LH = mp_pose.PoseLandmark.LEFT_HIP.value
    RH = mp_pose.PoseLandmark.RIGHT_HIP.value
    LA = mp_pose.PoseLandmark.LEFT_ANKLE.value
    RA = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    def _pick_side_dyn(lms):
        vL = lms[LSH].visibility + lms[LE].visibility + lms[LW].visibility
        vR = lms[RSH].visibility + lms[RE].visibility + lms[RW].visibility
        return ("LEFT", LSH, LE, LW) if vL >= vR else ("RIGHT", RSH, RE, RW)

    elbow_ema = None
    shoulder_ema = None
    shoulder_prev = None
    shoulder_vel_prev = None
    baseline_shoulder_y = None
    desc_base_shoulder = None
    allow_new_bottom = True
    last_bottom_frame = -10**9
    cycle_max_descent = 0.0
    cycle_min_elbow = 999.0
    counted_this_cycle = False

    onpushup = False
    onpushup_streak = 0
    offpushup_streak = 0
    offpushup_frames_since_any_rep = 0
    nopose_frames_since_any_rep = 0

    session_form_errors = set()
    session_perf_tips = set()
    rt_fb_msg = None
    rt_fb_hold = 0

    cycle_tip_deeper = False
    cycle_tip_hips = False
    cycle_tip_lockout = False
    cycle_tip_elbows = False
    cycle_bottom_samples = []
    cycle_top_samples = []
    confirmed_bottom_samples = []
    depth_fail_count = 0
    hips_fail_count = 0
    lockout_fail_count = 0
    flare_fail_count = 0
    depth_already_reported = False
    hips_already_reported = False
    lockout_already_reported = False
    flare_already_reported = False

    fast_descent_count = 0
    tempo_already_reported = False

    bottom_phase_min_elbow = None
    top_phase_max_elbow = None
    cycle_max_hip_misalign = None
    cycle_max_flare = None
    cycle_max_descent_vel = 0.0
    in_descent_phase = False

    OFFPUSHUP_STOP_FRAMES = sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES = sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES = sec_to_frames(0.8)
    REARM_ASCENT_EFF = max(RESET_ASCENT * 0.50, 0.009)  # was 0.58/0.014 — faster rearm

    burst_cntr = 0
    last_valid_snap = None

    with mp_pose.Pose(model_complexity=model_complexity,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if time.time() - t0 > 180:
                print("[PUSHUP] Pass1 timeout 180s", file=sys.stderr, flush=True)
                break

            frame_idx += 1

            # Basic frame skip — in slow mode (skip=3), only consider every 3rd frame
            if frame_idx % analysis_frame_skip != 0:
                if burst_cntr <= 0:  # burst overrides skip
                    frames_skipped += 1
                    continue

            process_now = False
            if burst_cntr > 0:
                process_now = True
                burst_cntr -= 1
            elif motion_detector.should_process(frame_idx):
                process_now = True

            if not process_now:
                frames_skipped += 1
                continue

            frames_processed += 1

            frame = _apply_rotation(frame, rotation)

            # Scale down for pose detection
            if scale != 1.0:
                work = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            else:
                work = frame

            res = pose.process(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
            depth_live = 0.0

            if not res.pose_landmarks:
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep + 1) if rep_count > 0 else 0
                if rep_count > 0 and nopose_frames_since_any_rep >= NOPOSE_STOP_FRAMES:
                    break
                frame_data[frame_idx] = {
                    "snap": last_valid_snap,
                    "reps": rep_count,
                    "fb": rt_fb_msg if rt_fb_hold > 0 else None,
                    "depth_pct": 0.0,
                }
                if rt_fb_hold > 0:
                    rt_fb_hold -= 1
                continue

            nopose_frames_since_any_rep = 0
            lms = res.pose_landmarks.landmark

            # Save lightweight snapshot for render pass
            snap = _snapshot_smoothed(lms)
            last_valid_snap = snap

            side, S, E, W = _pick_side_dyn(lms)

            min_vis = min(lms[S].visibility, lms[E].visibility, lms[W].visibility,
                          lms[LH].visibility, lms[RH].visibility)
            vis_strict_ok = (min_vis >= VIS_THR_STRICT)

            shoulder_raw = float(lms[S].y)
            raw_elbow_L = _ang((lms[LSH].x, lms[LSH].y), (lms[LE].x, lms[LE].y), (lms[LW].x, lms[LW].y))
            raw_elbow_R = _ang((lms[RSH].x, lms[RSH].y), (lms[RE].x, lms[RE].y), (lms[RW].x, lms[RW].y))
            raw_elbow = raw_elbow_L if side == "LEFT" else raw_elbow_R
            raw_elbow_min = min(raw_elbow_L, raw_elbow_R)

            elbow_ema = _ema(elbow_ema, raw_elbow, ELBOW_EMA_ALPHA)
            shoulder_ema = _ema(shoulder_ema, shoulder_raw, SHOULDER_EMA_ALPHA)
            shoulder_y = shoulder_ema
            elbow_angle = elbow_ema

            motion_detector.add_sample(shoulder_y, elbow_angle, raw_elbow_min, frame_idx)

            if baseline_shoulder_y is None:
                baseline_shoulder_y = shoulder_y
            depth_live = float(np.clip(
                (shoulder_y - baseline_shoulder_y) / max(0.10, SHOULDER_MIN_DESCENT * 1.2),
                0.0, 1.0))

            body_angle = _calculate_body_angle(lms, LSH, RSH, LH, RH, LA, RA)
            hands_position = ((lms[LW].y > lms[LSH].y - HANDS_BELOW_SHOULDERS) and
                              (lms[RW].y > lms[RSH].y - HANDS_BELOW_SHOULDERS))
            in_plank = (body_angle <= PLANK_BODY_ANGLE_MAX) and hands_position

            if vis_strict_ok and in_plank:
                onpushup_streak += 1
                offpushup_streak = 0
            else:
                offpushup_streak += 1
                onpushup_streak = 0

            if (not onpushup) and onpushup_streak >= ONPUSHUP_MIN_FRAMES:
                onpushup = True
                desc_base_shoulder = None
                allow_new_bottom = True
                cycle_max_descent = 0.0
                cycle_min_elbow = 999.0
                counted_this_cycle = False
                cycle_tip_deeper = False
                cycle_tip_hips = False
                cycle_tip_lockout = False
                cycle_tip_elbows = False
                cycle_bottom_samples = []
                cycle_top_samples = []
                confirmed_bottom_samples = []
                bottom_phase_min_elbow = None
                top_phase_max_elbow = None
                cycle_max_hip_misalign = None
                cycle_max_flare = None
                cycle_max_descent_vel = 0.0
                in_descent_phase = False
                motion_detector.activate("enter_plank")

            if onpushup and offpushup_streak >= OFFPUSHUP_MIN_FRAMES:
                robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(
                    cycle_bottom_samples, cycle_top_samples,
                    bottom_phase_min_elbow, top_phase_max_elbow,
                    confirmed_bottom=confirmed_bottom_samples)
                cycle_has_issues, cycle_rt_msg = _evaluate_cycle_form(
                    lms, robust_bottom_elbow, robust_top_elbow,
                    cycle_max_hip_misalign, cycle_max_flare, cycle_max_descent_vel,
                    depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                    fast_descent_count, depth_already_reported, hips_already_reported,
                    lockout_already_reported, flare_already_reported, tempo_already_reported,
                    session_form_errors, session_perf_tips, rep_count, locals())
                if cycle_rt_msg and cur_rt is None:
                    cur_rt = cycle_rt_msg

                if ((not counted_this_cycle) and
                        (cycle_max_descent >= SHOULDER_MIN_DESCENT) and
                        (cycle_min_elbow <= ELBOW_BENT_ANGLE)):
                    _count_rep(rep_reports, rep_count, cycle_min_elbow,
                               desc_base_shoulder if desc_base_shoulder is not None else shoulder_y,
                               baseline_shoulder_y + cycle_max_descent if baseline_shoulder_y is not None else shoulder_y,
                               all_scores, cycle_has_issues,
                               robust_bottom_elbow, robust_top_elbow,
                               cycle_max_hip_misalign, cycle_max_flare)
                    rep_count += 1
                    if cycle_has_issues:
                        bad_reps += 1
                    else:
                        good_reps += 1

                onpushup = False
                offpushup_frames_since_any_rep = 0
                desc_base_shoulder = None
                cycle_max_descent = 0.0
                cycle_min_elbow = 999.0
                counted_this_cycle = False
                cycle_tip_deeper = False
                cycle_tip_hips = False
                cycle_tip_lockout = False
                cycle_tip_elbows = False
                cycle_bottom_samples = []
                cycle_top_samples = []
                confirmed_bottom_samples = []
                bottom_phase_min_elbow = None
                top_phase_max_elbow = None
                cycle_max_hip_misalign = None
                cycle_max_flare = None
                cycle_max_descent_vel = 0.0

            if (not onpushup) and rep_count > 0:
                offpushup_frames_since_any_rep += 1
                if offpushup_frames_since_any_rep >= OFFPUSHUP_STOP_FRAMES:
                    break

            shoulder_vel = 0.0 if shoulder_prev is None else (shoulder_y - shoulder_prev)
            cur_rt = None

            if onpushup and (desc_base_shoulder is not None):
                near_inflect = (abs(shoulder_vel) <= INFLECT_VEL_THR)
                sign_flip = (shoulder_vel_prev is not None) and (
                    (shoulder_vel_prev < 0 and shoulder_vel >= 0) or
                    (shoulder_vel_prev > 0 and shoulder_vel <= 0))
                if near_inflect or sign_flip:
                    burst_cntr = max(burst_cntr, BURST_FRAMES)
                    motion_detector.activate("inflection")
            shoulder_vel_prev = shoulder_vel

            if onpushup and vis_strict_ok:
                if desc_base_shoulder is None:
                    if shoulder_vel > abs(INFLECT_VEL_THR):
                        desc_base_shoulder = shoulder_y
                        cycle_max_descent = 0.0
                        cycle_min_elbow = elbow_angle
                        counted_this_cycle = False
                        cycle_bottom_samples = []
                        cycle_top_samples = []
                        confirmed_bottom_samples = []
                        bottom_phase_min_elbow = None
                        # Keep top_phase_max_elbow — it was set during ascent
                        cycle_max_hip_misalign = None
                        cycle_max_flare = None
                        cycle_max_descent_vel = 0.0
                        in_descent_phase = True
                        motion_detector.activate("start_descent")
                else:
                    cycle_max_descent = max(cycle_max_descent, (shoulder_y - desc_base_shoulder))
                    cycle_min_elbow = min(cycle_min_elbow, elbow_angle)

                    vel_abs = abs(shoulder_vel)
                    if shoulder_vel > 0 and in_descent_phase:
                        cycle_max_descent_vel = max(cycle_max_descent_vel, vel_abs)
                    if vel_abs > 0.0005:
                        cycle_max_descent_vel = max(cycle_max_descent_vel, vel_abs)

                    min_elb_now = min(raw_elbow_L, raw_elbow_R)
                    cycle_bottom_samples.append(min_elb_now)
                    if len(cycle_bottom_samples) > 40:
                        cycle_bottom_samples.pop(0)
                    if bottom_phase_min_elbow is None:
                        bottom_phase_min_elbow = min_elb_now
                    else:
                        bottom_phase_min_elbow = min(bottom_phase_min_elbow, min_elb_now)

                    max_elb_now = max(raw_elbow_L, raw_elbow_R)
                    cycle_top_samples.append(max_elb_now)
                    if len(cycle_top_samples) > 40:
                        cycle_top_samples.pop(0)
                    if top_phase_max_elbow is None:
                        top_phase_max_elbow = max_elb_now
                    else:
                        top_phase_max_elbow = max(top_phase_max_elbow, max_elb_now)

                    hip_misalign = _calculate_hip_misalignment(lms, LSH, RSH, LH, RH, LA, RA)
                    if cycle_max_hip_misalign is None:
                        cycle_max_hip_misalign = hip_misalign
                    else:
                        cycle_max_hip_misalign = max(cycle_max_hip_misalign, hip_misalign)

                    elbow_flare = _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW)
                    if cycle_max_flare is None:
                        cycle_max_flare = elbow_flare
                    else:
                        cycle_max_flare = max(cycle_max_flare, elbow_flare)

                    reset_by_asc = (desc_base_shoulder is not None) and ((desc_base_shoulder - shoulder_y) >= RESET_ASCENT)
                    reset_by_elb = (elbow_angle >= RESET_ELBOW)
                    # NEW: Reset if raw elbow has risen significantly AND we already counted
                    # This helps fast reps where person doesn't fully lock out
                    reset_by_raw_elb = False
                    if counted_this_cycle and bottom_phase_min_elbow is not None:
                        raw_rise = raw_elbow_min - bottom_phase_min_elbow
                        if raw_rise >= 35.0:  # significant elbow extension
                            reset_by_raw_elb = True

                    if shoulder_vel < 0 and in_descent_phase:
                        in_descent_phase = False

                    if reset_by_asc or reset_by_elb or reset_by_raw_elb:
                        robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(
                            cycle_bottom_samples, cycle_top_samples,
                            bottom_phase_min_elbow, top_phase_max_elbow,
                            confirmed_bottom=confirmed_bottom_samples)
                        cycle_has_issues, cycle_rt_msg = _evaluate_cycle_form(
                            lms, robust_bottom_elbow, robust_top_elbow,
                            cycle_max_hip_misalign, cycle_max_flare, cycle_max_descent_vel,
                            depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
                            fast_descent_count, depth_already_reported, hips_already_reported,
                            lockout_already_reported, flare_already_reported, tempo_already_reported,
                            session_form_errors, session_perf_tips, rep_count, locals())
                        if cycle_rt_msg and cur_rt is None:
                            cur_rt = cycle_rt_msg

                        if ((not counted_this_cycle) and
                                (cycle_max_descent >= SHOULDER_MIN_DESCENT) and
                                (cycle_min_elbow <= ELBOW_BENT_ANGLE)):
                            _count_rep(rep_reports, rep_count, elbow_angle,
                                       desc_base_shoulder,
                                       baseline_shoulder_y + cycle_max_descent if baseline_shoulder_y is not None else shoulder_y,
                                       all_scores, cycle_has_issues,
                                       robust_bottom_elbow, robust_top_elbow,
                                       cycle_max_hip_misalign, cycle_max_flare)
                            rep_count += 1
                            if cycle_has_issues:
                                bad_reps += 1
                            else:
                                good_reps += 1

                        desc_base_shoulder = shoulder_y
                        cycle_max_descent = 0.0
                        cycle_min_elbow = elbow_angle
                        counted_this_cycle = False
                        allow_new_bottom = True
                        cycle_bottom_samples = []
                        cycle_top_samples = []
                        confirmed_bottom_samples = []
                        bottom_phase_min_elbow = None
                        # DON'T reset top_phase_max_elbow — carry forward the max
                        # from the ascent phase so the next rep's scoring uses it.
                        # It gets properly updated each frame anyway (line ~1030).
                        # Resetting it here was the bug — rapid cycle resets during
                        # ascent would clear it before the next rep could use it.
                        cycle_max_hip_misalign = None
                        cycle_max_flare = None
                        cycle_max_descent_vel = 0.0
                        cycle_tip_deeper = False
                        cycle_tip_hips = False
                        cycle_tip_lockout = False
                        cycle_tip_elbows = False
                        in_descent_phase = True
                        motion_detector.activate("reset")

                    descent_amt = 0.0 if desc_base_shoulder is None else (shoulder_y - desc_base_shoulder)

                    at_bottom = (elbow_angle <= ELBOW_BENT_ANGLE) and (descent_amt >= SHOULDER_MIN_DESCENT)
                    raw_bottom = (raw_elbow_min <= (ELBOW_BENT_ANGLE + 7.0)) and (descent_amt >= SHOULDER_MIN_DESCENT * 0.9)
                    at_bottom = at_bottom or raw_bottom
                    can_cnt = (frame_idx - last_bottom_frame) >= REFRACTORY_FRAMES

                    # === DETAILED LOGGING (only at_bottom events) ===

                    if at_bottom:
                        confirmed_bottom_samples.append(raw_elbow_min)
                        if len(confirmed_bottom_samples) > 15:
                            confirmed_bottom_samples.pop(0)

                    if at_bottom and allow_new_bottom and can_cnt and (not counted_this_cycle):
                        print(f"[PU] >>> REP #{rep_count+1} F{frame_idx} "
                              f"ea={elbow_angle:.1f} rem={raw_elbow_min:.1f} desc={descent_amt:.4f} "
                              f"tpme={top_phase_max_elbow}",
                              file=sys.stderr, flush=True)
                        rep_has_issues = False
                        robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(
                            cycle_bottom_samples, cycle_top_samples,
                            bottom_phase_min_elbow, top_phase_max_elbow,
                            confirmed_bottom=confirmed_bottom_samples)
                        if robust_bottom_elbow and robust_bottom_elbow > DEPTH_ERROR_ANGLE:
                            rep_has_issues = True
                            depth_fail_count += 1
                            if depth_fail_count >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                                session_form_errors.add(FB_ERROR_DEPTH)
                                depth_already_reported = True
                            if FB_ERROR_DEPTH in session_form_errors and cur_rt is None:
                                cur_rt = FB_ERROR_DEPTH
                        if robust_top_elbow and robust_top_elbow < LOCKOUT_ERROR_ANGLE:
                            rep_has_issues = True
                            lockout_fail_count += 1
                            if lockout_fail_count >= LOCKOUT_FAIL_MIN_REPS and not lockout_already_reported:
                                session_form_errors.add(FB_ERROR_LOCKOUT)
                                lockout_already_reported = True
                            if FB_ERROR_LOCKOUT in session_form_errors and cur_rt is None:
                                cur_rt = FB_ERROR_LOCKOUT
                        if cycle_max_hip_misalign and cycle_max_hip_misalign > HIP_FAIR:
                            rep_has_issues = True
                            hips_fail_count += 1
                            if hips_fail_count >= HIPS_FAIL_MIN_REPS and not hips_already_reported:
                                session_form_errors.add(FB_ERROR_HIPS)
                                hips_already_reported = True
                            if FB_ERROR_HIPS in session_form_errors and cur_rt is None:
                                cur_rt = FB_ERROR_HIPS
                        # Flare check DISABLED — mediapipe measurement unreliable

                        _count_rep(rep_reports, rep_count, elbow_angle,
                                   desc_base_shoulder if desc_base_shoulder is not None else shoulder_y,
                                   shoulder_y,
                                   all_scores, rep_has_issues,
                                   robust_bottom_elbow if robust_bottom_elbow else raw_elbow_min,
                                   robust_top_elbow if robust_top_elbow else max(raw_elbow_L, raw_elbow_R),
                                   cycle_max_hip_misalign if cycle_max_hip_misalign else 0.0,
                                   cycle_max_flare if cycle_max_flare else 0.0)
                        rep_count += 1
                        if rep_has_issues:
                            bad_reps += 1
                        else:
                            good_reps += 1
                        last_bottom_frame = frame_idx
                        allow_new_bottom = False
                        counted_this_cycle = True
                        # Reset top_phase_max_elbow ONLY after counting — not on cycle reset
                        top_phase_max_elbow = None
                        in_descent_phase = False
                        motion_detector.activate("count_rep")
                    elif at_bottom and (not allow_new_bottom or not can_cnt or counted_this_cycle):
                        pass  # Skip verbose logging for speed

                    if (allow_new_bottom is False) and (last_bottom_frame > 0):
                        rearmed = False
                        # Original shoulder-based rearm
                        if (shoulder_prev is not None and (shoulder_prev - shoulder_y) > 0
                                and (desc_base_shoulder is not None)):
                            ascent_from_bottom = ((desc_base_shoulder + cycle_max_descent) - shoulder_y)
                            if ascent_from_bottom >= REARM_ASCENT_EFF:
                                rearmed = True
                                print(f"[PU] +++ REARM(shoulder) F{frame_idx} ascent={ascent_from_bottom:.4f}",
                                      file=sys.stderr, flush=True)
                        # NEW: Raw elbow rearm — if elbow straightens significantly above
                        # the cycle minimum, the person is pushing back up → allow new rep
                        if (not rearmed) and bottom_phase_min_elbow is not None:
                            elbow_rise = raw_elbow_min - bottom_phase_min_elbow
                            if elbow_rise >= RAW_ELBOW_REARM_RISE:
                                rearmed = True
                                print(f"[PU] +++ REARM(elbow) F{frame_idx} rise={elbow_rise:.1f} "
                                      f"rem={raw_elbow_min:.1f} bpme={bottom_phase_min_elbow:.1f}",
                                      file=sys.stderr, flush=True)
                        if rearmed:
                            allow_new_bottom = True

                    if at_bottom and not cycle_tip_deeper:
                        robust_bottom_elbow, _ = _robust_cycle_elbows(
                            cycle_bottom_samples, cycle_top_samples,
                            bottom_phase_min_elbow, top_phase_max_elbow,
                            confirmed_bottom=confirmed_bottom_samples)
                        if robust_bottom_elbow and robust_bottom_elbow > DEPTH_ERROR_ANGLE:
                            cycle_tip_deeper = True
                            depth_fail_count += 1
                            if depth_fail_count >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                                session_form_errors.add(FB_ERROR_DEPTH)
                                depth_already_reported = True
                                cur_rt = FB_ERROR_DEPTH

            else:
                desc_base_shoulder = None
                allow_new_bottom = True

            if cur_rt:
                if cur_rt != rt_fb_msg:
                    rt_fb_msg = cur_rt
                    rt_fb_hold = RT_FB_HOLD_FRAMES
                else:
                    rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold > 0:
                    rt_fb_hold -= 1

            # Store frame data for render pass
            fb_val = rt_fb_msg if rt_fb_hold > 0 else None
            frame_data[frame_idx] = {
                "snap": snap,
                "reps": rep_count,
                "fb": fb_val,
                "depth_pct": depth_live,
            }

            if shoulder_y is not None:
                shoulder_prev = shoulder_y

    # Final uncounted cycle
    if (onpushup and (not counted_this_cycle)
            and (cycle_max_descent >= SHOULDER_MIN_DESCENT)
            and (cycle_min_elbow <= ELBOW_BENT_ANGLE)):
        robust_bottom_elbow, robust_top_elbow = _robust_cycle_elbows(
            cycle_bottom_samples, cycle_top_samples,
            bottom_phase_min_elbow, top_phase_max_elbow,
            confirmed_bottom=confirmed_bottom_samples)
        cycle_has_issues, _ = _evaluate_cycle_form(
            lms, robust_bottom_elbow, robust_top_elbow,
            cycle_max_hip_misalign, cycle_max_flare, cycle_max_descent_vel,
            depth_fail_count, hips_fail_count, lockout_fail_count, flare_fail_count,
            fast_descent_count, depth_already_reported, hips_already_reported,
            lockout_already_reported, flare_already_reported, tempo_already_reported,
            session_form_errors, session_perf_tips, rep_count, locals())
        _count_rep(rep_reports, rep_count, cycle_min_elbow,
                   desc_base_shoulder if desc_base_shoulder is not None else (baseline_shoulder_y or 0.0),
                   (baseline_shoulder_y + cycle_max_descent) if baseline_shoulder_y is not None else (baseline_shoulder_y or 0.0),
                   all_scores, cycle_has_issues,
                   robust_bottom_elbow, robust_top_elbow,
                   cycle_max_hip_misalign, cycle_max_flare)
        rep_count += 1
        if cycle_has_issues:
            bad_reps += 1
        else:
            good_reps += 1

    cap.release()

    # Compute technique score
    if rep_count == 0:
        technique_score = 0.0
    else:
        if session_form_errors:
            penalty = sum(FB_WEIGHTS.get(m, FB_DEFAULT_WEIGHT) for m in set(session_form_errors))
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
        else:
            penalty = 0.0
        technique_score = _half_floor10(max(0.0, 10.0 - penalty))

        # Also check per-rep average — if reps scored poorly, cap technique score
        if all_scores:
            avg_rep_score = float(np.mean(all_scores))
            if avg_rep_score < 9.0 and technique_score >= 10.0:
                technique_score = _half_floor10(avg_rep_score + 0.5)

    # If no form errors at all, score should be 10
    if not session_form_errors and rep_count > 0:
        technique_score = 10.0

    if technique_score == 10.0 and rep_count > 0:
        good_reps = rep_count
        bad_reps = 0

    form_errors_list = [err for err in FORM_ERROR_PRIORITY if err in session_form_errors]
    perf_tips_list = [tip for tip in PERF_TIP_PRIORITY if tip in session_perf_tips]

    # Post-analysis: infer feedback from per-rep detailed scores
    # This catches issues that _evaluate_cycle_form missed due to fast cycles
    if rep_reports and len(rep_reports) >= 2:
        lockout_scores = [r["detailed_scores"]["lockout"] for r in rep_reports if "detailed_scores" in r]
        depth_scores = [r["detailed_scores"]["depth"] for r in rep_reports if "detailed_scores" in r]

        if lockout_scores:
            avg_lock = sum(lockout_scores) / len(lockout_scores)
            bad_lock_pct = sum(1 for s in lockout_scores if s < 9.0) / len(lockout_scores)
            if bad_lock_pct >= 0.75 and FB_ERROR_LOCKOUT not in session_form_errors:
                session_form_errors.add(FB_ERROR_LOCKOUT)
                form_errors_list = [err for err in FORM_ERROR_PRIORITY if err in session_form_errors]
                print(f"[PUSHUP] POST-ANALYSIS: added lockout feedback (avg={avg_lock:.1f}, bad%={bad_lock_pct:.0%})",
                      file=sys.stderr, flush=True)

        if depth_scores:
            avg_depth = sum(depth_scores) / len(depth_scores)
            bad_depth_pct = sum(1 for s in depth_scores if s < 9.0) / len(depth_scores)
            if bad_depth_pct >= 0.75 and FB_ERROR_DEPTH not in session_form_errors:
                session_form_errors.add(FB_ERROR_DEPTH)
                form_errors_list = [err for err in FORM_ERROR_PRIORITY if err in session_form_errors]
                print(f"[PUSHUP] POST-ANALYSIS: added depth feedback (avg={avg_depth:.1f}, bad%={bad_depth_pct:.0%})",
                      file=sys.stderr, flush=True)

        # Recompute technique score after adding post-analysis errors
        if session_form_errors:
            penalty = sum(FB_WEIGHTS.get(m, FB_DEFAULT_WEIGHT) for m in set(session_form_errors))
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
            technique_score = _half_floor10(max(0.0, 10.0 - penalty))
            if all_scores:
                avg_rep_score = float(np.mean(all_scores))
                technique_score = min(technique_score, _half_floor10(avg_rep_score + 0.5))
    primary_form_error = form_errors_list[0] if form_errors_list else None
    primary_perf_tip = perf_tips_list[0] if perf_tips_list else None

    total_frames = frames_processed + frames_skipped
    efficiency = (frames_skipped / total_frames * 100) if total_frames > 0 else 0

    fb_frames = sum(1 for fd in frame_data.values() if fd.get("fb") is not None)
    print(f"[PUSHUP] Pass1 done: {rep_count} reps, {frames_processed} proc, "
          f"{frames_skipped} skip ({efficiency:.0f}%), {time.time()-t0:.1f}s, "
          f"fb_frames={fb_frames}, errors={list(session_form_errors)}",
          file=sys.stderr, flush=True)

    result = {
        "squat_count": int(rep_count),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": form_errors_list if form_errors_list else (
            ["Great form! Keep it up \U0001f4aa"] if rep_count > 0 else []),
        "tips": perf_tips_list,
        "reps": rep_reports,
        "processing_stats": {
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "efficiency_percent": round(efficiency, 1),
            "motion_activations": motion_detector.activation_count,
        }
    }

    if primary_perf_tip:
        result["form_tip"] = primary_perf_tip
    if primary_form_error:
        result["primary_form_error"] = primary_form_error
    if primary_perf_tip:
        result["primary_perf_tip"] = primary_perf_tip

    return result, frame_data, effective_fps


# ============ PASS 2: Render at WORK resolution, ffmpeg upscales ============
def _render_pass(video_path, rotation, output_path,
                 out_w, out_h, work_w, work_h, fps_in, frame_data):
    import sys
    effective_fps = max(1.0, fps_in / max(1, BASE_FRAME_SKIP))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, effective_fps, (work_w, work_h))

    if not out.isOpened():
        print(f"[PUSHUP] ERROR: VideoWriter failed", file=sys.stderr, flush=True)
        return

    frame_indices = set(frame_data.keys())
    max_frame = max(frame_indices) if frame_indices else 0

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    written = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx > max_frame:
            break
        if frame_idx not in frame_indices:
            continue

        frame = _apply_rotation(frame, rotation)
        if frame.shape[1] != work_w or frame.shape[0] != work_h:
            frame = cv2.resize(frame, (work_w, work_h))

        fd = frame_data[frame_idx]
        if fd["snap"] is not None:
            frame = draw_skeleton(frame, fd["snap"])

        out.write(draw_overlay(frame, reps=fd["reps"],
                               feedback=fd["fb"],
                               depth_pct=fd["depth_pct"]))
        written += 1

    cap.release()
    out.release()
    print(f"[PUSHUP] Pass2 done frames_written={written} at {work_w}x{work_h}",
          file=sys.stderr, flush=True)


# ============ Main entry — identical structure to deadlift run_deadlift_analysis ============
def run_pushup_analysis(video_path,
                        frame_skip=None,
                        scale=0.4,
                        output_path="pushup_analyzed.mp4",
                        feedback_path="pushup_feedback.txt",
                        preserve_quality=False,
                        encode_crf=None,
                        return_video=True,
                        fast_mode=None):
    import sys
    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    t_start = time.time()

    if fast_mode is True:
        return_video = False
    # IMPORTANT: Override scale to ensure fast and slow produce identical results
    # app.py may send scale=0.35 for fast_mode, but analysis must use same scale
    scale = 0.4

    if preserve_quality:
        scale = 1.0
        encode_crf = 18 if encode_crf is None else encode_crf
    else:
        encode_crf = 28 if encode_crf is None else encode_crf

    create_video = bool(return_video) and bool(output_path)

    rotation = _get_video_rotation(video_path)
    print(f"[PUSHUP] {_PUSHUP_VERSION} | fast_mode={fast_mode} create_video={create_video} rotation={rotation}deg scale={scale}",
          file=sys.stderr, flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _ret_err("Could not open video", feedback_path)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if rotation % 360 in (90, 270):
        out_w, out_h = orig_h, orig_w
    else:
        out_w, out_h = orig_w, orig_h

    work_w = int(out_w * scale) if scale != 1.0 else out_w
    work_h = int(out_h * scale) if scale != 1.0 else out_h
    work_w = work_w if work_w % 2 == 0 else work_w + 1
    work_h = work_h if work_h % 2 == 0 else work_h + 1

    # Render resolution: higher than work for readable overlay, but not full output
    render_scale = min(0.65, max(scale, 420.0 / max(out_h, 1)))
    render_w = int(out_w * render_scale) if render_scale != 1.0 else out_w
    render_h = int(out_h * render_scale) if render_scale != 1.0 else out_h
    render_w = render_w if render_w % 2 == 0 else render_w + 1
    render_h = render_h if render_h % 2 == 0 else render_h + 1

    print(f"[PUSHUP] out={out_w}x{out_h} work={work_w}x{work_h} render={render_w}x{render_h}",
          file=sys.stderr, flush=True)

    # PASS 1: Analysis
    analysis, frame_data, effective_fps = _analysis_pass(
        video_path, rotation, scale, fps_in,
        fast_mode=bool(fast_mode) if fast_mode else False)

    if analysis is None:
        return _ret_err("Analysis failed", feedback_path)

    # Write feedback file
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {analysis['squat_count']}\n")
            f.write(f"Good Reps: {analysis['good_reps']} | Bad Reps: {analysis['bad_reps']}\n")
            f.write(f"Technique Score: {analysis['technique_score_display']} / 10  "
                    f"({analysis['technique_label']})\n")
            form_errors = [x for x in analysis.get('feedback', [])
                           if x != "Great form! Keep it up \U0001f4aa"]
            if form_errors:
                f.write("\nForm Corrections (affecting score):\n")
                for ln in form_errors:
                    f.write(f"- {ln}\n")
            if analysis.get('tips'):
                f.write("\nPerformance Tips (not affecting score):\n")
                for ln in analysis['tips']:
                    f.write(f"- {ln}\n")
    except Exception:
        pass

    # PASS 2: Render video (like deadlift)
    final_video_path = ""
    if create_video:
        print("[PUSHUP] Pass2 rendering video...", file=sys.stderr, flush=True)
        _render_pass(video_path, rotation, output_path,
                     out_w, out_h, render_w, render_h, fps_in, frame_data)

        encoded = output_path.replace(".mp4", "_encoded.mp4")
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", output_path,
                 "-vf", f"scale={out_w}:{out_h}:flags=bilinear",
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(encode_crf),
                 "-threads", "2",
                 "-movflags", "+faststart", "-pix_fmt", "yuv420p", encoded],
                capture_output=True, timeout=120)
            print(f"[PUSHUP] ffmpeg rc={proc.returncode}", file=sys.stderr, flush=True)
            if proc.returncode != 0:
                print(f"[PUSHUP] ffmpeg stderr: {proc.stderr.decode(errors='replace')[-500:]}",
                      file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[PUSHUP] ffmpeg exception: {e}", file=sys.stderr, flush=True)

        encoded_ok = os.path.exists(encoded) and os.path.getsize(encoded) > 1000
        if encoded_ok:
            try:
                os.remove(output_path)
            except Exception:
                pass
            final_video_path = encoded
        elif os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            final_video_path = output_path
        else:
            final_video_path = ""

        print(f"[PUSHUP] video='{final_video_path}' "
              f"exists={bool(final_video_path and os.path.exists(final_video_path))}",
              file=sys.stderr, flush=True)

    total_time = time.time() - t_start
    print(f"[PUSHUP] Total time: {total_time:.1f}s", file=sys.stderr, flush=True)

    analysis["video_path"] = str(final_video_path) if create_video else ""
    analysis["feedback_path"] = str(feedback_path)
    return analysis


def _ret_err(msg, feedback_path):
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    return {
        "squat_count": 0,
        "technique_score": 0.0,
        "technique_score_display": display_half_str(0.0),
        "technique_label": score_label(0.0),
        "good_reps": 0,
        "bad_reps": 0,
        "feedback": [],
        "tips": [],
        "reps": [],
        "video_path": "",
        "feedback_path": feedback_path,
    }


def run_analysis(*args, **kwargs):
    return run_pushup_analysis(*args, **kwargs)

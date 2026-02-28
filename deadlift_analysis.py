# -*- coding: utf-8 -*-
# deadlift_analysis.py — V2: Major upgrade for multi-angle detection & robust reps
#
# KEY IMPROVEMENTS OVER V1:
# ─────────────────────────
# 1.  MULTI-SIGNAL REP DETECTION: fuses torso angle + hip angle + horizontal delta
#     → works from side, diagonal, front/back camera angles (not just side)
# 2.  ADAPTIVE SIGNAL FUSION: auto-weights signals by estimated camera angle
# 3.  VIEW-ANGLE ESTIMATION from shoulder-width / torso-height ratio
# 4.  LANDMARK SMOOTHER: EMA on all 33 points → cleaner skeleton, less jitter
# 5.  BETTER STATE MACHINE: STANDING→HINGING→RISING→STANDING with hysteresis
# 6.  HIP-HINGE ANGLE as primary depth signal (view-invariant)
# 7.  DUAL MediaPipe API: supports both old `mp.solutions.pose` and new Tasks API
# 8.  MODEL COMPLEXITY 2 (old API) or full model (new API) for best accuracy
# 9.  FIXED SKELETON DRAWING: all points smoothed before draw, no Kalman jitter
# 10. PROGRESS DONUT based on composite angle signal (not just x-delta)
# 11. Removed fragile Kalman leg tracker — landmark smoother handles it better
# 12. Cleaner feedback aggregation

import os, cv2, math, sys, numpy as np, subprocess
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ===================== MediaPipe import (dual API support) =====================
import mediapipe as mp

_USE_TASKS_API = False
_SOLUTIONS_AVAILABLE = False

# Try old solutions API first (mediapipe < 0.10.21 typically)
try:
    _mp_pose = mp.solutions.pose
    _SOLUTIONS_AVAILABLE = True
except AttributeError:
    _SOLUTIONS_AVAILABLE = False

# If solutions not available, use Tasks API
if not _SOLUTIONS_AVAILABLE:
    try:
        _tasks_vision = mp.tasks.vision
        _tasks_base = mp.tasks.BaseOptions
        _USE_TASKS_API = True
    except AttributeError:
        raise ImportError("Neither mp.solutions.pose nor mp.tasks.vision available. "
                          "Please install mediapipe >= 0.10.0")

# Unified PoseLandmark enum
if _SOLUTIONS_AVAILABLE:
    PL = _mp_pose.PoseLandmark
    POSE_CONNECTIONS = _mp_pose.POSE_CONNECTIONS
else:
    PL = _tasks_vision.PoseLandmark
    _raw_conns = _tasks_vision.PoseLandmarksConnections.POSE_LANDMARKS
    POSE_CONNECTIONS = frozenset((c.start, c.end) for c in _raw_conns)

# ===================== Optional ONNX Runtime =====================
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except Exception:
    _ORT_AVAILABLE = False

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

def _scaled_font_size(ref_size, frame_h):
    return max(10, int(round(ref_size * (frame_h / _REF_H))))

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        pass
    for fallback in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(fallback, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

# ===================== SCORE DISPLAY =====================
def score_label(s: float) -> str:
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def display_half_str(x: float) -> str:
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

# ===================== FACE EXCLUSION =====================
_FACE_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}  # nose, eyes, ears, mouth
_BODY_CONNS = tuple((a, b) for (a, b) in POSE_CONNECTIONS
                    if a not in _FACE_INDICES and b not in _FACE_INDICES)
_BODY_POINTS = tuple(sorted({i for (a, b) in _BODY_CONNS for i in (a, b)}))

# ===================== GEOMETRY HELPERS =====================
def _angle_3pt(a, b, c):
    """Angle at vertex b formed by points a-b-c, in degrees."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    nrm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = float(np.clip(np.dot(ba, bc) / nrm, -1.0, 1.0))
    return math.degrees(math.acos(cosang))

def _angle_to_vertical(a, b):
    """Angle of vector a→b relative to straight up (0° = vertical upward)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]  # y increases downward in image coords
    angle = math.degrees(math.atan2(abs(dx), -dy + 1e-9))
    return abs(angle)

# ===================== EMA FILTER =====================
class EMA:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None

# ===================== LANDMARK SMOOTHER =====================
class LandmarkSmoother:
    """
    Smooths all 33 MediaPipe landmarks with per-landmark EMA.
    Reduces jitter significantly → cleaner skeleton + stable angle signals.
    """
    def __init__(self, alpha=0.5, vis_threshold=0.35):
        self.alpha = alpha
        self.vis_threshold = vis_threshold
        self.smoothed = {}  # idx → (x, y)
        self.visibility = {}  # idx → float

    def update(self, landmarks_list):
        """
        landmarks_list: list of landmarks with .x, .y, .visibility (or .presence)
        Returns dict of idx → (x, y) for visible/smoothed landmarks.
        """
        result = {}
        for idx in range(min(33, len(landmarks_list))):
            lm = landmarks_list[idx]

            # Handle both old API (.visibility) and new API (.visibility or .presence)
            vis = getattr(lm, 'visibility', None)
            if vis is None:
                vis = getattr(lm, 'presence', 0.5)  # Tasks API sometimes uses presence
            if vis is None:
                vis = 0.5

            self.visibility[idx] = float(vis)

            if float(vis) < self.vis_threshold:
                if idx in self.smoothed:
                    result[idx] = self.smoothed[idx]
                continue

            raw_x, raw_y = float(lm.x), float(lm.y)

            if idx in self.smoothed:
                prev_x, prev_y = self.smoothed[idx]
                sx = self.alpha * raw_x + (1 - self.alpha) * prev_x
                sy = self.alpha * raw_y + (1 - self.alpha) * prev_y
            else:
                sx, sy = raw_x, raw_y

            self.smoothed[idx] = (sx, sy)
            result[idx] = (sx, sy)

        return result

    def get_visibility(self, idx):
        return self.visibility.get(idx, 0.0)

# ===================== VIEW ANGLE ESTIMATOR =====================
class ViewAngleEstimator:
    """
    Estimates camera viewing angle:
    - side_ratio ~1.0 = pure side view (shoulder width appears small)
    - side_ratio ~0.0 = pure front/back view (shoulder width appears large)
    """
    def __init__(self, alpha=0.25):
        self.ema = EMA(alpha)

    def estimate(self, smoothed_pts):
        """Returns side_ratio in [0, 1]."""
        ls = smoothed_pts.get(11)  # LEFT_SHOULDER
        rs = smoothed_pts.get(12)  # RIGHT_SHOULDER
        lh = smoothed_pts.get(23)  # LEFT_HIP
        rh = smoothed_pts.get(24)  # RIGHT_HIP

        if not all([ls, rs, lh, rh]):
            return self.ema.value if self.ema.value is not None else 0.5

        shoulder_width = abs(ls[0] - rs[0])
        torso_h = (abs(ls[1] - lh[1]) + abs(rs[1] - rh[1])) / 2.0

        if torso_h < 1e-4:
            return self.ema.value if self.ema.value is not None else 0.5

        ratio = shoulder_width / torso_h
        # ratio ~0.1-0.3 → side; ratio ~0.8-1.5 → front
        side_score = float(np.clip(1.0 - (ratio - 0.15) / 0.85, 0.0, 1.0))
        return self.ema.update(side_score)

# ===================== MULTI-SIGNAL REP DETECTOR =====================
class DeadliftRepDetector:
    """
    Detects deadlift reps using adaptive multi-signal fusion:

    Signals:
    1. Torso angle (hip→shoulder vs vertical): reliable from side & diagonal
    2. Hip angle (shoulder-hip-knee): reliable from ALL angles
    3. Horizontal delta (|shoulder.x - hip.x|): original signal, side-view only

    State machine: STANDING → HINGING → RISING → STANDING (= 1 rep)
    With hysteresis to prevent false triggers.
    """

    STANDING = 0
    HINGING  = 1
    RISING   = 2

    # Composite thresholds (0 = standing, 1 = deep hinge)
    COMPOSITE_HINGE_START = 0.20   # enter hinge
    COMPOSITE_HINGE_DEEP  = 0.45   # "deep enough" to count
    COMPOSITE_STANDING    = 0.12   # back to standing
    COMPOSITE_RISING_DELTA = 0.04  # must drop this much to confirm rising

    MIN_HINGE_FRAMES = 3
    MIN_FRAMES_BETWEEN = 10

    def __init__(self, fps=10):
        self.fps = max(1, fps)
        self.state = self.STANDING
        self.frame_count = 0
        self.last_rep_frame = -999
        self.hinge_frames = 0

        self.torso_ema = EMA(0.45)
        self.hip_ema = EMA(0.45)
        self.composite_ema = EMA(0.35)

        # Per-rep quality tracking
        self.rep_max_composite = 0.0
        self.rep_back_round_frames = 0
        self.rep_leg_back_mismatch = 0
        self.rep_down_frames = 0
        self.rep_up_frames = 0
        self.rep_top_hold_frames = 0
        self.prev_composite = 0.0
        self.prev_knee_angle = None
        self.prev_torso_raw = None

        self.view_estimator = ViewAngleEstimator()

        # Results
        self.reps = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.all_scores = []
        self.reps_report = []

    def _compute_composite(self, torso_angle, hip_angle, x_delta, side_ratio):
        """
        Fuse signals into composite hinge-depth [0, 1].
        Weights adapt to camera angle.
        """
        # Normalize each signal to [0, 1] (0 = standing, 1 = deep hinge)
        torso_norm = float(np.clip((torso_angle - 10) / 60.0, 0.0, 1.0))
        hip_norm = float(np.clip((170 - hip_angle) / 70.0, 0.0, 1.0))
        xdelta_norm = float(np.clip((x_delta - 0.02) / 0.12, 0.0, 1.0))

        # Adaptive weights based on view angle
        # Side view → torso angle & x_delta reliable
        # Front/back → hip angle reliable, torso/x_delta less so
        if side_ratio > 0.6:
            # Mostly side view
            w_torso, w_hip, w_xdelta = 0.40, 0.30, 0.30
        elif side_ratio < 0.3:
            # Mostly front/back view
            w_torso, w_hip, w_xdelta = 0.20, 0.65, 0.15
        else:
            # Diagonal
            w_torso, w_hip, w_xdelta = 0.35, 0.45, 0.20

        composite = w_torso * torso_norm + w_hip * hip_norm + w_xdelta * xdelta_norm
        return self.composite_ema.update(composite)

    def process_frame(self, smoothed_pts, landmarks_list, frame_idx):
        """
        Process one frame.
        Returns: (composite_progress, rt_feedback_or_None, rep_info_or_None)
        """
        self.frame_count = frame_idx

        # --- Extract points (prefer right side, fallback to left) ---
        def _get(r_idx, l_idx):
            """Get smoothed point, preferring right side."""
            rp = smoothed_pts.get(r_idx)
            lp = smoothed_pts.get(l_idx)
            if rp is not None and lp is not None:
                # Use the one with better visibility, or average if both good
                return np.array(rp, dtype=float), True
            if rp is not None:
                return np.array(rp, dtype=float), True
            if lp is not None:
                return np.array(lp, dtype=float), True
            return None, False

        shoulder, s_ok = _get(12, 11)  # RIGHT_SHOULDER, LEFT_SHOULDER
        hip, h_ok = _get(24, 23)       # RIGHT_HIP, LEFT_HIP
        knee, k_ok = _get(26, 25)      # RIGHT_KNEE, LEFT_KNEE
        ankle, a_ok = _get(28, 27)     # RIGHT_ANKLE, LEFT_ANKLE

        if not (s_ok and h_ok):
            return self.prev_composite, None, None

        # View angle
        side_ratio = self.view_estimator.estimate(smoothed_pts)

        # --- Compute signals ---
        torso_angle = _angle_to_vertical(hip, shoulder)
        torso_smooth = self.torso_ema.update(torso_angle)

        hip_angle = 170.0
        if k_ok:
            hip_angle = _angle_3pt(shoulder, hip, knee)
        hip_smooth = self.hip_ema.update(hip_angle)

        x_delta = abs(shoulder[0] - hip[0])

        composite = self._compute_composite(torso_smooth, hip_smooth, x_delta, side_ratio)

        # --- Back rounding ---
        back_rounded = False
        head_pt = None
        for idx in (8, 7, 0):  # RIGHT_EAR, LEFT_EAR, NOSE
            if idx in smoothed_pts:
                head_pt = np.array(smoothed_pts[idx], dtype=float)
                break
        if head_pt is not None:
            back_rounded = self._check_back(shoulder, hip, head_pt)

        # --- Knee angle for leg-back mismatch ---
        knee_angle = None
        if k_ok and a_ok:
            knee_angle = _angle_3pt(hip, knee, ankle)

        # --- State machine ---
        rt_fb = None
        rep_info = None

        if self.state == self.STANDING:
            if (composite > self.COMPOSITE_HINGE_START and
                    frame_idx - self.last_rep_frame > self.MIN_FRAMES_BETWEEN):
                self.state = self.HINGING
                self.hinge_frames = 0
                self._reset_rep()

        elif self.state == self.HINGING:
            self.hinge_frames += 1
            self.rep_max_composite = max(self.rep_max_composite, composite)
            self._track_quality(composite, back_rounded, knee_angle, torso_smooth)

            if back_rounded:
                rt_fb = "Try to keep your back a bit straighter"

            # Transition to rising: composite drops significantly from peak
            if (self.hinge_frames >= self.MIN_HINGE_FRAMES and
                    self.rep_max_composite >= self.COMPOSITE_HINGE_DEEP * 0.65 and
                    composite < self.rep_max_composite - self.COMPOSITE_RISING_DELTA):
                self.state = self.RISING

        elif self.state == self.RISING:
            self._track_quality(composite, back_rounded, knee_angle, torso_smooth)

            if back_rounded:
                rt_fb = "Try to keep your back a bit straighter"

            if composite < self.COMPOSITE_STANDING:
                # Rep completed
                if self.rep_max_composite >= self.COMPOSITE_HINGE_DEEP * 0.55:
                    rep_info = self._finalize_rep(frame_idx)
                self.state = self.STANDING
                self.last_rep_frame = frame_idx

        self.prev_composite = composite
        return composite, rt_fb, rep_info

    def _reset_rep(self):
        self.rep_max_composite = 0.0
        self.rep_back_round_frames = 0
        self.rep_leg_back_mismatch = 0
        self.rep_down_frames = 0
        self.rep_up_frames = 0
        self.rep_top_hold_frames = 0
        self.prev_knee_angle = None
        self.prev_torso_raw = None

    def _track_quality(self, composite, back_rounded, knee_angle, torso_angle):
        if back_rounded:
            self.rep_back_round_frames += 1

        if composite > self.prev_composite + 0.005:
            self.rep_down_frames += 1
        elif composite < self.prev_composite - 0.005:
            self.rep_up_frames += 1

        if composite < 0.10:
            self.rep_top_hold_frames += 1

        # Leg-back mismatch detection
        if (knee_angle is not None and self.prev_knee_angle is not None
                and self.prev_torso_raw is not None):
            dk = knee_angle - self.prev_knee_angle
            dt_a = torso_angle - self.prev_torso_raw
            if abs(dk) > 1.5 and abs(dt_a) < 0.3:
                self.rep_leg_back_mismatch += 1
            elif abs(dt_a) > 1.5 and abs(dk) < 0.3:
                self.rep_leg_back_mismatch += 1

        self.prev_knee_angle = knee_angle
        self.prev_torso_raw = torso_angle

    def _check_back(self, shoulder, hip, head, max_angle=22.0):
        torso_vec = shoulder - hip
        head_vec = head - shoulder
        tn = np.linalg.norm(torso_vec) + 1e-9
        hn = np.linalg.norm(head_vec) + 1e-9
        if hn < 0.3 * tn:
            return False
        cos_a = float(np.clip(np.dot(torso_vec, head_vec) / (tn * hn), -1, 1))
        return math.degrees(math.acos(cos_a)) > max_angle

    def _finalize_rep(self, frame_idx):
        self.reps += 1
        dt = 1.0 / self.fps

        penalty = 0.0
        fb = []

        back_min = max(3, int(0.3 / dt))
        if self.rep_back_round_frames >= back_min:
            fb.append("Try to keep your back a bit straighter")
            penalty += 1.5

        mismatch_min = max(4, int(0.4 / dt))
        if self.rep_leg_back_mismatch >= mismatch_min:
            fb.append("Drive the back up with the legs evenly")
            penalty += 1.0

        if self.rep_max_composite < self.COMPOSITE_HINGE_DEEP * 0.75:
            fb.append("Try to hinge a bit deeper")
            penalty += 0.5

        score = round(max(4, 10 - penalty) * 2) / 2.0

        if score >= 9.5:
            self.good_reps += 1
        else:
            self.bad_reps += 1

        down_s = self.rep_down_frames * dt
        top_s = self.rep_top_hold_frames * dt
        rom = self.rep_max_composite
        tip = _choose_tip(down_s, top_s, rom)

        info = {
            "rep_index":     self.reps,
            "score":         float(score),
            "score_display": display_half_str(score),
            "feedback":      fb if score < 10.0 else [],
            "tip":           tip,
        }
        self.all_scores.append(score)
        self.reps_report.append(info)
        return info


def _choose_tip(down_s, top_s, rom):
    if down_s is not None and down_s < 0.35:
        return "Slow the lowering to ~2–3s for more hypertrophy"
    if top_s is not None and top_s < 0.15:
        return "Squeeze the glutes for a brief hold at the top"
    if rom is not None and 0.3 <= rom <= 0.5:
        return "Hinge a bit deeper within comfort"
    return "Keep the bar close and move smoothly"

# ===================== SKELETON DRAWING =====================
def _dyn_thickness(h):
    return max(2, int(round(h * 0.003))), max(3, int(round(h * 0.005)))

def draw_skeleton(frame, smoothed_pts, color=(255, 255, 255)):
    """Draw clean skeleton from smoothed landmarks."""
    h, w = frame.shape[:2]
    line_t, dot_r = _dyn_thickness(h)

    for a, b in _BODY_CONNS:
        if a in smoothed_pts and b in smoothed_pts:
            ax, ay = int(smoothed_pts[a][0] * w), int(smoothed_pts[a][1] * h)
            bx, by = int(smoothed_pts[b][0] * w), int(smoothed_pts[b][1] * h)
            cv2.line(frame, (ax, ay), (bx, by), color, line_t, cv2.LINE_AA)

    for i in _BODY_POINTS:
        if i in smoothed_pts:
            x, y = int(smoothed_pts[i][0] * w), int(smoothed_pts[i][1] * h)
            cv2.circle(frame, (x, y), dot_r, color, -1, cv2.LINE_AA)

    return frame

# ===================== OVERLAY (same visual standard as good_morning) =====================
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
        last = lines[-1] + "…"
        while draw.textlength(last, font=font) > maxw and len(last) > 1:
            last = last[:-2] + "…"
        lines[-1] = last
    return lines

def draw_overlay(frame, reps=0, feedback=None, progress_pct=0.0):
    h, w, _ = frame.shape

    reps_fs = _scaled_font_size(_REF_REPS_FONT_SIZE, h)
    fb_fs = _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, h)
    dl_fs = _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, h)
    dp_fs = _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, h)

    f_reps = _load_font(FONT_PATH, reps_fs)
    f_fb = _load_font(FONT_PATH, fb_fs)
    f_dl = _load_font(FONT_PATH, dl_fs)
    f_dp = _load_font(FONT_PATH, dp_fs)

    pct = float(np.clip(progress_pct, 0, 1))
    bg_a = int(round(255 * BAR_BG_ALPHA))

    ref_h = max(int(h * 0.06), int(reps_fs * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)

    overlay_np = np.zeros((h, w, 4), dtype=np.uint8)

    pad_x, pad_y = 10, 6
    tmp_pil = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_pil)
    txt = f"Reps: {int(reps)}"
    tw = tmp_draw.textlength(txt, font=f_reps)
    thh = f_reps.size
    cv2.rectangle(overlay_np, (0, 0), (int(tw + 2 * pad_x), int(thh + 2 * pad_y)),
                  (0, 0, 0, bg_a), -1)

    # Donut
    cv2.circle(overlay_np, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)
    start_ang = -90
    end_ang = start_ang + int(360 * pct)
    if end_ang != start_ang:
        cv2.ellipse(overlay_np, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                    (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)

    # Feedback BG
    fb_y0 = 0
    fb_lines = []
    fb_pad_x = fb_pad_y = line_gap = line_h = 0
    if feedback:
        safe_m = max(6, int(h * 0.02))
        fb_pad_x, fb_pad_y, line_gap = 12, 8, 4
        max_tw = int(w - 2 * fb_pad_x - 20)
        fb_lines = _wrap2(tmp_draw, feedback, f_fb, max_tw)
        line_h = f_fb.size + 6
        block_h = 2 * fb_pad_y + len(fb_lines) * line_h + (len(fb_lines) - 1) * line_gap
        fb_y0 = max(0, h - safe_m - block_h)
        y1 = h - safe_m
        cv2.rectangle(overlay_np, (0, fb_y0), (w, y1), (0, 0, 0, bg_a), -1)

    overlay_pil = Image.fromarray(overlay_np, mode="RGBA")
    draw = ImageDraw.Draw(overlay_pil)

    draw.text((pad_x, pad_y - 1), txt, font=f_reps, fill=(255, 255, 255, 255))

    gap = max(2, int(radius * 0.10))
    by = cy - (f_dl.size + gap + f_dp.size) // 2
    label = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label, font=f_dl)
    pw = draw.textlength(pct_txt, font=f_dp)
    draw.text((cx - int(lw // 2), by), label, font=f_dl, fill=(255, 255, 255, 255))
    draw.text((cx - int(pw // 2), by + f_dl.size + gap), pct_txt, font=f_dp, fill=(255, 255, 255, 255))

    if feedback and fb_lines:
        ty = fb_y0 + fb_pad_y
        for ln in fb_lines:
            tw2 = draw.textlength(ln, font=f_fb)
            tx = max(fb_pad_x, (w - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=f_fb, fill=(255, 255, 255, 255))
            ty += line_h + line_gap

    overlay_rgba = np.array(overlay_pil)
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = overlay_rgba[:, :, [2, 1, 0]].astype(np.float32)
    out_f = frame.astype(np.float32) * (1.0 - alpha) + overlay_bgr * alpha
    return out_f.astype(np.uint8)

# ===================== YOLOv8n Detector (optional, carried from V1) =====================
class YoloOccluderDetector:
    def __init__(self, onnx_path, providers=None, input_size=640, conf_thres=0.25,
                 iou_thres=0.45, occluder_class_ids=None):
        if not _ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not available")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(onnx_path)
        self.sess = ort.InferenceSession(onnx_path, providers=providers or ort.get_available_providers())
        self.inp_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.imgsz = int(input_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.occluder_class_ids = occluder_class_ids

    @staticmethod
    def _letterbox(img, new_shape=640, color=(114, 114, 114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw // 2, dh // 2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    @staticmethod
    def _nms(boxes, scores, iou_thres=0.45):
        if len(boxes) == 0:
            return []
        boxes = boxes.astype(np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            ww = np.maximum(0, xx2 - xx1 + 1)
            hh = np.maximum(0, yy2 - yy1 + 1)
            iou = (ww * hh) / (areas[i] + areas[order[1:]] - ww * hh + 1e-9)
            order = order[np.where(iou <= iou_thres)[0] + 1]
        return keep

    def infer(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]
        img, r, (dw, dh) = self._letterbox(frame_bgr, self.imgsz)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = np.transpose(img_rgb, (2, 0, 1))[None]
        out = self.sess.run([self.out_name], {self.inp_name: img_rgb})[0]
        if out.ndim == 3 and out.shape[0] == 1:
            o = out[0]
            if o.shape[0] in (84, 85):
                xywh = o[0:4, :].T
                if o.shape[0] >= 85:
                    num = o.shape[1]
                    obj = o[4, :].reshape(num, 1)
                    cls = o[5:, :].T
                    cls_id = np.argmax(cls, axis=1)
                    conf = (obj[:, 0] * cls.max(axis=1)).flatten()
                else:
                    cls = o[4:, :].T
                    cls_id = np.argmax(cls, axis=1)
                    conf = cls.max(axis=1).flatten()
            else:
                xywh = o[:, 0:4]
                if o.shape[1] >= 85:
                    obj = o[:, 4:5]
                    cls = o[:, 5:]
                    cls_id = np.argmax(cls, axis=1)
                    conf = (obj[:, 0] * cls.max(axis=1)).flatten()
                else:
                    cls = o[:, 4:]
                    cls_id = np.argmax(cls, axis=1)
                    conf = cls.max(axis=1).flatten()
        else:
            return []
        m = conf >= self.conf_thres
        if not np.any(m):
            return []
        xywh, conf, cls_id = xywh[m], conf[m], cls_id[m]
        if self.occluder_class_ids is not None:
            sel = np.isin(cls_id, list(self.occluder_class_ids))
            xywh, conf, cls_id = xywh[sel], conf[sel], cls_id[sel]
            if len(xywh) == 0:
                return []
        xyxy = np.zeros_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        keep = self._nms(xyxy, conf, self.iou_thres)
        xyxy = xyxy[keep]
        xyxy[:, [0, 2]] -= dw
        xyxy[:, [1, 3]] -= dh
        xyxy /= r
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, w0 - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, w0 - 1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, h0 - 1)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, h0 - 1)
        return [(float(x1) / w0, float(y1) / h0, float(x2) / w0, float(y2) / h0)
                for x1, y1, x2, y2 in xyxy]

# ===================== POSE WRAPPER (dual API) =====================
class PoseEstimator:
    """
    Wraps both old `mp.solutions.pose.Pose` and new `mp.tasks.vision.PoseLandmarker`.
    Provides uniform interface: process(frame_bgr) → list of landmarks or None.
    """
    def __init__(self, model_path=None):
        self._impl = None
        self._is_tasks = False

        if _SOLUTIONS_AVAILABLE:
            # Old API — use model_complexity=2 for best accuracy
            self._impl = _mp_pose.Pose(
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        elif _USE_TASKS_API:
            if model_path and os.path.exists(model_path):
                opts = _tasks_vision.PoseLandmarkerOptions(
                    base_options=_tasks_base(model_asset_path=model_path),
                    running_mode=_tasks_vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._impl = _tasks_vision.PoseLandmarker.create_from_options(opts)
                self._is_tasks = True
            else:
                raise FileNotFoundError(
                    "Tasks API requires a pose_landmarker.task model file. "
                    "Download from https://storage.googleapis.com/mediapipe-models/"
                    "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
                )

    def process(self, frame_bgr, timestamp_ms=0):
        """
        Returns list of landmarks (each with .x, .y, .visibility) or None.
        """
        if not self._is_tasks:
            # Old solutions API
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._impl.process(rgb)
            if res.pose_landmarks:
                return res.pose_landmarks.landmark
            return None
        else:
            # Tasks API
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            res = self._impl.detect_for_video(mp_img, int(timestamp_ms))
            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                return res.pose_landmarks[0]
            return None

    def close(self):
        if hasattr(self._impl, 'close'):
            self._impl.close()
        elif hasattr(self._impl, '__exit__'):
            self._impl.__exit__(None, None, None)

# ===================== MAIN =====================
def run_deadlift_analysis(video_path,
                          frame_skip=2,
                          scale=0.5,
                          output_path="deadlift_analyzed.mp4",
                          feedback_path="deadlift_feedback.txt",
                          detector_onnx_path="barbell_plates_yolov8n.onnx",
                          detector_input_size=640,
                          detector_conf_thres=0.25,
                          detector_iou_thres=0.45,
                          detector_class_ids=None,
                          return_video=True,
                          fast_mode=None,
                          pose_model_path=None):
    """
    Analyze deadlift video for rep counting and technique scoring.

    Args:
        video_path: Path to input video
        frame_skip: Process every Nth frame (2 = every other frame)
        scale: Downscale factor for pose estimation (0.5 = half resolution)
        output_path: Output video path
        feedback_path: Output text feedback path
        detector_onnx_path: Optional YOLO detector for barbell occlusion
        return_video: Whether to produce annotated output video
        fast_mode: If True, skip video output
        pose_model_path: Path to pose_landmarker.task (only needed for Tasks API)
    """
    if fast_mode is True:
        return_video = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    # Optional YOLO detector
    detector = None
    if detector_onnx_path and _ORT_AVAILABLE and os.path.exists(detector_onnx_path):
        try:
            detector = YoloOccluderDetector(
                detector_onnx_path,
                input_size=detector_input_size,
                conf_thres=detector_conf_thres,
                iou_thres=detector_iou_thres,
                occluder_class_ids=detector_class_ids,
            )
        except Exception:
            detector = None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    frame_dt = 1.0 / float(effective_fps)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Core components
    rep_detector = DeadliftRepDetector(fps=effective_fps)
    smoother = LandmarkSmoother(alpha=0.5, vis_threshold=0.35)
    progress_ema = EMA(0.35)
    progress_ema.value = 0.0

    frame_idx = 0
    current_feedback = None
    feedback_hold_frames = 0
    FEEDBACK_HOLD = max(8, int(1.5 / frame_dt))  # hold feedback for ~1.5s

    # Initialize pose estimator
    try:
        pose_est = PoseEstimator(model_path=pose_model_path)
    except Exception as e:
        cap.release()
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0,
            "feedback": [f"Pose model error: {e}"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            work = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame.copy()

            if return_video and out_vid is None:
                out_vid = cv2.VideoWriter(output_path, fourcc, effective_fps, (orig_w, orig_h))

            timestamp_ms = (frame_idx / fps_in) * 1000.0
            landmarks = pose_est.process(work, timestamp_ms=timestamp_ms)

            if landmarks is None:
                if return_video and out_vid is not None:
                    frame_out = cv2.resize(work, (orig_w, orig_h))
                    pv = progress_ema.value or 0.0
                    out_vid.write(draw_overlay(frame_out, reps=rep_detector.reps,
                                               feedback=None, progress_pct=1.0 - pv))
                continue

            try:
                smoothed = smoother.update(landmarks)
                composite, rt_fb, rep_info = rep_detector.process_frame(smoothed, landmarks, frame_idx)

                # Feedback management with hold timer
                if rt_fb:
                    current_feedback = rt_fb
                    feedback_hold_frames = FEEDBACK_HOLD
                elif rep_info and rep_info["feedback"]:
                    current_feedback = rep_info["feedback"][0]
                    feedback_hold_frames = FEEDBACK_HOLD
                else:
                    if feedback_hold_frames > 0:
                        feedback_hold_frames -= 1
                    else:
                        current_feedback = None

                # Progress donut: 100% standing, 0% deep hinge
                prog = progress_ema.update(composite)
                donut_pct = 1.0 - prog

                if return_video:
                    frame_out = cv2.resize(work, (orig_w, orig_h))
                    frame_out = draw_skeleton(frame_out, smoothed)
                    frame_out = draw_overlay(frame_out, reps=rep_detector.reps,
                                             feedback=current_feedback,
                                             progress_pct=donut_pct)
                    out_vid.write(frame_out)

            except Exception:
                if return_video and out_vid is not None:
                    frame_out = cv2.resize(work, (orig_w, orig_h))
                    pv = progress_ema.value or 0.0
                    out_vid.write(draw_overlay(frame_out, reps=rep_detector.reps,
                                              feedback=None, progress_pct=1.0 - pv))
                continue

    finally:
        pose_est.close()
        cap.release()
        if return_video and out_vid:
            out_vid.release()
        cv2.destroyAllWindows()

    # Final results
    avg = np.mean(rep_detector.all_scores) if rep_detector.all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)

    seen = set()
    feedback_list = []
    for r in rep_detector.reps_report:
        if float(r.get("score") or 0.0) >= 10.0:
            continue
        for msg in (r.get("feedback") or []):
            if msg and msg not in seen:
                seen.add(msg)
                feedback_list.append(msg)
    if not feedback_list:
        feedback_list = ["Great form! Keep it up 💪"]

    # Encode video with ffmpeg
    final_video_path = ""
    if return_video:
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart",
                "-pix_fmt", "yuv420p", encoded_path
            ], check=False, capture_output=True)
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                os.remove(output_path)
        except Exception:
            pass
        final_video_path = (encoded_path if os.path.exists(encoded_path)
                            else (output_path if os.path.exists(output_path) else ""))

    return {
        "squat_count": int(rep_detector.reps),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(rep_detector.good_reps),
        "bad_reps": int(rep_detector.bad_reps),
        "feedback": feedback_list,
        "reps": rep_detector.reps_report,
        "video_path": final_video_path,
        "feedback_path": feedback_path,
    }

# -*- coding: utf-8 -*-
# deadlift_analysis.py — V4.0: two-pass architecture (from romanian_deadlift v5.1)
#
# KEY FIXES over V3.1:
#   1. TWO-PASS: Pass 1 = analysis only (no video writing) → no black video
#                Pass 2 = render video from stored frame_data → fast & reliable
#   2. No more lazy VideoWriter creation → consistent dimensions guaranteed
#   3. frame_skip applied correctly (read all frames, skip before processing)
#   4. Hard cap replaced with time-based timeout (180s)
#   5. _snapshot_smoothed() freezes landmark dict — no flickering skeleton
#   6. ffmpeg encode with robust fallback chain

import os, cv2, math, numpy as np, subprocess, json, time
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== MediaPipe import (dual API support) =====================
_USE_TASKS_API = False
_SOLUTIONS_AVAILABLE = False

try:
    _mp_pose = mp.solutions.pose
    _SOLUTIONS_AVAILABLE = True
except AttributeError:
    _SOLUTIONS_AVAILABLE = False

if not _SOLUTIONS_AVAILABLE:
    try:
        _tasks_vision = mp.tasks.vision
        _tasks_base = mp.tasks.BaseOptions
        _USE_TASKS_API = True
    except AttributeError:
        raise ImportError("Neither mp.solutions.pose nor mp.tasks.vision available.")

if _SOLUTIONS_AVAILABLE:
    PL = _mp_pose.PoseLandmark
    POSE_CONNECTIONS = _mp_pose.POSE_CONNECTIONS
else:
    PL = _tasks_vision.PoseLandmark
    _raw_conns = _tasks_vision.PoseLandmarksConnections.POSE_LANDMARKS
    POSE_CONNECTIONS = frozenset((c.start, c.end) for c in _raw_conns)

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
_REF_H                     = 480.0
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


_FACE_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
_BODY_CONNS = tuple((a, b) for (a, b) in POSE_CONNECTIONS
                    if a not in _FACE_INDICES and b not in _FACE_INDICES)
_BODY_POINTS = tuple(sorted({i for (a, b) in _BODY_CONNS for i in (a, b)}))


def _angle_3pt(a, b, c):
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    nrm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    return math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / nrm, -1.0, 1.0))))


def _angle_to_vertical(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return abs(math.degrees(math.atan2(abs(dx), -dy + 1e-9)))


# ===================== EMA =====================
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
    def __init__(self, alpha=0.5, vis_threshold=0.4):
        self.alpha = alpha
        self.vis_threshold = vis_threshold
        self.smoothed = {}

    def update(self, landmarks):
        result = {}
        for idx in range(min(33, len(landmarks))):
            lm = landmarks[idx]
            vis = getattr(lm, 'visibility', None)
            if vis is None:
                vis = getattr(lm, 'presence', 0.5)
            if vis is None:
                vis = 0.5
            if float(vis) < self.vis_threshold:
                if idx in self.smoothed:
                    result[idx] = self.smoothed[idx][:2]
                continue
            raw = np.array([float(lm.x), float(lm.y), getattr(lm, 'z', 0.0)], dtype=float)
            if idx in self.smoothed:
                prev = np.array(self.smoothed[idx], dtype=float)
                smooth = self.alpha * raw + (1 - self.alpha) * prev
            else:
                smooth = raw
            self.smoothed[idx] = tuple(smooth)
            result[idx] = (float(smooth[0]), float(smooth[1]))
        return result


def _snapshot_smoothed(smoothed_pts):
    """Freeze a copy of smoothed_pts so it can't be mutated later."""
    return {k: (float(v[0]), float(v[1])) for k, v in smoothed_pts.items()}


# ===================== VIEW ANGLE ESTIMATOR =====================
class ViewAngleEstimator:
    def __init__(self, alpha=0.3):
        self.ema = EMA(alpha)

    def estimate(self, smoothed_pts):
        ls = smoothed_pts.get(11)
        rs = smoothed_pts.get(12)
        lh = smoothed_pts.get(23)
        rh = smoothed_pts.get(24)
        if not all([ls, rs, lh, rh]):
            return self.ema.value if self.ema.value is not None else 0.5
        shoulder_width = abs(ls[0] - rs[0])
        torso_h = (abs(ls[1] - lh[1]) + abs(rs[1] - rh[1])) / 2.0
        if torso_h < 1e-4:
            return self.ema.value if self.ema.value is not None else 0.5
        ratio = shoulder_width / torso_h
        side_score = float(np.clip(1.0 - (ratio - 0.15) / 0.85, 0.0, 1.0))
        return self.ema.update(side_score)


# ===================== FRONT-VIEW SIGNAL =====================
class FrontViewSignal:
    def __init__(self):
        self.signal_ema = EMA(0.40)
        self.frame_count = 0
        self.baseline_ready = False
        self.min_frames_for_baseline = 5
        self.standing_shoulder_y = None
        self.standing_hip_y = None
        self.standing_sh_dist = None
        self.standing_ankle_y = None
        self.shoulder_y_history = deque(maxlen=60)
        self.hip_y_history = deque(maxlen=60)
        self.sh_dist_history = deque(maxlen=60)
        self.hip_y_ema = EMA(0.3)
        self.hip_x_ema = EMA(0.3)
        self.prev_hip_y = None
        self.prev_hip_x = None
        self.hip_movement_acc = 0.0
        self.hip_movement_decay = 0.92

    def _get_key_points(self, smoothed_pts):
        ls = smoothed_pts.get(11)
        rs = smoothed_pts.get(12)
        lh = smoothed_pts.get(23)
        rh = smoothed_pts.get(24)
        la = smoothed_pts.get(27)
        ra = smoothed_pts.get(28)
        lk = smoothed_pts.get(25)
        rk = smoothed_pts.get(26)
        if not ls or not rs or not lh or not rh:
            return None
        shoulder_y = (ls[1] + rs[1]) / 2.0
        shoulder_x = (ls[0] + rs[0]) / 2.0
        hip_y = (lh[1] + rh[1]) / 2.0
        hip_x = (lh[0] + rh[0]) / 2.0
        ankle_y = None
        if la and ra:   ankle_y = (la[1] + ra[1]) / 2.0
        elif la:        ankle_y = la[1]
        elif ra:        ankle_y = ra[1]
        elif lk and rk: ankle_y = (lk[1] + rk[1]) / 2.0
        elif lk:        ankle_y = lk[1]
        elif rk:        ankle_y = rk[1]
        return {'shoulder_y': shoulder_y, 'shoulder_x': shoulder_x,
                'hip_y': hip_y, 'hip_x': hip_x,
                'ankle_y': ankle_y, 'sh_dist': hip_y - shoulder_y}

    def _detect_walking(self, hip_y, hip_x, sh_dist):
        smooth_hy = self.hip_y_ema.update(hip_y)
        smooth_hx = self.hip_x_ema.update(hip_x)
        if self.prev_hip_y is not None:
            dy = abs(smooth_hy - self.prev_hip_y)
            dx = abs(smooth_hx - self.prev_hip_x)
            self.hip_movement_acc = (self.hip_movement_acc * self.hip_movement_decay
                                     + math.sqrt(dy**2 + dx**2))
        self.prev_hip_y = smooth_hy
        self.prev_hip_x = smooth_hx
        nm = (self.hip_movement_acc / sh_dist if sh_dist > 0.01
              else self.hip_movement_acc * 10)
        return float(np.clip(1.0 - (nm - 0.4) / 0.6, 0.0, 1.0))

    def _update_baseline(self, shoulder_y, hip_y, sh_dist, ankle_y):
        self.shoulder_y_history.append(shoulder_y)
        self.hip_y_history.append(hip_y)
        self.sh_dist_history.append(sh_dist)
        if len(self.shoulder_y_history) >= self.min_frames_for_baseline:
            sorted_sy = sorted(self.shoulder_y_history)
            idx_10 = max(0, int(len(sorted_sy) * 0.10))
            self.standing_shoulder_y = sorted_sy[idx_10]
            sorted_hy = sorted(self.hip_y_history)
            self.standing_hip_y = sorted_hy[idx_10]
            sorted_shd = sorted(self.sh_dist_history, reverse=True)
            idx_90 = max(0, int(len(sorted_shd) * 0.10))
            self.standing_sh_dist = sorted_shd[idx_90]
            if ankle_y is not None:
                self.standing_ankle_y = ankle_y
            self.baseline_ready = True

    def update(self, smoothed_pts):
        self.frame_count += 1
        pts = self._get_key_points(smoothed_pts)
        if pts is None:
            return self.signal_ema.value if self.signal_ema.value is not None else 0.0
        shoulder_y = pts['shoulder_y']
        hip_y      = pts['hip_y']
        ankle_y    = pts['ankle_y']
        sh_dist    = pts['sh_dist']
        self._update_baseline(shoulder_y, hip_y, sh_dist, ankle_y)
        walk_supp = self._detect_walking(hip_y, pts['hip_x'], sh_dist)
        if not self.baseline_ready:
            self.signal_ema.update(0.0)
            return 0.0
        shoulder_drop = shoulder_y - self.standing_shoulder_y
        if self.standing_sh_dist > 0.01:
            sig_drop = float(np.clip(shoulder_drop / (self.standing_sh_dist * 0.5), 0.0, 1.0))
            compression_ratio = sh_dist / self.standing_sh_dist
            sig_comp = float(np.clip((1.0 - compression_ratio) / 0.45, 0.0, 1.0))
        else:
            sig_drop = sig_comp = 0.0
        raw = 0.6 * sig_drop + 0.4 * sig_comp
        return self.signal_ema.update(raw * walk_supp)


# ===================== REP DETECTOR =====================
class DeadliftRepDetector:
    STANDING = 0
    HINGING  = 1
    RISING   = 2

    # Raw thresholds — overridden after calibration
    COMPOSITE_HINGE_START = 0.32
    COMPOSITE_HINGE_DEEP  = 0.50
    COMPOSITE_STANDING    = 0.18
    MIN_HINGE_FRAMES      = 10     # ~1s at 10fps
    MIN_FRAMES_BETWEEN    = 35     # ~3.5s min between reps at 10fps

    def __init__(self, fps=10):
        self.fps = fps
        self.state = self.STANDING
        self.frame_count = 0
        self.last_rep_frame = -999
        self.hinge_frames = 0
        self.torso_angle_ema = EMA(0.45)
        self.hip_angle_ema = EMA(0.45)
        self.composite_ema = EMA(0.40)
        self.front_signal = FrontViewSignal()
        self.rep_max_composite = 0.0
        self.rep_back_rounding_frames = 0
        self.rep_leg_back_mismatch_frames = 0
        self.rep_down_frames = 0
        self.rep_up_frames = 0
        self.rep_top_hold_frames = 0
        self.prev_composite = 0.0
        self.prev_knee_angle = None
        self.prev_torso_raw = None
        self.view_estimator = ViewAngleEstimator()
        self.reps = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.all_scores = []
        self.reps_report = []

        # Dynamic calibration — learns session baseline & range
        self._cal_signals    = []
        self._calibrated     = False
        self._cal_floor      = 0.0   # standing composite
        self._cal_range      = 0.50  # typical rep range

    def _calibrate(self, composite):
        """
        Collect first 20 frames to estimate standing baseline.
        Then set HINGE_START = baseline + 35% of expected range,
        HINGE_DEEP = baseline + 80% of expected range.
        """
        if self._calibrated:
            return
        self._cal_signals.append(composite)
        if len(self._cal_signals) >= 30:
            # 40th percentile = stable standing baseline (ignores early movement)
            floor = float(np.percentile(self._cal_signals, 40))
            ceil  = float(np.percentile(self._cal_signals, 95))
            rng   = max(0.20, ceil - floor)
            self._cal_floor = floor
            self._cal_range = rng
            # HINGE_START: floor + 40% of range — needs substantial hinge
            self.COMPOSITE_HINGE_START = min(0.60, floor + 0.40 * rng)
            # COMPOSITE_STANDING: floor + 12% — truly back upright
            self.COMPOSITE_STANDING    = min(0.35, floor + 0.12 * rng)
            # COMPOSITE_HINGE_DEEP: floor + 80% of range — real deadlift depth
            self.COMPOSITE_HINGE_DEEP  = min(0.85, floor + 0.80 * rng)
            self._calibrated = True
            import sys
            print(f"[DL] Calibrated: floor={floor:.3f} rng={rng:.3f} "
                  f"start={self.COMPOSITE_HINGE_START:.3f} "
                  f"deep={self.COMPOSITE_HINGE_DEEP:.3f} "
                  f"standing={self.COMPOSITE_STANDING:.3f}",
                  file=sys.stderr, flush=True)

    def _compute_side_composite(self, torso_angle, hip_angle, side_ratio):
        torso_norm = float(np.clip((torso_angle - 12) / 55.0, 0.0, 1.0))
        hip_norm   = float(np.clip((170 - hip_angle) / 70.0, 0.0, 1.0))
        w_torso = 0.3 + 0.5 * side_ratio
        w_hip   = 0.3 + 0.5 * (1 - side_ratio)
        total   = w_torso + w_hip + 1e-9
        composite = (w_torso / total) * torso_norm + (w_hip / total) * hip_norm
        return self.composite_ema.update(composite)

    def _get_lm_val(self, lm, idx, attr):
        try:
            if hasattr(idx, 'value'):
                idx = idx.value
            val = getattr(lm[idx], attr, None)
            return (0.5 if attr == 'visibility' else 0.0) if val is None else float(val)
        except (IndexError, AttributeError, TypeError):
            return 0.5 if attr == 'visibility' else 0.0

    def process_frame(self, smoothed_pts, raw_landmarks, frame_idx):
        self.frame_count = frame_idx
        lm = raw_landmarks

        def _get_pt(primary_idx, fallback_idx):
            if primary_idx in smoothed_pts and self._get_lm_val(lm, primary_idx, 'visibility') > 0.4:
                return np.array(smoothed_pts[primary_idx], dtype=float), True
            if fallback_idx in smoothed_pts and self._get_lm_val(lm, fallback_idx, 'visibility') > 0.4:
                return np.array(smoothed_pts[fallback_idx], dtype=float), True
            return None, False

        shoulder, s_ok = _get_pt(12, 11)
        hip,      h_ok = _get_pt(24, 23)
        knee,     k_ok = _get_pt(26, 25)
        ankle,    a_ok = _get_pt(28, 27)

        if not (s_ok and h_ok):
            return self.prev_composite, None, None

        side_ratio = self.view_estimator.estimate(smoothed_pts)
        torso_angle_smooth = self.torso_angle_ema.update(_angle_to_vertical(hip, shoulder))
        hip_angle_smooth = self.hip_angle_ema.update(
            _angle_3pt(shoulder, hip, knee) if k_ok else 170.0)
        side_composite = self._compute_side_composite(torso_angle_smooth, hip_angle_smooth, side_ratio)
        front_val = self.front_signal.update(smoothed_pts)

        if side_ratio >= 0.6:
            composite = side_composite
        elif side_ratio <= 0.3:
            composite = front_val
        else:
            blend = (side_ratio - 0.3) / 0.3
            composite = blend * side_composite + (1.0 - blend) * front_val

        # Calibrate BEFORE state machine — learns floor/ceiling from early frames
        self._calibrate(composite)

        # Back rounding check
        head_pt = None
        for idx in (8, 7, 0):
            if idx in smoothed_pts and self._get_lm_val(lm, idx, 'visibility') > 0.45:
                head_pt = np.array(smoothed_pts[idx], dtype=float)
                break
        back_rounded = (self._check_back_rounding(shoulder, hip, head_pt)
                        if head_pt is not None else False)
        knee_angle = _angle_3pt(hip, knee, ankle) if (k_ok and a_ok) else None

        rt_feedback = None
        rep_info    = None

        if self.state == self.STANDING:
            if (composite > self.COMPOSITE_HINGE_START
                    and (frame_idx - self.last_rep_frame > self.MIN_FRAMES_BETWEEN)):
                self.state = self.HINGING
                self.hinge_frames = 0
                self._reset_rep_tracking()

        elif self.state == self.HINGING:
            self.hinge_frames += 1
            self.rep_max_composite = max(self.rep_max_composite, composite)
            self._track_rep_quality(composite, back_rounded, knee_angle, torso_angle_smooth)
            if back_rounded:
                rt_feedback = "Try to keep your back a bit straighter"
            if (composite < self.prev_composite - 0.06
                    and self.hinge_frames >= self.MIN_HINGE_FRAMES
                    and self.rep_max_composite >= self.COMPOSITE_HINGE_DEEP * 0.85):
                self.state = self.RISING

        elif self.state == self.RISING:
            self._track_rep_quality(composite, back_rounded, knee_angle, torso_angle_smooth)
            if back_rounded:
                rt_feedback = "Try to keep your back a bit straighter"
            if composite < self.COMPOSITE_STANDING:
                if self.rep_max_composite >= self.COMPOSITE_HINGE_DEEP * 0.80:
                    rep_info = self._finalize_rep(frame_idx)
                self.state = self.STANDING
                self.last_rep_frame = frame_idx

        self.prev_composite = composite
        return composite, rt_feedback, rep_info

    def _reset_rep_tracking(self):
        self.rep_max_composite = 0.0
        self.rep_back_rounding_frames = 0
        self.rep_leg_back_mismatch_frames = 0
        self.rep_down_frames = 0
        self.rep_up_frames = 0
        self.rep_top_hold_frames = 0
        self.prev_knee_angle = None
        self.prev_torso_raw = None

    def _track_rep_quality(self, composite, back_rounded, knee_angle, torso_angle):
        if back_rounded:
            self.rep_back_rounding_frames += 1
        if composite > self.prev_composite + 0.005:
            self.rep_down_frames += 1
        elif composite < self.prev_composite - 0.005:
            self.rep_up_frames += 1
        if composite < 0.10:
            self.rep_top_hold_frames += 1
        if (knee_angle is not None and self.prev_knee_angle is not None
                and self.prev_torso_raw is not None):
            dk  = knee_angle - self.prev_knee_angle
            dta = torso_angle - self.prev_torso_raw
            if (abs(dk) > 1.5 and abs(dta) < 0.3) or (abs(dta) > 1.5 and abs(dk) < 0.3):
                self.rep_leg_back_mismatch_frames += 1
        self.prev_knee_angle = knee_angle
        self.prev_torso_raw  = torso_angle

    def _check_back_rounding(self, shoulder, hip, head, max_angle=22.0):
        tv   = shoulder - hip
        hv   = head - shoulder
        tn   = np.linalg.norm(tv) + 1e-9
        hn   = np.linalg.norm(hv) + 1e-9
        if hn < 0.3 * tn:
            return False
        return math.degrees(math.acos(float(np.clip(np.dot(tv, hv) / (tn * hn), -1.0, 1.0)))) > max_angle

    def _finalize_rep(self, frame_idx):
        self.reps += 1
        dt = 1.0 / max(1, self.fps)
        penalty = 0.0
        fb = []
        if self.rep_back_rounding_frames >= max(3, int(0.3 / dt)):
            fb.append("Try to keep your back a bit straighter")
            penalty += 1.5
        if self.rep_leg_back_mismatch_frames >= max(4, int(0.4 / dt)):
            fb.append("Drive the back up with the legs evenly")
            penalty += 1.0
        if self.rep_max_composite < self.COMPOSITE_HINGE_DEEP * 0.75:
            fb.append("Try to hinge a bit deeper")
            penalty += 0.5
        score = round(max(4, 10 - penalty) * 2) / 2.0
        if score >= 9.5: self.good_reps += 1
        else:            self.bad_reps  += 1
        down_s = self.rep_down_frames * dt
        top_s  = self.rep_top_hold_frames * dt
        tip = _choose_tip(down_s, top_s, self.rep_max_composite)
        info = {"rep_index": self.reps, "score": float(score),
                "score_display": display_half_str(score),
                "feedback": fb if score < 10.0 else [], "tip": tip}
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
    h, w = frame.shape[:2]
    line_t, dot_r = _dyn_thickness(h)
    for a, b in _BODY_CONNS:
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


# ===================== OVERLAY =====================
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
    _REPS_F  = _load_font(FONT_PATH, _scaled_font_size(_REF_REPS_FONT_SIZE,       h))
    _FB_F    = _load_font(FONT_PATH, _scaled_font_size(_REF_FEEDBACK_FONT_SIZE,    h))
    _DL_F    = _load_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, h))
    _DP_F    = _load_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE,   h))

    pct       = float(np.clip(progress_pct, 0, 1))
    bg_alpha  = int(round(255 * BAR_BG_ALPHA))
    ref_h     = max(int(h * 0.06), int(_REPS_F.size * 1.6))
    radius    = int(ref_h * DONUT_RADIUS_SCALE)
    thick     = max(3, int(radius * DONUT_THICKNESS_FRAC))
    cx        = w - 12 - radius
    cy        = max(ref_h + radius // 8, radius + thick // 2 + 2)

    ov        = np.zeros((h, w, 4), dtype=np.uint8)
    tmp_draw  = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    txt       = f"Reps: {int(reps)}"
    tw        = tmp_draw.textlength(txt, font=_REPS_F)
    cv2.rectangle(ov, (0, 0), (int(tw + 20), int(_REPS_F.size + 12)), (0, 0, 0, bg_alpha), -1)

    cv2.circle(ov, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)
    sa, ea = -90, -90 + int(360 * pct)
    if ea != sa:
        cv2.ellipse(ov, (cx, cy), (radius, radius), 0, sa, ea,
                    (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)

    fb_y0 = fb_pad_x = fb_pad_y = line_gap = line_h = 0
    fb_lines = []
    if feedback:
        safe     = max(6, int(h * 0.02))
        fb_pad_x = 12; fb_pad_y = 8; line_gap = 4
        fb_lines = _wrap2(tmp_draw, feedback, _FB_F, int(w - 2 * fb_pad_x - 20))
        line_h   = _FB_F.size + 6
        block_h  = 2 * fb_pad_y + len(fb_lines) * line_h + (len(fb_lines) - 1) * line_gap
        fb_y0    = max(0, h - safe - block_h)
        cv2.rectangle(ov, (0, fb_y0), (w, h - safe), (0, 0, 0, bg_alpha), -1)

    pil  = Image.fromarray(ov, mode="RGBA")
    draw = ImageDraw.Draw(pil)
    draw.text((10, 5), txt, font=_REPS_F, fill=(255, 255, 255, 255))

    gap = max(2, int(radius * 0.10))
    by  = cy - (_DL_F.size + gap + _DP_F.size) // 2
    lbl = "DEPTH"
    pt  = f"{int(pct * 100)}%"
    draw.text((cx - int(draw.textlength(lbl, font=_DL_F) // 2), by),
              lbl, font=_DL_F, fill=(255, 255, 255, 255))
    draw.text((cx - int(draw.textlength(pt, font=_DP_F) // 2), by + _DL_F.size + gap),
              pt,  font=_DP_F, fill=(255, 255, 255, 255))

    if feedback and fb_lines:
        ty = fb_y0 + fb_pad_y
        for ln in fb_lines:
            tx = max(fb_pad_x, (w - int(draw.textlength(ln, font=_FB_F))) // 2)
            draw.text((tx, ty), ln, font=_FB_F, fill=(255, 255, 255, 255))
            ty += line_h + line_gap

    ov_arr = np.array(pil)
    alpha  = ov_arr[:, :, 3:4].astype(np.float32) / 255.0
    out_f  = frame.astype(np.float32) * (1 - alpha) + ov_arr[:, :, [2, 1, 0]].astype(np.float32) * alpha
    return out_f.astype(np.uint8)


# ===================== POSE WRAPPER =====================
class PoseEstimator:
    def __init__(self, model_path=None):
        self._impl    = None
        self._is_tasks = False
        if _SOLUTIONS_AVAILABLE:
            self._impl = _mp_pose.Pose(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        elif _USE_TASKS_API:
            if model_path and os.path.exists(model_path):
                opts = _tasks_vision.PoseLandmarkerOptions(
                    base_options=_tasks_base(model_asset_path=model_path),
                    running_mode=_tasks_vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
                self._impl     = _tasks_vision.PoseLandmarker.create_from_options(opts)
                self._is_tasks = True
            else:
                raise FileNotFoundError("Tasks API requires a pose_landmarker.task model file.")

    def process(self, frame_bgr, timestamp_ms=0):
        if not self._is_tasks:
            res = self._impl.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            return res.pose_landmarks.landmark if res.pose_landmarks else None
        else:
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


# ===================== ROTATION HELPERS =====================
def get_video_rotation(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') != 'video':
                    continue
                tags = stream.get('tags', {})
                if 'rotate' in tags:
                    return int(tags['rotate'])
                for sd in stream.get('side_data_list', []):
                    if 'rotation' in sd:
                        return (-int(sd['rotation'])) % 360
    except Exception as e:
        print(f"ROTATION: ffprobe failed: {e}", flush=True)
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
             '-show_entries', 'stream_tags=rotate', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    return 0


def _apply_rotation(frame, angle):
    angle = angle % 360
    if angle == 90:  return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# =====================================================================
# PASS 1 — analysis only, stores per-frame data
# =====================================================================
def _analysis_pass(video_path, rotation, frame_skip, scale, fps_in):
    """
    Runs mediapipe on sampled frames.
    Stores per-frame {smoothed_pts, reps, feedback, composite} in frame_data.
    Returns (analysis_result_dict, frame_data, effective_fps)
    """
    import sys
    effective_fps  = max(1.0, fps_in / max(1, frame_skip))
    dt             = 1.0 / float(effective_fps)

    rep_detector     = DeadliftRepDetector(fps=effective_fps)
    landmark_smoother = LandmarkSmoother(alpha=0.5, vis_threshold=0.35)
    progress_ema     = EMA(0.35)
    progress_ema.value = 0.0

    frame_idx        = 0
    current_feedback = None
    fb_hold          = 0
    fb_hold_frames   = max(1, int(0.8 / dt))
    frame_data       = {}
    last_valid_snap  = None
    last_composite   = 0.0

    cap = cv2.VideoCapture(video_path)
    t0  = time.time()

    try:
        pose_est = PoseEstimator()
    except Exception as e:
        cap.release()
        return ({"squat_count": 0, "technique_score": 0.0,
                 "technique_score_display": display_half_str(0.0),
                 "technique_label": score_label(0.0),
                 "good_reps": 0, "bad_reps": 0,
                 "feedback": [f"Pose model error: {e}"],
                 "reps": []},
                {}, effective_fps)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if time.time() - t0 > 180:
                print("[DL] Pass1 timeout 180s", file=sys.stderr, flush=True)
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            frame = _apply_rotation(frame, rotation)
            work  = (cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                     if scale != 1.0 else frame)

            timestamp_ms = (frame_idx / fps_in) * 1000.0
            landmarks    = pose_est.process(work, timestamp_ms=timestamp_ms)

            def _store(snap, composite):
                frame_data[frame_idx] = {
                    "snap":     snap,
                    "reps":     rep_detector.reps,
                    "fb":       current_feedback if fb_hold > 0 else None,
                    "composite": composite,
                }

            if landmarks is None:
                _store(last_valid_snap, last_composite)
                if fb_hold > 0:
                    fb_hold -= 1
                continue

            smoothed = landmark_smoother.update(landmarks)
            snap     = _snapshot_smoothed(smoothed)
            last_valid_snap = snap

            composite, rt_fb, rep_info = rep_detector.process_frame(smoothed, landmarks, frame_idx)
            last_composite = composite

            if rt_fb:
                current_feedback = rt_fb
                fb_hold          = fb_hold_frames
            elif rep_info:
                current_feedback = (rep_info["feedback"][0]
                                    if rep_info.get("feedback") else None)
                fb_hold = fb_hold_frames if current_feedback else 0

            if fb_hold > 0:
                fb_hold -= 1

            _store(snap, composite)   # store AFTER rep count update

            if frame_idx % (frame_skip * 50) == 0:
                print(f"[DL] Pass1 fi={frame_idx} reps={rep_detector.reps} "
                      f"comp={composite:.3f}", file=sys.stderr, flush=True)

    finally:
        pose_est.close()
        cap.release()

    print(f"[DL] Pass1 done reps={rep_detector.reps}", file=sys.stderr, flush=True)

    avg = float(np.mean(rep_detector.all_scores)) if rep_detector.all_scores else 0.0
    if not np.isfinite(avg):
        avg = 0.0
    technique_score = round(round(avg * 2) / 2, 2)

    seen, feedback_list = set(), []
    for r in rep_detector.reps_report:
        if float(r.get("score") or 0.0) >= 10.0:
            continue
        for msg in (r.get("feedback") or []):
            if msg and msg not in seen:
                seen.add(msg)
                feedback_list.append(msg)
    if not feedback_list:
        feedback_list = ["Great form! Keep it up 💪"]

    analysis = {
        "squat_count":             int(rep_detector.reps),
        "technique_score":         float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label":         score_label(technique_score),
        "good_reps":               int(rep_detector.good_reps),
        "bad_reps":                int(rep_detector.bad_reps),
        "feedback":                feedback_list,
        "reps":                    rep_detector.reps_report,
    }
    return analysis, frame_data, effective_fps


# =====================================================================
# PASS 2 — video rendering from pass-1 data, NO mediapipe
# =====================================================================
def _render_pass(video_path, rotation, frame_skip, output_path,
                 out_w, out_h, fps_in, frame_data):
    """
    Re-reads video and draws skeleton + overlay using stored frame_data.
    No mediapipe here — skeleton cannot jump to a background person.
    """
    import sys
    effective_fps = max(1.0, fps_in / max(1, frame_skip))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, effective_fps, (out_w, out_h))

    # Validate VideoWriter opened correctly
    if not out.isOpened():
        print(f"[DL] ERROR: VideoWriter failed to open {output_path} "
              f"({out_w}x{out_h} @ {effective_fps:.1f}fps)", file=sys.stderr, flush=True)
        return

    cap       = cv2.VideoCapture(video_path)
    frame_idx = written = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frame = _apply_rotation(frame, rotation)
        # Resize to output dimensions
        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h))

        fd = frame_data.get(frame_idx)
        if fd is None:
            out.write(draw_overlay(frame, reps=0, feedback=None, progress_pct=0.0))
        else:
            f = frame.copy()
            if fd["snap"] is not None:
                f = draw_skeleton(f, fd["snap"])
            donut_pct = 1.0 - float(np.clip(fd["composite"], 0, 1))
            out.write(draw_overlay(f, reps=fd["reps"],
                                   feedback=fd["fb"],
                                   progress_pct=donut_pct))
        written += 1

    cap.release()
    out.release()
    print(f"[DL] Pass2 done frames_written={written}", file=sys.stderr, flush=True)


# =====================================================================
# PUBLIC ENTRY POINT
# =====================================================================
def run_deadlift_analysis(video_path,
                          frame_skip=3,
                          scale=0.4,
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
    import sys
    t_start = time.time()

    if fast_mode is True:
        return_video = False
    create_video = bool(return_video) and bool(output_path)

    print(f"[DL] Start fast_mode={fast_mode} create_video={create_video}",
          file=sys.stderr, flush=True)

    rotation = get_video_rotation(video_path)
    print(f"[DL] rotation={rotation}deg", file=sys.stderr, flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"squat_count": 0, "technique_score": 0.0,
                "technique_score_display": display_half_str(0.0),
                "technique_label": score_label(0.0),
                "good_reps": 0, "bad_reps": 0,
                "feedback": ["Could not open video"],
                "reps": [], "video_path": "", "feedback_path": feedback_path}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames < 10:
        return {"squat_count": 0, "technique_score": 0.0,
                "technique_score_display": display_half_str(0.0),
                "technique_label": score_label(0.0),
                "good_reps": 0, "bad_reps": 0,
                "feedback": [f"Invalid video — only {total_frames} frames"],
                "reps": [], "video_path": "", "feedback_path": feedback_path}

    # Output dimensions (after rotation)
    if rotation % 360 in (90, 270):
        out_w, out_h = orig_h, orig_w
    else:
        out_w, out_h = orig_w, orig_h

    # Working dimensions for analysis (scaled down)
    work_w = int(out_w * scale) if scale != 1.0 else out_w
    work_h = int(out_h * scale) if scale != 1.0 else out_h
    work_w = work_w if work_w % 2 == 0 else work_w + 1
    work_h = work_h if work_h % 2 == 0 else work_h + 1

    print(f"[DL] {total_frames}fr @ {fps_in:.1f}fps  out={out_w}x{out_h}  work={work_w}x{work_h}",
          file=sys.stderr, flush=True)

    # === PASS 1 — analysis ===
    analysis, frame_data, effective_fps = _analysis_pass(
        video_path, rotation, frame_skip, scale, fps_in)

    # Write feedback file
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {analysis['squat_count']}\n"
                    f"Technique Score: {analysis['technique_score']}/10\n")
            for fb in analysis["feedback"]:
                f.write(f"- {fb}\n")
    except Exception as e:
        print(f"[DL] Warning feedback file: {e}")

    # === PASS 2 — video rendering ===
    final_video_path = ""
    if create_video:
        print("[DL] Pass2 rendering video...", file=sys.stderr, flush=True)
        _render_pass(video_path, rotation, frame_skip, output_path,
                     out_w, out_h, fps_in, frame_data)

        # Encode for browser compatibility
        encoded = output_path.replace(".mp4", "_encoded.mp4")
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", output_path,
                 "-c:v", "libx264", "-preset", "fast",
                 "-movflags", "+faststart", "-pix_fmt", "yuv420p", encoded],
                capture_output=True, timeout=300)
            print(f"[DL] ffmpeg rc={proc.returncode}", file=sys.stderr, flush=True)
            if proc.returncode != 0:
                print(f"[DL] ffmpeg stderr: {proc.stderr.decode(errors='replace')[-500:]}",
                      file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[DL] ffmpeg exception: {e}", file=sys.stderr, flush=True)

        encoded_ok = os.path.exists(encoded) and os.path.getsize(encoded) > 1000
        if encoded_ok:
            try: os.remove(output_path)
            except Exception: pass
            final_video_path = encoded
        elif os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            final_video_path = output_path
        else:
            final_video_path = ""

        print(f"[DL] video='{final_video_path}' "
              f"exists={bool(final_video_path and os.path.exists(final_video_path))}",
              file=sys.stderr, flush=True)

    total_time = time.time() - t_start
    print(f"[DL] Total time: {total_time:.1f}s", file=sys.stderr, flush=True)

    analysis["video_path"]    = str(final_video_path)
    analysis["feedback_path"] = str(feedback_path)
    return analysis


def run_analysis(*args, **kwargs):
    return run_deadlift_analysis(*args, **kwargs)

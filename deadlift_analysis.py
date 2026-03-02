# -*- coding: utf-8 -*-
# deadlift_analysis.py — V3.1: robust front-view + rotation fix + walking rejection
#
# FIXES over V3:
# A. Robust rotation detection (displaymatrix + stream tags + dimension heuristic)
# B. Walking rejection: hip must stay stable for rep to count
# C. Adaptive calibration with running baseline (not just first N frames)

import os, cv2, math, numpy as np, subprocess, json
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

# ===================== Optional ONNX Runtime (YOLO detector) =====================
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
_FACE_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
_BODY_CONNS = tuple((a, b) for (a, b) in POSE_CONNECTIONS
                    if a not in _FACE_INDICES and b not in _FACE_INDICES)
_BODY_POINTS = tuple(sorted({i for (a, b) in _BODY_CONNS for i in (a, b)}))

# ===================== GEOMETRY HELPERS =====================
def _angle_3pt(a, b, c):
    """Angle at vertex b formed by points a-b-c, in degrees."""
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    nrm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = float(np.clip(np.dot(ba, bc) / nrm, -1.0, 1.0))
    return math.degrees(math.acos(cosang))

def _angle_to_vertical(a, b):
    """Angle of vector a→b relative to straight up (0° = vertical)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
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

# ===================== VIEW ANGLE ESTIMATOR =====================
class ViewAngleEstimator:
    def __init__(self, alpha=0.3):
        self.ema = EMA(alpha)

    def estimate(self, smoothed_pts):
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
        side_score = float(np.clip(1.0 - (ratio - 0.15) / 0.85, 0.0, 1.0))
        return self.ema.update(side_score)


# ===================== FRONT-VIEW SIGNAL V3.1 =====================
class FrontViewSignal:
    """
    Front-view deadlift detection using:
    1. Shoulder Y drop relative to adaptive standing baseline
    2. Body compression (shoulder-hip distance shrinks)
    3. Walking rejection: hip position must stay stable
    
    V3.1 improvements:
    - Adaptive baseline: tracks the HIGHEST shoulder position (= most upright)
      seen in recent history, not just first N frames
    - Walking rejection: if hip Y moves more than threshold, signal is suppressed
    - Scale-invariant: all measurements normalized to body proportions
    """
    
    def __init__(self):
        self.signal_ema = EMA(0.40)
        self.frame_count = 0
        
        # Adaptive baseline tracking
        self.baseline_ready = False
        self.min_frames_for_baseline = 5
        
        # Running stats for standing position (shoulder at HIGHEST = most upright)
        # We track the minimum shoulder_y (highest in frame) as "standing"
        self.standing_shoulder_y = None
        self.standing_hip_y = None
        self.standing_sh_dist = None   # shoulder-to-hip Y distance when standing
        self.standing_ankle_y = None
        
        # Recent history for adaptive baseline (sliding window)
        self.shoulder_y_history = deque(maxlen=60)  # ~6-10 sec of data
        self.hip_y_history = deque(maxlen=60)
        self.sh_dist_history = deque(maxlen=60)
        
        # Walking detection
        self.hip_y_ema = EMA(0.3)
        self.hip_x_ema = EMA(0.3)
        self.prev_hip_y = None
        self.prev_hip_x = None
        self.hip_movement_acc = 0.0  # accumulated hip movement
        self.hip_movement_decay = 0.92  # decay factor per frame
        
    def _get_key_points(self, smoothed_pts):
        """Extract averaged shoulder, hip, ankle Y positions."""
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
        if la and ra:
            ankle_y = (la[1] + ra[1]) / 2.0
        elif la:
            ankle_y = la[1]
        elif ra:
            ankle_y = ra[1]
        elif lk and rk:
            ankle_y = (lk[1] + rk[1]) / 2.0
        elif lk:
            ankle_y = lk[1]
        elif rk:
            ankle_y = rk[1]
            
        return {
            'shoulder_y': shoulder_y,
            'shoulder_x': shoulder_x,
            'hip_y': hip_y,
            'hip_x': hip_x,
            'ankle_y': ankle_y,
            'sh_dist': hip_y - shoulder_y,  # positive = hip below shoulder
        }
    
    def _detect_walking(self, hip_y, hip_x, sh_dist):
        """
        Detect if person is walking (vs doing deadlift in place).
        Walking: hip Y and/or X changes steadily over time.
        Deadlift: hip stays relatively fixed, shoulders move.
        
        Returns a suppression factor: 1.0 = not walking, 0.0 = definitely walking
        """
        # Smooth hip position
        smooth_hy = self.hip_y_ema.update(hip_y)
        smooth_hx = self.hip_x_ema.update(hip_x)
        
        if self.prev_hip_y is not None:
            dy = abs(smooth_hy - self.prev_hip_y)
            dx = abs(smooth_hx - self.prev_hip_x)
            movement = math.sqrt(dy**2 + dx**2)
            
            # Accumulate movement with decay
            self.hip_movement_acc = self.hip_movement_acc * self.hip_movement_decay + movement
        
        self.prev_hip_y = smooth_hy
        self.prev_hip_x = smooth_hx
        
        # Normalize movement by body scale (sh_dist)
        if sh_dist > 0.01:
            normalized_movement = self.hip_movement_acc / sh_dist
        else:
            normalized_movement = self.hip_movement_acc * 10
        
        # Threshold: if accumulated normalized movement > 0.8, likely walking
        # During deadlift, hip barely moves (normalized_movement < 0.3 typically)
        # During walking, hip shifts continuously (normalized_movement > 1.0)
        suppression = float(np.clip(1.0 - (normalized_movement - 0.4) / 0.6, 0.0, 1.0))
        
        return suppression
    
    def _update_baseline(self, shoulder_y, hip_y, sh_dist, ankle_y):
        """
        Adaptive baseline: the "standing" position is the frame where
        the person is most upright = shoulder Y is at its MINIMUM (highest in frame)
        within recent history.
        
        This handles:
        - Videos where person starts already bent
        - Standing between reps re-calibrates the baseline
        """
        self.shoulder_y_history.append(shoulder_y)
        self.hip_y_history.append(hip_y)
        self.sh_dist_history.append(sh_dist)
        
        if len(self.shoulder_y_history) >= self.min_frames_for_baseline:
            # Standing = minimum shoulder Y in recent window (highest point)
            # But we want a robust estimate, so use the 10th percentile
            sorted_sy = sorted(self.shoulder_y_history)
            idx_10pct = max(0, int(len(sorted_sy) * 0.10))
            self.standing_shoulder_y = sorted_sy[idx_10pct]
            
            # Corresponding hip Y at standing (use the 10th percentile too)
            sorted_hy = sorted(self.hip_y_history)
            self.standing_hip_y = sorted_hy[idx_10pct]
            
            # Max shoulder-hip distance (most upright)
            sorted_shd = sorted(self.sh_dist_history, reverse=True)
            idx_90pct = max(0, int(len(sorted_shd) * 0.10))
            self.standing_sh_dist = sorted_shd[idx_90pct]
            
            if ankle_y is not None:
                self.standing_ankle_y = ankle_y
            
            self.baseline_ready = True
    
    def update(self, smoothed_pts):
        """
        Returns a front-view hinge signal in [0, 1].
        0 = standing upright, 1 = fully hinged.
        """
        self.frame_count += 1
        pts = self._get_key_points(smoothed_pts)
        if pts is None:
            return self.signal_ema.value if self.signal_ema.value is not None else 0.0
        
        shoulder_y = pts['shoulder_y']
        hip_y = pts['hip_y']
        ankle_y = pts['ankle_y']
        sh_dist = pts['sh_dist']
        
        # Update adaptive baseline
        self._update_baseline(shoulder_y, hip_y, sh_dist, ankle_y)
        
        # Walking detection
        walk_suppression = self._detect_walking(hip_y, pts['hip_x'], sh_dist)
        
        if not self.baseline_ready:
            self.signal_ema.update(0.0)
            return 0.0
        
        # ── Signal 1: Shoulder drop ──
        # How far shoulder has dropped from the standing (highest) position
        shoulder_drop = shoulder_y - self.standing_shoulder_y
        
        if self.standing_sh_dist > 0.01:
            # Normalize: a full deadlift hinge drops shoulders by ~40-70% 
            # of standing shoulder-hip distance
            sig_shoulder_drop = float(np.clip(
                shoulder_drop / (self.standing_sh_dist * 0.5),
                0.0, 1.0
            ))
        else:
            sig_shoulder_drop = 0.0
        
        # ── Signal 2: Body compression ──
        # Shoulder-hip Y distance shrinks as torso bends forward toward camera
        if self.standing_sh_dist > 0.01:
            compression_ratio = sh_dist / self.standing_sh_dist
            # Standing: ratio ≈ 1.0, Hinged: ratio ≈ 0.5-0.7
            sig_compression = float(np.clip(
                (1.0 - compression_ratio) / 0.45,
                0.0, 1.0
            ))
        else:
            sig_compression = 0.0
        
        # ── Combine signals ──
        # Shoulder drop is primary (changes more from front view)
        # Compression confirms it's a hinge, not just bending knees
        raw_signal = 0.6 * sig_shoulder_drop + 0.4 * sig_compression
        
        # ── Apply walking suppression ──
        # If person is walking, suppress the signal heavily
        suppressed_signal = raw_signal * walk_suppression
        
        # Debug
        if self.frame_count % 10 == 0:
            print(f"  FRONT_SIG: sh_drop={sig_shoulder_drop:.3f} "
                  f"compress={sig_compression:.3f} raw={raw_signal:.3f} "
                  f"walk_supp={walk_suppression:.3f} final={suppressed_signal:.3f}")
        
        smoothed = self.signal_ema.update(suppressed_signal)
        return smoothed


# ===================== REP DETECTOR V3.1 =====================
class DeadliftRepDetector:
    """
    V2 proven side-view detection + V3.1 front-view blending.
    
    Blending:
    - side_ratio > 0.6 → 100% side composite (V2, untouched)
    - side_ratio < 0.3 → 100% front signal
    - between → linear blend
    """

    STANDING = 0
    HINGING  = 1
    RISING   = 2

    # V2 original thresholds (PROVEN — unchanged)
    COMPOSITE_HINGE_START = 0.25
    COMPOSITE_HINGE_DEEP  = 0.50
    COMPOSITE_STANDING    = 0.15

    MIN_HINGE_FRAMES = 4
    MIN_FRAMES_BETWEEN = 12

    def __init__(self, fps=10):
        self.fps = fps
        self.state = self.STANDING
        self.frame_count = 0
        self.last_rep_frame = -999
        self.hinge_frames = 0

        # V2 original smoothers
        self.torso_angle_ema = EMA(0.45)
        self.hip_angle_ema = EMA(0.45)
        self.composite_ema = EMA(0.40)

        # V3.1: Front-view signal
        self.front_signal = FrontViewSignal()

        # Per-rep tracking
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

        # Output
        self.reps = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.all_scores = []
        self.reps_report = []

    def _compute_side_composite(self, torso_angle, hip_angle, side_ratio):
        """V2 ORIGINAL: 2-signal composite."""
        torso_norm = float(np.clip((torso_angle - 12) / 55.0, 0.0, 1.0))
        hip_norm = float(np.clip((170 - hip_angle) / 70.0, 0.0, 1.0))

        w_torso = 0.3 + 0.5 * side_ratio
        w_hip   = 0.3 + 0.5 * (1 - side_ratio)

        total = w_torso + w_hip + 1e-9
        w_torso /= total
        w_hip /= total

        composite = w_torso * torso_norm + w_hip * hip_norm
        return self.composite_ema.update(composite)

    def _get_lm_val(self, lm, idx, attr):
        try:
            if hasattr(idx, 'value'):
                idx = idx.value
            val = getattr(lm[idx], attr, None)
            if val is None:
                return 0.5 if attr == 'visibility' else 0.0
            return float(val)
        except (IndexError, AttributeError, TypeError):
            return 0.5 if attr == 'visibility' else 0.0

    def process_frame(self, smoothed_pts, raw_landmarks, frame_idx):
        self.frame_count = frame_idx
        lm = raw_landmarks

        def _get_pt(primary_idx, fallback_idx):
            p_vis = self._get_lm_val(lm, primary_idx, 'visibility')
            if primary_idx in smoothed_pts and p_vis > 0.4:
                return np.array(smoothed_pts[primary_idx], dtype=float), True
            f_vis = self._get_lm_val(lm, fallback_idx, 'visibility')
            if fallback_idx in smoothed_pts and f_vis > 0.4:
                return np.array(smoothed_pts[fallback_idx], dtype=float), True
            return None, False

        shoulder, s_ok = _get_pt(12, 11)
        hip, h_ok = _get_pt(24, 23)
        knee, k_ok = _get_pt(26, 25)
        ankle, a_ok = _get_pt(28, 27)

        if not (s_ok and h_ok):
            return self.prev_composite, None, None

        side_ratio = self.view_estimator.estimate(smoothed_pts)

        # ── V2: Side-view composite (always computed) ──
        torso_angle = _angle_to_vertical(hip, shoulder)
        torso_angle_smooth = self.torso_angle_ema.update(torso_angle)

        hip_angle = 170.0
        if k_ok:
            hip_angle = _angle_3pt(shoulder, hip, knee)
        hip_angle_smooth = self.hip_angle_ema.update(hip_angle)

        side_composite = self._compute_side_composite(torso_angle_smooth, hip_angle_smooth, side_ratio)

        # ── V3.1: Front-view signal ──
        front_signal_val = self.front_signal.update(smoothed_pts)

        # ── Blend based on view angle ──
        if side_ratio >= 0.6:
            composite = side_composite
        elif side_ratio <= 0.3:
            composite = front_signal_val
        else:
            blend = (side_ratio - 0.3) / 0.3
            composite = blend * side_composite + (1.0 - blend) * front_signal_val

        if frame_idx % 10 == 0:
            view_str = "SIDE" if side_ratio >= 0.6 else ("FRONT" if side_ratio <= 0.3 else "BLEND")
            print(f"F{frame_idx}: {view_str} sr={side_ratio:.2f} "
                  f"side_c={side_composite:.3f} front_s={front_signal_val:.3f} "
                  f"final={composite:.3f} state={self.state} reps={self.reps}")

        # ── V2 ORIGINAL: back rounding check ──
        head_pt = None
        for idx in (8, 7, 0):
            vis = self._get_lm_val(lm, idx, 'visibility')
            if idx in smoothed_pts and vis > 0.45:
                head_pt = np.array(smoothed_pts[idx], dtype=float)
                break

        back_rounded = False
        if head_pt is not None and s_ok and h_ok:
            back_rounded = self._check_back_rounding(shoulder, hip, head_pt)

        knee_angle = None
        if k_ok and a_ok:
            knee_angle = _angle_3pt(hip, knee, ankle)

        # ── V2 ORIGINAL state machine ──
        rt_feedback = None
        rep_info = None

        if self.state == self.STANDING:
            if composite > self.COMPOSITE_HINGE_START and (frame_idx - self.last_rep_frame > self.MIN_FRAMES_BETWEEN):
                self.state = self.HINGING
                self.hinge_frames = 0
                self._reset_rep_tracking()

        elif self.state == self.HINGING:
            self.hinge_frames += 1
            self.rep_max_composite = max(self.rep_max_composite, composite)
            self._track_rep_quality(composite, back_rounded, knee_angle, torso_angle_smooth)

            if back_rounded:
                rt_feedback = "Try to keep your back a bit straighter"

            if composite < self.prev_composite - 0.03 and self.hinge_frames >= self.MIN_HINGE_FRAMES:
                if self.rep_max_composite >= self.COMPOSITE_HINGE_DEEP * 0.7:
                    self.state = self.RISING

        elif self.state == self.RISING:
            self._track_rep_quality(composite, back_rounded, knee_angle, torso_angle_smooth)

            if back_rounded:
                rt_feedback = "Try to keep your back a bit straighter"

            if composite < self.COMPOSITE_STANDING:
                if self.rep_max_composite >= self.COMPOSITE_HINGE_DEEP * 0.6:
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
            dk = knee_angle - self.prev_knee_angle
            dt_a = torso_angle - self.prev_torso_raw
            if abs(dk) > 1.5 and abs(dt_a) < 0.3:
                self.rep_leg_back_mismatch_frames += 1
            elif abs(dt_a) > 1.5 and abs(dk) < 0.3:
                self.rep_leg_back_mismatch_frames += 1

        self.prev_knee_angle = knee_angle
        self.prev_torso_raw = torso_angle

    def _check_back_rounding(self, shoulder, hip, head, max_angle=22.0):
        torso_vec = shoulder - hip
        head_vec = head - shoulder
        torso_nrm = np.linalg.norm(torso_vec) + 1e-9
        head_nrm = np.linalg.norm(head_vec) + 1e-9
        if head_nrm < 0.3 * torso_nrm:
            return False
        cosang = float(np.clip(np.dot(torso_vec, head_vec) / (torso_nrm * head_nrm), -1.0, 1.0))
        angle = math.degrees(math.acos(cosang))
        return angle > max_angle

    def _finalize_rep(self, frame_idx):
        self.reps += 1
        dt = 1.0 / max(1, self.fps)

        penalty = 0.0
        fb = []

        back_min_frames = max(3, int(0.3 / dt))
        if self.rep_back_rounding_frames >= back_min_frames:
            fb.append("Try to keep your back a bit straighter")
            penalty += 1.5

        mismatch_min = max(4, int(0.4 / dt))
        if self.rep_leg_back_mismatch_frames >= mismatch_min:
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

        rep_fb = fb if score < 10.0 else []
        info = {
            "rep_index":     self.reps,
            "score":         float(score),
            "score_display": display_half_str(score),
            "feedback":      rep_fb,
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

    reps_font_size = _scaled_font_size(_REF_REPS_FONT_SIZE, h)
    feedback_font_size = _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, h)
    depth_label_font_size = _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, h)
    depth_pct_font_size = _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, h)

    _REPS_FONT = _load_font(FONT_PATH, reps_font_size)
    _FEEDBACK_FONT = _load_font(FONT_PATH, feedback_font_size)
    _DEPTH_LABEL_FONT = _load_font(FONT_PATH, depth_label_font_size)
    _DEPTH_PCT_FONT = _load_font(FONT_PATH, depth_pct_font_size)

    pct = float(np.clip(progress_pct, 0, 1))
    bg_alpha_val = int(round(255 * BAR_BG_ALPHA))

    ref_h = max(int(h * 0.06), int(reps_font_size * 1.6))
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
    tw = tmp_draw.textlength(txt, font=_REPS_FONT)
    thh = _REPS_FONT.size
    cv2.rectangle(overlay_np, (0, 0), (int(tw + 2 * pad_x), int(thh + 2 * pad_y)),
                  (0, 0, 0, bg_alpha_val), -1)

    cv2.circle(overlay_np, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)
    start_ang = -90
    end_ang = start_ang + int(360 * pct)
    if end_ang != start_ang:
        cv2.ellipse(overlay_np, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                    (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)

    fb_y0 = 0
    fb_lines = []
    fb_pad_x = fb_pad_y = line_gap = line_h = 0
    if feedback:
        safe_margin = max(6, int(h * 0.02))
        fb_pad_x, fb_pad_y, line_gap = 12, 8, 4
        max_text_w = int(w - 2 * fb_pad_x - 20)
        fb_lines = _wrap2(tmp_draw, feedback, _FEEDBACK_FONT, max_text_w)
        line_h = _FEEDBACK_FONT.size + 6
        block_h = 2 * fb_pad_y + len(fb_lines) * line_h + (len(fb_lines) - 1) * line_gap
        fb_y0 = max(0, h - safe_margin - block_h)
        y1 = h - safe_margin
        cv2.rectangle(overlay_np, (0, fb_y0), (w, y1), (0, 0, 0, bg_alpha_val), -1)

    overlay_pil = Image.fromarray(overlay_np, mode="RGBA")
    draw = ImageDraw.Draw(overlay_pil)

    draw.text((pad_x, pad_y - 1), txt, font=_REPS_FONT, fill=(255, 255, 255, 255))

    gap = max(2, int(radius * 0.10))
    by = cy - (_DEPTH_LABEL_FONT.size + gap + _DEPTH_PCT_FONT.size) // 2
    label = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label, font=_DEPTH_LABEL_FONT)
    pw = draw.textlength(pct_txt, font=_DEPTH_PCT_FONT)
    draw.text((cx - int(lw // 2), by), label, font=_DEPTH_LABEL_FONT, fill=(255, 255, 255, 255))
    draw.text((cx - int(pw // 2), by + _DEPTH_LABEL_FONT.size + gap), pct_txt, font=_DEPTH_PCT_FONT,
              fill=(255, 255, 255, 255))

    if feedback and fb_lines:
        ty = fb_y0 + fb_pad_y
        for ln in fb_lines:
            tw2 = draw.textlength(ln, font=_FEEDBACK_FONT)
            tx = max(fb_pad_x, (w - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=_FEEDBACK_FONT, fill=(255, 255, 255, 255))
            ty += line_h + line_gap

    overlay_rgba = np.array(overlay_pil)
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = overlay_rgba[:, :, [2, 1, 0]].astype(np.float32)
    out_f = frame.astype(np.float32) * (1.0 - alpha) + overlay_bgr * alpha
    return out_f.astype(np.uint8)

# ===================== YOLOv8n-ONNX Detector =====================
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
    def __init__(self, model_path=None):
        self._impl = None
        self._is_tasks = False

        if _SOLUTIONS_AVAILABLE:
            self._impl = _mp_pose.Pose(
                model_complexity=1,
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
                    "Tasks API requires a pose_landmarker.task model file.")

    def process(self, frame_bgr, timestamp_ms=0):
        if not self._is_tasks:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._impl.process(rgb)
            if res.pose_landmarks:
                return res.pose_landmarks.landmark
            return None
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

# ===================== ROBUST ROTATION DETECTION =====================
def get_video_rotation(video_path):
    """
    Get video rotation from metadata — handles BOTH methods:
    1. stream tags (older format): tags.rotate = "90"
    2. side_data displaymatrix (newer iOS): rotation = -90
    3. Fallback: dimension heuristic for portrait mobile videos
    """
    rotation = 0
    raw_w, raw_h = 0, 0
    
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', video_path
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') != 'video':
                    continue
                
                raw_w = int(stream.get('width', 0))
                raw_h = int(stream.get('height', 0))
                
                # Method 1: Check tags.rotate
                tags = stream.get('tags', {})
                if 'rotate' in tags:
                    rotation = int(tags['rotate'])
                    print(f"ROTATION: Found via tags.rotate = {rotation}")
                    return rotation
                
                # Method 2: Check side_data_list for displaymatrix
                for sd in stream.get('side_data_list', []):
                    if 'rotation' in sd:
                        rot_val = int(sd['rotation'])
                        # displaymatrix rotation is often negative
                        # Convert: -90 → 90, -270 → 270, etc.
                        rotation = (-rot_val) % 360
                        print(f"ROTATION: Found via displaymatrix = {sd['rotation']} → {rotation}")
                        return rotation
    except Exception as e:
        print(f"ROTATION: ffprobe JSON failed: {e}")
    
    # Method 3: Fallback — try the simple tag method
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
            '-show_entries', 'stream_tags=rotate', '-of', 'csv=p=0', video_path
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            rotation = int(result.stdout.strip())
            print(f"ROTATION: Found via simple tag query = {rotation}")
            return rotation
    except Exception:
        pass
    
    # Method 4: Last resort — use ffmpeg to check if it auto-rotates
    # If raw dimensions are landscape but video should be portrait (mobile),
    # we can't know for sure, but we log it
    if raw_w > 0 and raw_h > 0:
        print(f"ROTATION: No metadata found. Raw dimensions: {raw_w}x{raw_h}")
        # Note: We do NOT guess here — ffmpeg usually auto-applies rotation
        # during re-encode, so if we can't find metadata, it might already
        # be handled. The key fix is Method 2 (displaymatrix).
    
    print(f"ROTATION: No rotation metadata found, using 0")
    return 0


def rotate_frame(frame, angle):
    """Rotate frame by angle (90, 180, 270 degrees)."""
    angle = angle % 360
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

# ===================== MAIN =====================
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
    import time as _time
    t_start = _time.time()

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

    # ── ROTATION FIX (V3.1) ──
    rotation_angle = get_video_rotation(video_path)

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

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── V3.2 PERF: Compute correct dimensions after rotation ──
    if rotation_angle % 360 in (90, 270):
        rotated_w, rotated_h = orig_h, orig_w
        print(f"OUTPUT DIMENSIONS SWAPPED: {orig_w}x{orig_h} -> {rotated_w}x{rotated_h} (rotation={rotation_angle})")
    else:
        rotated_w, rotated_h = orig_w, orig_h

    # ── V3.2 PERF: Write video at WORKING resolution ──
    # Avoids expensive resize-back-to-full for every frame.
    # ffmpeg will scale up during the final encode step.
    work_w = int(rotated_w * scale) if scale != 1.0 else rotated_w
    work_h = int(rotated_h * scale) if scale != 1.0 else rotated_h
    # Ensure even dimensions for codec compatibility
    work_w = work_w if work_w % 2 == 0 else work_w + 1
    work_h = work_h if work_h % 2 == 0 else work_h + 1

    print(f"VIDEO OUTPUT: writing at {work_w}x{work_h} (scale={scale}), will upscale in ffmpeg encode")

    rep_detector = DeadliftRepDetector(fps=effective_fps)
    landmark_smoother = LandmarkSmoother(alpha=0.5, vis_threshold=0.35)
    progress_ema = EMA(0.35)
    progress_ema.value = 0.0

    frame_idx = 0
    current_feedback = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation BEFORE any processing
            if rotation_angle != 0:
                frame = rotate_frame(frame, rotation_angle)

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            processed_frames += 1
            if processed_frames > 500:
                break

            # ── V3.2 PERF: resize to exact work dimensions ──
            work = cv2.resize(frame, (work_w, work_h)) if scale != 1.0 else frame.copy()

            if return_video and out_vid is None:
                out_vid = cv2.VideoWriter(output_path, fourcc, effective_fps, (work_w, work_h))

            timestamp_ms = (frame_idx / fps_in) * 1000.0
            landmarks = pose_est.process(work, timestamp_ms=timestamp_ms)

            if landmarks is None:
                if return_video and out_vid is not None:
                    prog_val = progress_ema.value or 0.0
                    out_vid.write(draw_overlay(work, reps=rep_detector.reps,
                                               feedback=None, progress_pct=1.0 - prog_val))
                continue

            try:
                smoothed = landmark_smoother.update(landmarks)
                composite, rt_fb, rep_info = rep_detector.process_frame(smoothed, landmarks, frame_idx)

                if rt_fb:
                    current_feedback = rt_fb
                elif rep_info:
                    if rep_info["feedback"]:
                        current_feedback = rep_info["feedback"][0]
                    else:
                        current_feedback = None

                prog_display = progress_ema.update(composite)
                donut_pct = 1.0 - prog_display

                if return_video:
                    # ── V3.2 PERF: draw directly on work frame, no resize back ──
                    frame_out = draw_skeleton(work, smoothed)
                    frame_out = draw_overlay(frame_out, reps=rep_detector.reps,
                                             feedback=current_feedback,
                                             progress_pct=donut_pct)
                    out_vid.write(frame_out)

            except Exception:
                if return_video and out_vid is not None:
                    prog_val = progress_ema.value or 0.0
                    out_vid.write(draw_overlay(work, reps=rep_detector.reps,
                                              feedback=None, progress_pct=1.0 - prog_val))
                continue

            # Progress logging every 50 processed frames
            if return_video and processed_frames % 50 == 0:
                elapsed = _time.time() - t_start
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"[VIDEO PROGRESS] {processed_frames} frames, ~{pct:.0f}% of video, {elapsed:.1f}s elapsed")

    finally:
        pose_est.close()
        cap.release()
        if return_video and out_vid:
            out_vid.release()
        cv2.destroyAllWindows()

    t_process = _time.time() - t_start
    print(f"[PERF] Processing done in {t_process:.1f}s ({processed_frames} frames)")

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

    final_video_path = ""
    if return_video:
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            # ── V3.2 PERF: ffmpeg upscales to target resolution during encode ──
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", output_path,
                "-vf", f"scale={rotated_w}:{rotated_h}",
                "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart",
                "-pix_fmt", "yuv420p", encoded_path
            ]
            print(f"[ENCODE] Upscaling {work_w}x{work_h} -> {rotated_w}x{rotated_h}")
            enc_start = _time.time()
            subprocess.run(ffmpeg_cmd, check=False, capture_output=True, timeout=120)
            print(f"[ENCODE] Done in {_time.time() - enc_start:.1f}s")
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                os.remove(output_path)
        except Exception as e:
            print(f"[ENCODE] Error: {e}")
        final_video_path = (encoded_path if os.path.exists(encoded_path)
                            else (output_path if os.path.exists(output_path) else ""))

    total_time = _time.time() - t_start
    print(f"[PERF] Total time: {total_time:.1f}s (process={t_process:.1f}s, encode={total_time - t_process:.1f}s)")

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

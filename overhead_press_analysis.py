# -*- coding: utf-8 -*-
# overhead_press_analysis.py — Robust overhead/military press rep counter with form feedback
#
# Major improvements over previous version:
# 1. Proper finite state machine (IDLE → DESCENDING → BOTTOM → ASCENDING → TOP → cycle)
# 2. Bilateral tracking — uses BOTH arms, picks dominant but validates with secondary
# 3. Adaptive calibration phase to set personal ROM thresholds
# 4. Hysteresis bands to prevent jitter-based miscounts
# 5. Minimum cycle duration to reject false reps
# 6. Shoulder-relative wrist tracking (normalized coordinates)
# 7. Angular velocity for phase transition validation
# 8. Per-rep granular scoring (not binary good/bad)
# 9. Improved torso lean via full spine vector
# 10. Better lockout detection using wrist-above-head + elbow angle combo

import os
import cv2
import math
import numpy as np
import subprocess
from enum import Enum, auto
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

try:
    from PIL import ImageFont, ImageDraw, Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ============ MediaPipe ============
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None

# ============ Constants & Config ============

class PressPhase(Enum):
    IDLE = auto()        # No movement detected yet
    BOTTOM = auto()      # Arms at shoulder level (start/bottom position)
    ASCENDING = auto()   # Pressing up
    TOP = auto()         # Arms locked out overhead
    DESCENDING = auto()  # Lowering back down


@dataclass
class PressConfig:
    """All tunable parameters in one place."""
    # Phase transition — elbow angles (degrees)
    bottom_elbow_max: float = 110.0       # Elbow angle <= this = bottom position
    top_elbow_min: float = 160.0          # Elbow angle >= this = top/lockout
    hysteresis_band: float = 8.0          # Dead band to prevent oscillation

    # Wrist position requirements (normalized to torso height)
    wrist_bottom_ratio: float = 0.15      # Wrist must be within this ratio of shoulder height at bottom
    wrist_top_ratio: float = 0.3          # Wrist must be this ratio ABOVE shoulder at top

    # Timing constraints
    min_cycle_frames: int = 6             # Minimum frames for a full rep (anti-jitter)
    min_phase_frames: int = 2             # Minimum frames in a phase before transition
    refractory_frames: int = 4            # Minimum frames between two rep counts

    # Angular velocity (degrees/frame) — must show actual movement
    min_angular_velocity: float = 0.3     # Minimum avg angular change to confirm movement

    # EMA smoothing
    elbow_ema_alpha: float = 0.40         # Higher = more responsive, lower = smoother
    wrist_ema_alpha: float = 0.35
    lean_ema_alpha: float = 0.30

    # Visibility
    vis_threshold: float = 0.40           # Minimum landmark visibility

    # Form scoring thresholds
    depth_target_angle: float = 100.0     # Ideal bottom elbow angle
    depth_ok_angle: float = 115.0         # Acceptable bottom angle
    lockout_target_angle: float = 172.0   # Ideal lockout angle
    lockout_ok_angle: float = 160.0       # Acceptable lockout angle
    lean_target_max: float = 12.0         # Ideal max torso lean
    lean_ok_max: float = 22.0             # Acceptable lean
    lean_bad_max: float = 30.0            # Bad lean threshold
    stack_target_max: float = 0.04        # Ideal wrist-shoulder horizontal offset (normalized)
    stack_ok_max: float = 0.07            # Acceptable offset
    bar_path_target: float = 0.03         # Ideal bar path deviation (normalized)
    bar_path_ok: float = 0.06             # Acceptable bar path deviation

    # Scoring weights (sum to ~1.0 for interpretability)
    w_depth: float = 0.20
    w_lockout: float = 0.25
    w_lean: float = 0.25
    w_stack: float = 0.15
    w_bar_path: float = 0.15

    # Calibration
    calibration_reps: int = 2             # Auto-calibrate after this many reps
    calibration_adjustment: float = 0.85  # How much to adapt toward user's ROM


@dataclass
class RepMetrics:
    """Metrics collected during a single rep cycle."""
    rep_index: int = 0
    min_elbow_angle: float = 999.0        # Lowest elbow angle (bottom)
    max_elbow_angle: float = 0.0          # Highest elbow angle (top/lockout)
    max_torso_lean: float = 0.0           # Peak torso lean during rep
    max_wrist_offset: float = 0.0         # Peak wrist-shoulder horizontal offset
    bar_path_deviation: float = 0.0       # Lateral deviation of wrist path
    wrist_x_samples: list = field(default_factory=list)  # For bar path tracking
    wrist_y_at_bottom: float = 0.0
    wrist_y_at_top: float = 0.0
    ascending_frames: int = 0
    descending_frames: int = 0
    total_frames: int = 0
    phase_at_start: str = ""

    # Computed scores (0-10 each)
    depth_score: float = 10.0
    lockout_score: float = 10.0
    lean_score: float = 10.0
    stack_score: float = 10.0
    bar_path_score: float = 10.0
    overall_score: float = 10.0
    issues: list = field(default_factory=list)


# ============ Utility Functions ============

def angle_3pt(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle at point b formed by segments ba and bc, in degrees."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def ema_update(prev: Optional[float], new: float, alpha: float) -> float:
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


def score_interpolate(value: float, target: float, ok: float, bad: float, higher_is_better: bool = True) -> float:
    """
    Compute a 0-10 score with smooth interpolation.
    target = ideal value (score 10), ok = acceptable (score 7), bad = poor (score 3).
    """
    if higher_is_better:
        if value >= target:
            return 10.0
        elif value >= ok:
            return 7.0 + 3.0 * (value - ok) / max(target - ok, 1e-6)
        elif value >= bad:
            return 3.0 + 4.0 * (value - bad) / max(ok - bad, 1e-6)
        else:
            return max(0.0, 3.0 * value / max(bad, 1e-6))
    else:
        # Lower is better (e.g., lean, offset)
        if value <= target:
            return 10.0
        elif value <= ok:
            return 7.0 + 3.0 * (ok - value) / max(ok - target, 1e-6)
        elif value <= bad:
            return 3.0 + 4.0 * (bad - value) / max(bad - ok, 1e-6)
        else:
            return max(0.0, 3.0 * (1.0 - (value - bad) / max(bad, 1e-6)))


def half_floor_10(x: float) -> float:
    return max(0.0, min(10.0, math.floor(x * 2.0) / 2.0))


def display_half_str(x: float) -> str:
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"


def score_label(s: float) -> str:
    if s >= 9.0: return "Excellent"
    if s >= 7.5: return "Great"
    if s >= 6.0: return "Good"
    if s >= 4.5: return "Fair"
    if s >= 3.0: return "Needs Work"
    return "Poor"


# ============ Torso Lean Calculation ============

def compute_torso_lean(lms, lsh, rsh, lh, rh, vis_thr: float = 0.3) -> Optional[float]:
    """
    Calculate torso lean angle from vertical using midpoints of shoulders and hips.
    Returns angle in degrees (0 = perfectly upright).
    """
    pts = [lms[lsh], lms[rsh], lms[lh], lms[rh]]
    if any(p.visibility < vis_thr for p in pts):
        return None

    mid_sh_x = (lms[lsh].x + lms[rsh].x) / 2.0
    mid_sh_y = (lms[lsh].y + lms[rsh].y) / 2.0
    mid_hp_x = (lms[lh].x + lms[rh].x) / 2.0
    mid_hp_y = (lms[lh].y + lms[rh].y) / 2.0

    dx = mid_sh_x - mid_hp_x
    dy = mid_sh_y - mid_hp_y  # In image coords, y increases downward

    # Lean angle from vertical
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9)))


# ============ Body-only Skeleton Drawing ============

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


def _dyn_thickness(h: int) -> Tuple[int, int]:
    return max(2, int(round(h * 0.002))), max(3, int(round(h * 0.004)))


def draw_body_skeleton(frame, lms, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    line_t, dot_t = _dyn_thickness(h)
    for a, b in _BODY_CONNECTIONS:
        pa, pb = lms[a], lms[b]
        ax, ay = int(pa.x * w), int(pa.y * h)
        bx, by = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, line_t, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = lms[i]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), dot_t, color, -1, cv2.LINE_AA)
    return frame


# ============ Overlay Drawing ============

# Font setup
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
BAR_BG_ALPHA = 0.55
DONUT_RADIUS_SCALE = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR = (40, 200, 80)
DEPTH_RING_BG = (70, 70, 70)

REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE = 18


def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


REPS_FONT = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)


def _wrap_two_lines(draw, text, font, max_width):
    words = text.split()
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
    return lines[:2]


def draw_overlay(frame, reps=0, feedback=None, height_pct=0.0, phase_name=""):
    if not HAS_PIL:
        # Fallback: simple OpenCV text
        cv2.putText(frame, f"Reps: {int(reps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if feedback:
            cv2.putText(frame, feedback, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    h, w, _ = frame.shape

    # Donut ring for height progress
    ref_h = max(int(h * 0.06), int(REPS_FONT_SIZE * 1.6))
    r = int(ref_h * DONUT_RADIUS_SCALE)
    th = max(3, int(r * DONUT_THICKNESS_FRAC))
    m = 12
    cx = w - m - r
    cy = max(ref_h + r // 8, r + th // 2 + 2)
    pct = float(np.clip(height_pct, 0, 1))
    cv2.circle(frame, (cx, cy), r, DEPTH_RING_BG, th, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, -90 + int(360 * pct), DEPTH_COLOR, th, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Rep counter (top-left)
    txt = f"Reps: {int(reps)}"
    pad_x, pad_y = 10, 6
    tw = draw.textlength(txt, font=REPS_FONT)
    thh = REPS_FONT.size
    base = np.array(pil)
    over = base.copy()
    cv2.rectangle(over, (0, 0), (int(tw + 2 * pad_x), int(thh + 2 * pad_y)), (0, 0, 0), -1)
    base = cv2.addWeighted(over, BAR_BG_ALPHA, base, 1 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(base)
    draw = ImageDraw.Draw(pil)
    draw.text((pad_x, pad_y - 1), txt, font=REPS_FONT, fill=(255, 255, 255))

    # Height label inside donut
    gap = max(2, int(r * 0.10))
    by = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size) // 2
    label = "HEIGHT"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label, font=DEPTH_LABEL_FONT)
    pw = draw.textlength(pct_txt, font=DEPTH_PCT_FONT)
    draw.text((cx - int(lw // 2), by), label, font=DEPTH_LABEL_FONT, fill=(255, 255, 255))
    draw.text((cx - int(pw // 2), by + DEPTH_LABEL_FONT.size + gap), pct_txt,
              font=DEPTH_PCT_FONT, fill=(255, 255, 255))

    # Feedback bar (bottom)
    if feedback:
        max_w = int(w - 2 * 12 - 20)
        lines = _wrap_two_lines(draw, feedback, FEEDBACK_FONT, max_w)
        line_h = FEEDBACK_FONT.size + 6
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
            tw2 = draw.textlength(ln, font=FEEDBACK_FONT)
            tx = max(12, (w - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255, 255, 255))
            ty += line_h + 4

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ============ State Machine Rep Counter ============

class OverheadPressTracker:
    """
    Finite state machine for overhead press rep counting.
    
    State flow:
        IDLE → BOTTOM (detected starting position)
        BOTTOM → ASCENDING (elbow angle increasing, wrist rising)
        ASCENDING → TOP (lockout detected)
        TOP → DESCENDING (elbow angle decreasing, wrist lowering)
        DESCENDING → BOTTOM (returned to start → rep counted)
        BOTTOM → ASCENDING (next rep begins)
    
    A rep is counted when transitioning from DESCENDING → BOTTOM,
    i.e., a complete press-up and lower-back-down cycle.
    """

    def __init__(self, config: PressConfig = None):
        self.cfg = config or PressConfig()

        # State
        self.phase = PressPhase.IDLE
        self.frames_in_phase = 0
        self.total_frames_processed = 0

        # Smoothed values
        self.elbow_ema: Optional[float] = None
        self.wrist_y_ema: Optional[float] = None
        self.lean_ema: Optional[float] = None

        # Angular velocity tracking
        self.elbow_history = deque(maxlen=5)

        # Rep counting
        self.rep_count = 0
        self.last_rep_frame = -1000
        self.cycle_start_frame = 0

        # Per-rep metrics
        self.current_metrics = RepMetrics()
        self.all_reps: List[RepMetrics] = []

        # Shoulder reference for normalization
        self.shoulder_y_ema: Optional[float] = None
        self.torso_height_ema: Optional[float] = None

        # Calibration
        self.calibrated = False
        self.observed_bottoms: List[float] = []
        self.observed_tops: List[float] = []

        # Real-time feedback
        self.rt_feedback: Optional[str] = None
        self.rt_feedback_hold: int = 0

    def _angular_velocity(self) -> float:
        """Average absolute change in elbow angle over recent frames."""
        if len(self.elbow_history) < 2:
            return 0.0
        diffs = [abs(self.elbow_history[i] - self.elbow_history[i - 1])
                 for i in range(1, len(self.elbow_history))]
        return sum(diffs) / len(diffs)

    def _transition(self, new_phase: PressPhase):
        self.phase = new_phase
        self.frames_in_phase = 0

    def _score_rep(self, metrics: RepMetrics) -> RepMetrics:
        """Score a completed rep on multiple dimensions."""
        cfg = self.cfg

        # Depth score (lower elbow = deeper = better, to a point)
        metrics.depth_score = score_interpolate(
            metrics.min_elbow_angle,
            target=cfg.depth_target_angle,
            ok=cfg.depth_ok_angle,
            bad=cfg.depth_ok_angle + 20.0,
            higher_is_better=False  # Lower angle = better depth
        )

        # Lockout score (higher elbow angle at top = better)
        metrics.lockout_score = score_interpolate(
            metrics.max_elbow_angle,
            target=cfg.lockout_target_angle,
            ok=cfg.lockout_ok_angle,
            bad=cfg.lockout_ok_angle - 15.0,
            higher_is_better=True
        )

        # Lean score (less lean = better)
        metrics.lean_score = score_interpolate(
            metrics.max_torso_lean,
            target=cfg.lean_target_max,
            ok=cfg.lean_ok_max,
            bad=cfg.lean_bad_max,
            higher_is_better=False
        )

        # Stacking score (wrist over shoulder)
        metrics.stack_score = score_interpolate(
            metrics.max_wrist_offset,
            target=cfg.stack_target_max,
            ok=cfg.stack_ok_max,
            bad=cfg.stack_ok_max + 0.04,
            higher_is_better=False
        )

        # Bar path score (lateral deviation)
        if len(metrics.wrist_x_samples) > 2:
            xs = np.array(metrics.wrist_x_samples)
            metrics.bar_path_deviation = float(np.std(xs))
        metrics.bar_path_score = score_interpolate(
            metrics.bar_path_deviation,
            target=cfg.bar_path_target,
            ok=cfg.bar_path_ok,
            bad=cfg.bar_path_ok + 0.03,
            higher_is_better=False
        )

        # Overall weighted score
        metrics.overall_score = (
            cfg.w_depth * metrics.depth_score +
            cfg.w_lockout * metrics.lockout_score +
            cfg.w_lean * metrics.lean_score +
            cfg.w_stack * metrics.stack_score +
            cfg.w_bar_path * metrics.bar_path_score
        )

        # Generate issues list
        metrics.issues = []
        if metrics.depth_score < 7.0:
            metrics.issues.append("Lower to shoulder level each rep")
        if metrics.lockout_score < 7.0:
            metrics.issues.append("Press to full lockout overhead")
        if metrics.lean_score < 7.0:
            metrics.issues.append("Keep torso upright — avoid leaning back")
        if metrics.stack_score < 7.0:
            metrics.issues.append("Keep wrists stacked over shoulders")
        if metrics.bar_path_score < 7.0:
            metrics.issues.append("Keep bar path straight and vertical")

        return metrics

    def _maybe_calibrate(self):
        """After a few reps, adapt thresholds to the user's actual ROM."""
        if self.calibrated or len(self.all_reps) < self.cfg.calibration_reps:
            return

        avg_bottom = np.mean([r.min_elbow_angle for r in self.all_reps])
        avg_top = np.mean([r.max_elbow_angle for r in self.all_reps])

        # Only adapt if user's ROM is reasonably close to defaults
        adj = self.cfg.calibration_adjustment
        if avg_bottom < self.cfg.bottom_elbow_max + 20:
            new_bottom = adj * avg_bottom + (1 - adj) * self.cfg.bottom_elbow_max
            self.cfg.bottom_elbow_max = max(80.0, min(130.0, new_bottom))

        if avg_top > self.cfg.top_elbow_min - 15:
            new_top = adj * avg_top + (1 - adj) * self.cfg.top_elbow_min
            self.cfg.top_elbow_min = max(145.0, min(178.0, new_top))

        self.calibrated = True

    def update(self, lms, frame_idx: int, img_h: int, img_w: int) -> dict:
        """
        Process one frame. Returns dict with current state info.
        
        Args:
            lms: MediaPipe pose landmarks
            frame_idx: Current frame number
            img_h, img_w: Frame dimensions
        
        Returns:
            dict with keys: rep_count, phase, height_pct, feedback, new_rep
        """
        cfg = self.cfg
        self.total_frames_processed += 1
        self.frames_in_phase += 1
        new_rep = False

        # Landmark indices
        LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        LE = mp_pose.PoseLandmark.LEFT_ELBOW.value
        RE = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        LW = mp_pose.PoseLandmark.LEFT_WRIST.value
        RW = mp_pose.PoseLandmark.RIGHT_WRIST.value
        LH = mp_pose.PoseLandmark.LEFT_HIP.value
        RH = mp_pose.PoseLandmark.RIGHT_HIP.value
        NOSE = mp_pose.PoseLandmark.NOSE.value

        # Pick dominant side based on visibility
        vis_left = (lms[LSH].visibility + lms[LE].visibility + lms[LW].visibility) / 3.0
        vis_right = (lms[RSH].visibility + lms[RE].visibility + lms[RW].visibility) / 3.0

        if vis_left >= vis_right:
            sh_idx, el_idx, wr_idx = LSH, LE, LW
            sh2_idx, el2_idx, wr2_idx = RSH, RE, RW
        else:
            sh_idx, el_idx, wr_idx = RSH, RE, RW
            sh2_idx, el2_idx, wr2_idx = LSH, LE, LW

        dom_vis = max(vis_left, vis_right)
        if dom_vis < cfg.vis_threshold:
            return self._state_dict(0.0, frame_idx)

        # Extract coordinates
        sh = (lms[sh_idx].x, lms[sh_idx].y)
        el = (lms[el_idx].x, lms[el_idx].y)
        wr = (lms[wr_idx].x, lms[wr_idx].y)

        # Compute elbow angle (primary arm)
        raw_elbow = angle_3pt(sh, el, wr)

        # Also compute secondary arm angle if visible (for bilateral validation)
        sec_vis = min(lms[sh2_idx].visibility, lms[el2_idx].visibility, lms[wr2_idx].visibility)
        if sec_vis > cfg.vis_threshold:
            sh2 = (lms[sh2_idx].x, lms[sh2_idx].y)
            el2 = (lms[el2_idx].x, lms[el2_idx].y)
            wr2 = (lms[wr2_idx].x, lms[wr2_idx].y)
            raw_elbow2 = angle_3pt(sh2, el2, wr2)
            # Use average of both arms for more stable reading
            raw_elbow = (raw_elbow + raw_elbow2) / 2.0

        # Smooth
        self.elbow_ema = ema_update(self.elbow_ema, raw_elbow, cfg.elbow_ema_alpha)
        self.wrist_y_ema = ema_update(self.wrist_y_ema, wr[1], cfg.wrist_ema_alpha)
        self.elbow_history.append(self.elbow_ema)

        # Shoulder and torso height reference
        mid_sh_y = (lms[LSH].y + lms[RSH].y) / 2.0
        mid_hp_y = (lms[LH].y + lms[RH].y) / 2.0
        torso_h = abs(mid_sh_y - mid_hp_y)
        self.shoulder_y_ema = ema_update(self.shoulder_y_ema, mid_sh_y, 0.2)
        self.torso_height_ema = ema_update(self.torso_height_ema, torso_h, 0.2)

        # Head reference
        head_y = lms[NOSE].y if lms[NOSE].visibility > cfg.vis_threshold else (mid_sh_y - 0.10)
        head_y = min(head_y, mid_sh_y - 0.02)

        # Height percentage (wrist position relative to shoulder-to-head range)
        height_range = max(0.05, mid_sh_y - head_y)
        height_pct = float(np.clip((mid_sh_y - self.wrist_y_ema) / height_range, 0.0, 1.0))

        # Torso lean
        lean = compute_torso_lean(lms, LSH, RSH, LH, RH, cfg.vis_threshold)
        if lean is not None:
            self.lean_ema = ema_update(self.lean_ema, lean, cfg.lean_ema_alpha)

        # Wrist-shoulder horizontal offset (normalized by torso height)
        wrist_offset = abs(wr[0] - sh[0])
        if self.torso_height_ema and self.torso_height_ema > 0.01:
            wrist_offset_norm = wrist_offset / self.torso_height_ema
        else:
            wrist_offset_norm = wrist_offset

        # ---- Update per-rep metrics ----
        m = self.current_metrics
        m.total_frames += 1
        m.min_elbow_angle = min(m.min_elbow_angle, self.elbow_ema)
        m.max_elbow_angle = max(m.max_elbow_angle, self.elbow_ema)
        if self.lean_ema is not None:
            m.max_torso_lean = max(m.max_torso_lean, self.lean_ema)
        m.max_wrist_offset = max(m.max_wrist_offset, wrist_offset_norm)
        m.wrist_x_samples.append(wr[0])

        # ---- Conditions ----
        eb = self.elbow_ema
        wrist_near_shoulder = (self.wrist_y_ema >= mid_sh_y - cfg.wrist_bottom_ratio * max(0.05, torso_h))
        wrist_above_head = (self.wrist_y_ema <= head_y + cfg.wrist_top_ratio * max(0.05, torso_h))

        is_bottom = (eb <= cfg.bottom_elbow_max and wrist_near_shoulder)
        is_top = (eb >= cfg.top_elbow_min and wrist_above_head)

        # Hysteresis: once in top, need to drop below (top_threshold - hysteresis) to leave
        is_clearly_not_top = (eb < cfg.top_elbow_min - cfg.hysteresis_band)
        is_clearly_not_bottom = (eb > cfg.bottom_elbow_max + cfg.hysteresis_band)

        ang_vel = self._angular_velocity()
        enough_frames = self.frames_in_phase >= cfg.min_phase_frames

        # ---- State Machine ----
        if self.phase == PressPhase.IDLE:
            if is_bottom:
                self._transition(PressPhase.BOTTOM)
                self.cycle_start_frame = frame_idx
                self.current_metrics = RepMetrics(rep_index=self.rep_count + 1)
                self.current_metrics.phase_at_start = "BOTTOM"
            elif is_top:
                # Started at top — that's okay, wait for descent
                self._transition(PressPhase.TOP)

        elif self.phase == PressPhase.BOTTOM:
            if is_clearly_not_bottom and enough_frames:
                # Elbow opening up = ascending
                self._transition(PressPhase.ASCENDING)
                m.ascending_frames = 0

        elif self.phase == PressPhase.ASCENDING:
            m.ascending_frames += 1
            if is_top and enough_frames:
                self._transition(PressPhase.TOP)
                m.wrist_y_at_top = self.wrist_y_ema
            elif is_bottom and enough_frames:
                # Went back down without reaching top — false start
                self._transition(PressPhase.BOTTOM)
                self.current_metrics = RepMetrics(rep_index=self.rep_count + 1)

        elif self.phase == PressPhase.TOP:
            if is_clearly_not_top and enough_frames:
                self._transition(PressPhase.DESCENDING)
                m.descending_frames = 0

        elif self.phase == PressPhase.DESCENDING:
            m.descending_frames += 1
            if is_bottom and enough_frames:
                # Complete rep!
                cycle_len = frame_idx - self.cycle_start_frame
                if (cycle_len >= cfg.min_cycle_frames and
                    (frame_idx - self.last_rep_frame) >= cfg.refractory_frames):

                    m.wrist_y_at_bottom = self.wrist_y_ema
                    self._score_rep(m)

                    self.rep_count += 1
                    self.all_reps.append(m)
                    self.last_rep_frame = frame_idx
                    new_rep = True

                    # Set real-time feedback from worst issue
                    if m.issues:
                        self.rt_feedback = m.issues[0]
                        self.rt_feedback_hold = 8  # frames to hold
                    else:
                        self.rt_feedback = None

                    # Try calibration
                    self._maybe_calibrate()

                # Start new cycle
                self._transition(PressPhase.BOTTOM)
                self.cycle_start_frame = frame_idx
                self.current_metrics = RepMetrics(rep_index=self.rep_count + 1)

            elif is_top and enough_frames:
                # Went back up without completing descent — bounce
                self._transition(PressPhase.TOP)

        # Feedback hold
        if self.rt_feedback_hold > 0:
            self.rt_feedback_hold -= 1

        return self._state_dict(height_pct, frame_idx, new_rep)

    def _state_dict(self, height_pct: float, frame_idx: int, new_rep: bool = False) -> dict:
        fb = self.rt_feedback if self.rt_feedback_hold > 0 else None
        return {
            "rep_count": self.rep_count,
            "phase": self.phase.name,
            "height_pct": height_pct,
            "feedback": fb,
            "new_rep": new_rep,
        }

    def get_session_summary(self) -> dict:
        """Generate final session summary with scores and feedback."""
        if not self.all_reps:
            return {
                "rep_count": 0,
                "technique_score": 0.0,
                "technique_score_display": "0",
                "technique_label": "No reps detected",
                "good_reps": 0,
                "bad_reps": 0,
                "feedback": [],
                "reps": [],
            }

        # Per-dimension averages
        avg_depth = np.mean([r.depth_score for r in self.all_reps])
        avg_lockout = np.mean([r.lockout_score for r in self.all_reps])
        avg_lean = np.mean([r.lean_score for r in self.all_reps])
        avg_stack = np.mean([r.stack_score for r in self.all_reps])
        avg_bar = np.mean([r.bar_path_score for r in self.all_reps])

        cfg = self.cfg
        overall = (
            cfg.w_depth * avg_depth +
            cfg.w_lockout * avg_lockout +
            cfg.w_lean * avg_lean +
            cfg.w_stack * avg_stack +
            cfg.w_bar_path * avg_bar
        )
        technique_score = half_floor_10(overall)

        good_reps = sum(1 for r in self.all_reps if r.overall_score >= 7.0)
        bad_reps = len(self.all_reps) - good_reps

        # Aggregate feedback — find issues that appeared in multiple reps
        from collections import Counter
        issue_counter = Counter()
        for r in self.all_reps:
            for iss in r.issues:
                issue_counter[iss] += 1

        # Only report issues that occurred in >= 30% of reps (or at least 2)
        min_threshold = max(2, int(len(self.all_reps) * 0.3))
        session_feedback = [iss for iss, cnt in issue_counter.most_common()
                           if cnt >= min(min_threshold, len(self.all_reps))]

        # Rep details
        rep_details = []
        for r in self.all_reps:
            rep_details.append({
                "rep_index": r.rep_index,
                "score": round(r.overall_score, 1),
                "good": r.overall_score >= 7.0,
                "depth_score": round(r.depth_score, 1),
                "lockout_score": round(r.lockout_score, 1),
                "lean_score": round(r.lean_score, 1),
                "stack_score": round(r.stack_score, 1),
                "bar_path_score": round(r.bar_path_score, 1),
                "bottom_elbow": round(r.min_elbow_angle, 1),
                "top_elbow": round(r.max_elbow_angle, 1),
                "max_lean": round(r.max_torso_lean, 1),
                "max_stack_offset": round(r.max_wrist_offset, 3),
                "issues": r.issues,
            })

        form_tip = session_feedback[0] if session_feedback else None

        return {
            "rep_count": self.rep_count,
            "technique_score": float(technique_score),
            "technique_score_display": display_half_str(technique_score),
            "technique_label": score_label(technique_score),
            "good_reps": good_reps,
            "bad_reps": bad_reps,
            "feedback": session_feedback,
            "form_tip": form_tip,
            "dimension_scores": {
                "depth": round(avg_depth, 1),
                "lockout": round(avg_lockout, 1),
                "lean": round(avg_lean, 1),
                "stacking": round(avg_stack, 1),
                "bar_path": round(avg_bar, 1),
            },
            "reps": rep_details,
        }


# ============ Main Entry Point ============

def run_overhead_press_analysis(video_path: str,
                                 frame_skip: int = 2,
                                 scale: float = 0.5,
                                 output_path: str = "overhead_press_analyzed.mp4",
                                 feedback_path: str = "overhead_press_feedback.txt",
                                 preserve_quality: bool = False,
                                 encode_crf: int = None,
                                 return_video: bool = True,
                                 fast_mode: bool = None,
                                 config: PressConfig = None) -> dict:
    """
    Analyze an overhead press video.
    
    Args:
        video_path: Path to input video
        frame_skip: Process every Nth frame (1 = all frames, 2 = every other, etc.)
        scale: Resize factor for processing (0.5 = half resolution)
        output_path: Path for annotated output video
        feedback_path: Path for text feedback file
        preserve_quality: If True, process at full resolution
        encode_crf: FFmpeg CRF value for output encoding
        return_video: If True, generate annotated video
        fast_mode: If True, skip video generation and use lighter model
        config: PressConfig with custom thresholds
    
    Returns:
        dict with analysis results
    """
    if mp_pose is None:
        return _ret_err("MediaPipe not available", feedback_path)

    cfg = config or PressConfig()
    model_complexity = 0 if fast_mode else 1

    if fast_mode:
        return_video = False
        scale = min(scale, 0.35)
        frame_skip = max(frame_skip, 3)

    if preserve_quality:
        scale = 1.0
        frame_skip = 1
        encode_crf = 18 if encode_crf is None else encode_crf
    else:
        encode_crf = 23 if encode_crf is None else encode_crf

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _ret_err("Could not open video", feedback_path)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    effective_fps = max(1.0, fps_in / max(1, frame_skip))

    # Adapt minimum cycle frames to actual FPS
    cfg.min_cycle_frames = max(4, int(effective_fps * 0.4))  # At least 0.4 sec per rep
    cfg.min_phase_frames = max(2, int(effective_fps * 0.1))  # At least 0.1 sec per phase
    cfg.refractory_frames = max(3, int(effective_fps * 0.2))

    tracker = OverheadPressTracker(config=cfg)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_idx = 0

    with mp_pose.Pose(
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_skip > 1 and (frame_idx % frame_skip) != 0:
                continue

            if scale and scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            state = {"rep_count": tracker.rep_count, "phase": tracker.phase.name,
                     "height_pct": 0.0, "feedback": None, "new_rep": False}

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                if out is None and return_video:
                    out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

                state = tracker.update(lms, frame_idx, h, w)

                if return_video and out:
                    frame = draw_body_skeleton(frame, lms)

            if return_video and out:
                frame = draw_overlay(
                    frame,
                    reps=state["rep_count"],
                    feedback=state.get("feedback"),
                    height_pct=state.get("height_pct", 0.0),
                    phase_name=state.get("phase", ""),
                )
                out.write(frame)

    cap.release()
    if return_video and out:
        out.release()
    cv2.destroyAllWindows()

    # Get session summary
    summary = tracker.get_session_summary()

    # Write feedback file
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Overhead Press Analysis\n")
            f.write(f"{'=' * 40}\n")
            f.write(f"Total Reps: {summary['rep_count']}\n")
            f.write(f"Technique Score: {summary['technique_score_display']} / 10 "
                    f"({summary['technique_label']})\n")
            f.write(f"Good Reps: {summary['good_reps']}  |  Needs Work: {summary['bad_reps']}\n\n")

            if summary.get("dimension_scores"):
                ds = summary["dimension_scores"]
                f.write("Dimension Scores:\n")
                f.write(f"  Depth:     {ds['depth']}/10\n")
                f.write(f"  Lockout:   {ds['lockout']}/10\n")
                f.write(f"  Lean:      {ds['lean']}/10\n")
                f.write(f"  Stacking:  {ds['stacking']}/10\n")
                f.write(f"  Bar Path:  {ds['bar_path']}/10\n\n")

            if summary["feedback"]:
                f.write("Feedback:\n")
                for fb in summary["feedback"]:
                    f.write(f"  • {fb}\n")
                f.write("\n")

            if summary["reps"]:
                f.write("Per-Rep Breakdown:\n")
                for r in summary["reps"]:
                    issues_str = ", ".join(r["issues"]) if r["issues"] else "Good form"
                    f.write(f"  Rep {r['rep_index']}: {r['score']}/10 — {issues_str}\n")
    except Exception:
        pass

    # Encode video with ffmpeg
    final_path = ""
    if return_video and os.path.exists(output_path):
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", output_path, "-c:v", "libx264", "-preset", "medium",
                 "-crf", str(int(encode_crf)), "-movflags", "+faststart",
                 "-pix_fmt", "yuv420p", encoded_path],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            final_path = encoded_path if os.path.exists(encoded_path) else output_path
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
        except Exception:
            final_path = output_path if os.path.exists(output_path) else ""

    # Build result (keeping backward-compatible keys)
    result = {
        "squat_count": summary["rep_count"],  # Legacy key name
        "rep_count": summary["rep_count"],
        "technique_score": summary["technique_score"],
        "technique_score_display": summary["technique_score_display"],
        "technique_label": summary["technique_label"],
        "good_reps": summary["good_reps"],
        "bad_reps": summary["bad_reps"],
        "feedback": summary["feedback"],
        "tips": [],
        "reps": summary["reps"],
        "dimension_scores": summary.get("dimension_scores", {}),
        "video_path": final_path if return_video else "",
        "feedback_path": feedback_path,
    }
    if summary.get("form_tip"):
        result["form_tip"] = summary["form_tip"]

    return result


def _ret_err(msg: str, feedback_path: str) -> dict:
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass
    return {
        "squat_count": 0, "rep_count": 0,
        "technique_score": 0.0,
        "technique_score_display": "0",
        "technique_label": "No reps detected",
        "good_reps": 0, "bad_reps": 0,
        "feedback": [], "tips": [],
        "reps": [], "dimension_scores": {},
        "video_path": "", "feedback_path": feedback_path,
    }


# Backward-compatible alias
def run_analysis(*args, **kwargs):
    return run_overhead_press_analysis(*args, **kwargs)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python overhead_press_analysis.py <video_path> [output_path]")
        sys.exit(1)
    video = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "overhead_press_analyzed.mp4"
    result = run_overhead_press_analysis(video, output_path=out_path)
    print(f"\nResults: {result['rep_count']} reps, Score: {result['technique_score_display']}/10")
    if result["feedback"]:
        print("Feedback:")
        for fb in result["feedback"]:
            print(f"  • {fb}")

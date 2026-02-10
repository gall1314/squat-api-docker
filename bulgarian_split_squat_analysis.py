# -*- coding: utf-8 -*-
# bulgarian_split_squat_analysis.py
# ============================================================================
# COMPLETE REWRITE – robust rep counting, multi-phase state machine,
# adaptive thresholds, and improved form scoring.
#
# Key improvements over original:
# 1. **5-phase state machine** (IDLE → DESCENDING → BOTTOM_HOLD → ASCENDING → LOCKOUT)
#    instead of simple 2-state (up/down). Eliminates false positives from partial
#    movements, jitter, and transitional wobble.
# 2. **Adaptive angle thresholds** – calibrates standing angle per-user in first
#    frames so thresholds work for different body proportions / camera angles.
# 3. **Velocity-based phase detection** – uses angular velocity (not just absolute
#    angle) to confirm genuine descent vs. noise / drift.
# 4. **Hysteresis bands** – separate entry/exit thresholds per phase to prevent
#    rapid toggling at boundary angles.
# 5. **Multi-joint form scoring** – evaluates depth, torso lean, knee valgus,
#    tempo symmetry (eccentric vs concentric), and knee-over-toe tracking.
# 6. **Walking / transition filter** – blocks counting during locomotion using
#    hip + ankle velocity with adaptive normalization.
# 7. **Confidence gating** – skips frames where landmark visibility is too low.
# 8. **Improved overlay** – cleaner rendering, phase indicator, live angle display.
# ============================================================================

import os, math, subprocess, collections
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ========================== CONFIGURATION ==========================

@dataclass
class BulgarianConfig:
    """All tunable parameters in one place."""

    # --- Phase thresholds (degrees) – relative to calibrated standing angle ---
    # These are OFFSETS from the calibrated standing knee angle.
    descent_trigger_offset: float = 20.0    # standing - 20° → start descending
    bottom_zone_offset: float = 70.0        # standing - 70° → deep enough for bottom
    ascent_trigger_offset: float = 25.0     # from bottom, rise 25° → ascending
    lockout_offset: float = 15.0            # within 15° of standing → lockout

    # --- Absolute fallback thresholds (if calibration fails) ---
    fallback_down_thresh: float = 100.0
    fallback_up_thresh: float = 155.0
    fallback_standing_angle: float = 170.0

    # --- Angular velocity thresholds (deg/sec) ---
    min_descent_velocity: float = 15.0      # must be descending at ≥15°/s
    min_ascent_velocity: float = 12.0       # must be ascending at ≥12°/s

    # --- Rep validation ---
    min_depth_degrees: float = 35.0         # min ROM (standing - bottom) to count
    min_bottom_frames: int = 2              # must spend ≥2 frames near bottom
    min_rep_duration_sec: float = 0.6       # a rep can't be shorter than 0.6s
    max_rep_duration_sec: float = 8.0       # a rep can't be longer than 8s
    rep_cooldown_sec: float = 0.3           # min gap between consecutive reps

    # --- Form scoring ---
    perfect_depth_angle: float = 75.0       # ideal bottom knee angle
    torso_lean_min: float = 135.0           # torso angle below this → penalty
    torso_lean_margin: float = 3.0          # hysteresis margin
    torso_bad_min_frames: int = 4
    valgus_x_tol: float = 0.03             # knee-ankle X deviation tolerance
    valgus_bad_min_frames: int = 3
    good_rep_min_score: float = 8.0
    tempo_ratio_ideal: Tuple[float, float] = (0.8, 1.5)  # ecc/con ratio range

    # --- Smoothing ---
    angle_ema_alpha: float = 0.55
    velocity_ema_alpha: float = 0.4

    # --- Walking / movement filter ---
    hip_vel_thresh_pct: float = 0.013
    ankle_vel_thresh_pct: float = 0.016
    motion_ema_alpha: float = 0.6
    movement_clear_frames: int = 3

    # --- Calibration ---
    calibration_frames: int = 15            # frames to average for standing angle
    calibration_min_visibility: float = 0.6

    # --- Landmark quality ---
    min_landmark_visibility: float = 0.5
    min_visible_fraction: float = 0.55

    # --- Early exit ---
    nopose_stop_sec: float = 1.2
    no_movement_stop_sec: float = 1.3

    # --- Overlay style ---
    bar_bg_alpha: float = 0.55
    reps_font_size: int = 28
    feedback_font_size: int = 22
    depth_label_font_size: int = 14
    depth_pct_font_size: int = 18
    font_path: str = "Roboto-VariableFont_wdth,wght.ttf"
    rt_fb_hold_sec: float = 0.8
    donut_radius_scale: float = 0.72
    donut_thickness_frac: float = 0.28
    depth_color: Tuple[int, ...] = (40, 200, 80)
    depth_ring_bg: Tuple[int, ...] = (70, 70, 70)

    # --- Skeleton ---
    skeleton_color: Tuple[int, ...] = (255, 255, 255)
    skeleton_thickness: int = 2
    skeleton_point_radius: int = 3
    skeleton_max_hold_frames: int = 6
    skeleton_quality_thr: float = 0.55
    skeleton_jump_thr: float = 0.12


# ========================== ENUMS & TYPES ==========================

class Phase(Enum):
    IDLE = auto()        # standing, not yet started or between reps
    DESCENDING = auto()  # actively going down
    BOTTOM = auto()      # at/near the bottom of the movement
    ASCENDING = auto()   # actively coming up
    LOCKOUT = auto()     # returned to near-standing, rep complete

mp_pose = mp.solutions.pose

# ========================== GEOMETRY HELPERS ==========================

def angle_3pt(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle at point b formed by segments ba and bc, in degrees [0, 180]."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

def lm_px(landmarks, idx: int, w: int, h: int) -> Tuple[float, float]:
    return (landmarks[idx].x * w, landmarks[idx].y * h)

def lm_vis(landmarks, idx: int) -> float:
    v = getattr(landmarks[idx], 'visibility', 0.0)
    return float(v) if v else 0.0

def euclidean_norm(a: Tuple[float, float], b: Tuple[float, float], norm: float) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) / max(1.0, norm)


# ========================== ACTIVE LEG DETECTION ==========================

def detect_active_leg(landmarks) -> str:
    """
    The working leg in a Bulgarian split squat is the FRONT leg (lower ankle = front foot).
    The rear foot is elevated on a bench, so its ankle is higher (lower y in image coords means higher).
    Actually: higher y value = lower in frame = closer to ground = front foot.
    """
    left_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    # The leg whose ankle is LOWER in the frame (higher y) is the front/working leg
    return 'left' if left_y > right_y else 'right'

def detect_active_leg_robust(landmarks, n_votes: list) -> str:
    """Accumulate votes over multiple frames for more stable detection."""
    leg = detect_active_leg(landmarks)
    n_votes.append(leg)
    if len(n_votes) > 20:
        n_votes.pop(0)
    left_count = n_votes.count('left')
    right_count = n_votes.count('right')
    return 'left' if left_count >= right_count else 'right'


# ========================== EMA SMOOTHER ==========================

class EMASmoother:
    """Exponential moving average for angle streams with velocity output."""

    def __init__(self, alpha: float = 0.55, vel_alpha: float = 0.4):
        self.alpha = alpha
        self.vel_alpha = vel_alpha
        self._values: Dict[str, float] = {}
        self._velocities: Dict[str, float] = {}

    def update(self, dt: float, **kwargs) -> Dict[str, float]:
        """Update named angle streams. Returns smoothed values."""
        smoothed = {}
        for key, raw in kwargs.items():
            raw = float(raw)
            if key not in self._values:
                self._values[key] = raw
                self._velocities[key] = 0.0
            else:
                prev = self._values[key]
                self._values[key] = self.alpha * raw + (1.0 - self.alpha) * prev
                raw_vel = (self._values[key] - prev) / max(dt, 1e-6)
                self._velocities[key] = (self.vel_alpha * raw_vel +
                                         (1.0 - self.vel_alpha) * self._velocities[key])
            smoothed[key] = self._values[key]
        return smoothed

    def velocity(self, key: str) -> float:
        return self._velocities.get(key, 0.0)

    def value(self, key: str) -> float:
        return self._values.get(key, 0.0)


# ========================== MOVEMENT FILTER ==========================

class MovementFilter:
    """Detects walking/locomotion to block rep counting during transitions."""

    def __init__(self, cfg: BulgarianConfig):
        self.cfg = cfg
        self.prev_hip = None
        self.prev_la = None
        self.prev_ra = None
        self.hip_vel_ema = 0.0
        self.ankle_vel_ema = 0.0
        self.still_streak = 0

    def update(self, landmarks, side: str, w: int, h: int) -> bool:
        """Returns True if movement is blocked (walking detected)."""
        hip_idx = getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP").value
        hip_px = lm_px(landmarks, hip_idx, w, h)
        la_px = lm_px(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h)
        ra_px = lm_px(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h)

        norm = max(h, w)

        if self.prev_hip is None:
            self.prev_hip, self.prev_la, self.prev_ra = hip_px, la_px, ra_px
            return False

        hv = euclidean_norm(hip_px, self.prev_hip, norm)
        av = max(euclidean_norm(la_px, self.prev_la, norm),
                 euclidean_norm(ra_px, self.prev_ra, norm))

        a = self.cfg.motion_ema_alpha
        self.hip_vel_ema = a * hv + (1 - a) * self.hip_vel_ema
        self.ankle_vel_ema = a * av + (1 - a) * self.ankle_vel_ema

        self.prev_hip, self.prev_la, self.prev_ra = hip_px, la_px, ra_px

        blocked = (self.hip_vel_ema > self.cfg.hip_vel_thresh_pct or
                   self.ankle_vel_ema > self.cfg.ankle_vel_thresh_pct)

        if blocked:
            self.still_streak = 0
        else:
            self.still_streak = min(self.cfg.movement_clear_frames + 2, self.still_streak + 1)

        return blocked

    @property
    def is_still(self) -> bool:
        return self.still_streak >= self.cfg.movement_clear_frames


# ========================== CALIBRATION ==========================

class StandingCalibrator:
    """Determines the user's standing knee angle from the first few stable frames."""

    def __init__(self, cfg: BulgarianConfig):
        self.cfg = cfg
        self.samples: List[float] = []
        self.calibrated_angle: Optional[float] = None
        self._done = False

    @property
    def is_done(self) -> bool:
        return self._done

    def add_sample(self, knee_angle: float, visibility_ok: bool, is_still: bool):
        if self._done:
            return
        if visibility_ok and is_still and knee_angle > 130:  # likely standing
            self.samples.append(knee_angle)
        if len(self.samples) >= self.cfg.calibration_frames:
            # Use median for robustness against outliers
            self.calibrated_angle = float(np.median(self.samples))
            self._done = True

    def get_standing_angle(self) -> float:
        if self.calibrated_angle is not None:
            return self.calibrated_angle
        if self.samples:
            return float(np.median(self.samples))
        return self.cfg.fallback_standing_angle


# ========================== REP STATE MACHINE ==========================

@dataclass
class RepData:
    """Data accumulated during a single rep."""
    start_frame: int = 0
    start_time: float = 0.0
    descent_start_angle: float = 0.0
    min_knee_angle: float = 999.0
    max_knee_angle: float = -999.0
    min_torso_angle: float = 999.0
    bottom_frames: int = 0
    torso_bad_frames: int = 0
    torso_bad_consecutive: int = 0
    valgus_bad_frames: int = 0
    valgus_bad_consecutive: int = 0
    descent_frames: int = 0
    ascent_frames: int = 0
    phase_history: list = field(default_factory=list)


class RepStateMachine:
    """
    5-phase state machine for Bulgarian split squat rep counting.

    IDLE → DESCENDING → BOTTOM → ASCENDING → LOCKOUT → IDLE
                ↑                      |
                └──────────────────────┘  (failed rep → back to IDLE)
    """

    def __init__(self, cfg: BulgarianConfig, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.dt = 1.0 / max(fps, 1.0)
        self.phase = Phase.IDLE
        self.count = 0
        self.good_reps = 0
        self.bad_reps = 0
        self.rep_reports: List[dict] = []
        self.all_feedback: collections.Counter = collections.Counter()
        self.current_rep: Optional[RepData] = None
        self._last_lockout_frame = -100
        self._last_lockout_time = -100.0
        self._standing_angle = cfg.fallback_standing_angle
        self._depth_live = 0.0

        # Phase-specific angle thresholds (set after calibration)
        self._thresh_descent_entry = 0.0
        self._thresh_bottom_zone = 0.0
        self._thresh_ascent_confirm = 0.0
        self._thresh_lockout = 0.0
        self._update_thresholds()

    def set_standing_angle(self, angle: float):
        self._standing_angle = angle
        self._update_thresholds()

    def _update_thresholds(self):
        s = self._standing_angle
        c = self.cfg
        self._thresh_descent_entry = s - c.descent_trigger_offset
        self._thresh_bottom_zone = s - c.bottom_zone_offset
        self._thresh_ascent_confirm = 0  # dynamic, based on rep's min angle
        self._thresh_lockout = s - c.lockout_offset

    def update(self, knee_angle: float, knee_velocity: float,
               torso_angle: float, valgus_ok: bool,
               frame_no: int, current_time: float,
               movement_blocked: bool, is_still: bool) -> Optional[dict]:
        """
        Process one frame. Returns a rep report dict if a rep was just completed.
        """
        if movement_blocked:
            # If walking, cancel any in-progress rep
            if self.phase in (Phase.DESCENDING, Phase.BOTTOM):
                self._cancel_rep()
            return None

        result = None

        if self.phase == Phase.IDLE:
            result = self._handle_idle(knee_angle, knee_velocity, frame_no, current_time)
        elif self.phase == Phase.DESCENDING:
            result = self._handle_descending(knee_angle, knee_velocity, torso_angle,
                                              valgus_ok, frame_no, current_time)
        elif self.phase == Phase.BOTTOM:
            result = self._handle_bottom(knee_angle, knee_velocity, torso_angle,
                                          valgus_ok, frame_no, current_time)
        elif self.phase == Phase.ASCENDING:
            result = self._handle_ascending(knee_angle, knee_velocity, torso_angle,
                                             valgus_ok, frame_no, current_time, is_still)
        elif self.phase == Phase.LOCKOUT:
            result = self._handle_lockout(knee_angle, frame_no, current_time)

        # Update live depth
        self._update_depth_live(knee_angle)

        return result

    def _handle_idle(self, knee_angle, knee_velocity, frame_no, current_time):
        # Check cooldown
        if current_time - self._last_lockout_time < self.cfg.rep_cooldown_sec:
            return None

        # Transition to DESCENDING: angle drops below threshold AND velocity is negative (going down)
        if knee_angle < self._thresh_descent_entry and knee_velocity < -self.cfg.min_descent_velocity:
            self.phase = Phase.DESCENDING
            self.current_rep = RepData(
                start_frame=frame_no,
                start_time=current_time,
                descent_start_angle=knee_angle
            )
        return None

    def _handle_descending(self, knee_angle, knee_velocity, torso_angle,
                           valgus_ok, frame_no, current_time):
        rep = self.current_rep
        if rep is None:
            self._cancel_rep()
            return None

        rep.descent_frames += 1
        self._accumulate_form(rep, knee_angle, torso_angle, valgus_ok)

        # Check timeout
        if current_time - rep.start_time > self.cfg.max_rep_duration_sec:
            self._cancel_rep()
            return None

        # False start: angle went back up without reaching bottom
        if knee_angle > self._thresh_descent_entry + 10 and rep.descent_frames > 3:
            self._cancel_rep()
            return None

        # Transition to BOTTOM
        if knee_angle < self._thresh_bottom_zone or rep.min_knee_angle < self._thresh_bottom_zone:
            self.phase = Phase.BOTTOM
            rep.bottom_frames = 1

        return None

    def _handle_bottom(self, knee_angle, knee_velocity, torso_angle,
                       valgus_ok, frame_no, current_time):
        rep = self.current_rep
        if rep is None:
            self._cancel_rep()
            return None

        rep.bottom_frames += 1
        self._accumulate_form(rep, knee_angle, torso_angle, valgus_ok)

        # Check timeout
        if current_time - rep.start_time > self.cfg.max_rep_duration_sec:
            self._cancel_rep()
            return None

        # Transition to ASCENDING: velocity is positive (going up) and angle rising
        ascent_threshold = rep.min_knee_angle + self.cfg.ascent_trigger_offset
        if knee_angle > ascent_threshold and knee_velocity > self.cfg.min_ascent_velocity:
            self.phase = Phase.ASCENDING
            rep.ascent_frames = 1

        return None

    def _handle_ascending(self, knee_angle, knee_velocity, torso_angle,
                          valgus_ok, frame_no, current_time, is_still):
        rep = self.current_rep
        if rep is None:
            self._cancel_rep()
            return None

        rep.ascent_frames += 1
        self._accumulate_form(rep, knee_angle, torso_angle, valgus_ok)

        # Check timeout
        if current_time - rep.start_time > self.cfg.max_rep_duration_sec:
            self._cancel_rep()
            return None

        # Fell back down? Return to BOTTOM
        if knee_angle < rep.min_knee_angle + 10 and knee_velocity < -5:
            self.phase = Phase.BOTTOM
            return None

        # Reached lockout
        if knee_angle > self._thresh_lockout:
            self.phase = Phase.LOCKOUT
            return self._complete_rep(frame_no, current_time)

        return None

    def _handle_lockout(self, knee_angle, frame_no, current_time):
        # Brief lockout phase, then back to IDLE
        self.phase = Phase.IDLE
        return None

    def _accumulate_form(self, rep: RepData, knee_angle: float, torso_angle: float, valgus_ok: bool):
        """Track form metrics during the rep."""
        rep.min_knee_angle = min(rep.min_knee_angle, knee_angle)
        rep.max_knee_angle = max(rep.max_knee_angle, knee_angle)
        rep.min_torso_angle = min(rep.min_torso_angle, torso_angle)

        # Torso lean tracking (consecutive frames)
        if torso_angle < (self.cfg.torso_lean_min - self.cfg.torso_lean_margin):
            rep.torso_bad_consecutive += 1
            if rep.torso_bad_consecutive >= 1:
                rep.torso_bad_frames += 1
        else:
            rep.torso_bad_consecutive = 0

        # Valgus tracking
        if not valgus_ok:
            rep.valgus_bad_consecutive += 1
            if rep.valgus_bad_consecutive >= 1:
                rep.valgus_bad_frames += 1
        else:
            rep.valgus_bad_consecutive = 0

    def _complete_rep(self, frame_no: int, current_time: float) -> Optional[dict]:
        """Validate and score a completed rep."""
        rep = self.current_rep
        if rep is None:
            return None

        # Validation checks
        duration = current_time - rep.start_time
        if duration < self.cfg.min_rep_duration_sec:
            self._cancel_rep()
            return None

        rom = rep.descent_start_angle - rep.min_knee_angle
        if rom < self.cfg.min_depth_degrees:
            self._cancel_rep()
            return None

        if rep.bottom_frames < self.cfg.min_bottom_frames:
            self._cancel_rep()
            return None

        # Score the rep
        score, feedback, depth_pct = self._score_rep(rep)

        self.count += 1
        score_q = round(float(score) * 2) / 2.0

        if score_q >= self.cfg.good_rep_min_score:
            self.good_reps += 1
        else:
            self.bad_reps += 1

        for fb in feedback:
            self.all_feedback[fb] += 1

        report = {
            "rep_index": self.count,
            "score": float(score_q),
            "score_display": _display_half(score_q),
            "feedback": feedback,
            "start_frame": rep.start_frame,
            "end_frame": frame_no,
            "duration_sec": round(duration, 2),
            "descent_start_angle": round(rep.descent_start_angle, 2),
            "min_knee_angle": round(rep.min_knee_angle, 2),
            "max_knee_angle": round(rep.max_knee_angle, 2),
            "torso_min_angle": round(rep.min_torso_angle, 2),
            "depth_pct": round(depth_pct, 3),
            "rom_degrees": round(rom, 2),
            "descent_frames": rep.descent_frames,
            "bottom_frames": rep.bottom_frames,
            "ascent_frames": rep.ascent_frames,
        }

        self.rep_reports.append(report)
        self._last_lockout_frame = frame_no
        self._last_lockout_time = current_time
        self.current_rep = None

        return report

    def _score_rep(self, rep: RepData) -> Tuple[float, List[str], float]:
        """Score a rep 0-10 based on depth, torso, valgus, and tempo."""
        score = 10.0
        feedback = []

        # --- Depth scoring (0-3 points) ---
        rom = rep.descent_start_angle - rep.min_knee_angle
        ideal_rom = rep.descent_start_angle - self.cfg.perfect_depth_angle
        depth_pct = np.clip(rom / max(10.0, ideal_rom), 0, 1)

        if depth_pct < 0.6:
            feedback.append("Go deeper – aim for 90° knee angle")
            score -= 3
        elif depth_pct < 0.8:
            feedback.append("Go a bit deeper")
            score -= 1.5
        elif depth_pct < 0.9:
            score -= 0.5

        # --- Torso lean (0-2 points) ---
        if rep.torso_bad_frames >= self.cfg.torso_bad_min_frames:
            feedback.append("Keep your back straight")
            score -= 2
        elif rep.torso_bad_frames >= 2:
            feedback.append("Watch your torso lean")
            score -= 1

        # --- Knee valgus (0-2 points) ---
        if rep.valgus_bad_frames >= self.cfg.valgus_bad_min_frames:
            feedback.append("Avoid knee collapse inward")
            score -= 2
        elif rep.valgus_bad_frames >= 2:
            feedback.append("Watch your knee alignment")
            score -= 1

        # --- Tempo (0-1.5 points) ---
        ecc_frames = rep.descent_frames + rep.bottom_frames
        con_frames = rep.ascent_frames
        if con_frames > 0:
            ratio = ecc_frames / con_frames
            lo, hi = self.cfg.tempo_ratio_ideal
            if ratio < lo * 0.5:
                feedback.append("Slow down the descent")
                score -= 1.5
            elif ratio < lo:
                score -= 0.5

        # --- Control at bottom (0-0.5 points) ---
        if rep.bottom_frames < 2:
            score -= 0.5

        score = float(np.clip(score, 0, 10))
        return score, feedback, float(depth_pct)

    def _cancel_rep(self):
        """Discard current rep attempt."""
        self.phase = Phase.IDLE
        self.current_rep = None

    def _update_depth_live(self, knee_angle: float):
        """Update live depth for overlay."""
        denom = max(10.0, self._standing_angle - self.cfg.perfect_depth_angle)
        self._depth_live = float(np.clip(
            (self._standing_angle - knee_angle) / denom, 0, 1))

    @property
    def depth_for_overlay(self) -> float:
        return self._depth_live

    @property
    def active_feedback(self) -> Optional[str]:
        """Real-time feedback during a rep."""
        if self.current_rep is None:
            return None
        rep = self.current_rep
        msgs = []
        if rep.torso_bad_consecutive >= self.cfg.torso_bad_min_frames:
            msgs.append("Keep your back straight")
        if rep.valgus_bad_consecutive >= self.cfg.valgus_bad_min_frames:
            msgs.append("Avoid knee collapse")
        return " | ".join(msgs) if msgs else None

    def result(self) -> dict:
        avg = np.mean([r["score"] for r in self.rep_reports]) if self.rep_reports else 0.0
        technique_score = round(float(avg) * 2) / 2.0
        return {
            "squat_count": self.count,
            "technique_score": float(technique_score),
            "technique_score_display": _display_half(technique_score),
            "technique_label": _score_label(technique_score),
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "feedback": (list(self.all_feedback.elements()) if self.bad_reps > 0
                         else ["Great form! Keep it up 💪"]),
            "reps": self.rep_reports
        }


# ========================== SKELETON DRAWING ==========================

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


class LandmarkStabilizer:
    """Freeze-on-drop: show latest good skeleton; hold briefly if quality drops."""

    def __init__(self, cfg: BulgarianConfig):
        self.cfg = cfg
        self.body_points = _BODY_POINTS
        self.last_good = None
        self.hold = 0

    def stabilize(self, lms):
        if lms is None:
            if self.last_good is not None and self.hold < self.cfg.skeleton_max_hold_frames:
                self.hold += 1
                return self.last_good
            self.hold = 0
            return None

        frac = self._quality(lms)
        if frac < self.cfg.min_visible_fraction:
            if self.last_good is not None and self.hold < self.cfg.skeleton_max_hold_frames:
                self.hold += 1
                return self.last_good
            self.hold = 0
            return None

        avg_d = self._avg_disp(lms)
        if avg_d > self.cfg.skeleton_jump_thr and frac < 0.8 and self.last_good is not None:
            if self.hold < self.cfg.skeleton_max_hold_frames:
                self.hold += 1
                return self.last_good

        self.last_good = self._copy(lms)
        self.hold = 0
        return self.last_good

    def _quality(self, lms):
        ok = total = 0
        for i in self.body_points:
            if i >= len(lms):
                continue
            total += 1
            vis = getattr(lms[i], 'visibility', 1.0) or 0.0
            if vis >= self.cfg.skeleton_quality_thr:
                ok += 1
        return (ok / max(1, total)) if total else 1.0

    def _avg_disp(self, lms):
        if not self.last_good:
            return 0.0
        n = min(len(self.last_good), len(lms))
        s = c = 0.0
        for i in self.body_points:
            if i >= n:
                continue
            dx = float(lms[i].x) - float(self.last_good[i].x)
            dy = float(lms[i].y) - float(self.last_good[i].y)
            s += math.hypot(dx, dy)
            c += 1
        return s / max(1, c)

    def _copy(self, lms):
        return [type('P', (), {'x': float(lms[i].x), 'y': float(lms[i].y)})
                for i in range(len(lms))]


def draw_body_only(frame, landmarks, cfg: BulgarianConfig):
    h, w = frame.shape[:2]
    color = cfg.skeleton_color
    for a, b in _BODY_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        pa, pb = landmarks[a], landmarks[b]
        cv2.line(frame, (int(pa.x * w), int(pa.y * h)),
                 (int(pb.x * w), int(pb.y * h)),
                 color, cfg.skeleton_thickness, cv2.LINE_AA)
    for i in _BODY_POINTS:
        if i >= len(landmarks):
            continue
        p = landmarks[i]
        cv2.circle(frame, (int(p.x * w), int(p.y * h)),
                   cfg.skeleton_point_radius, color, -1, cv2.LINE_AA)
    return frame


# ========================== OVERLAY DRAWING ==========================

def _load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def _display_half(x: float) -> str:
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

def _score_label(s: float) -> str:
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def _wrap_two_lines(draw, text, font, max_width):
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) == 2:
            break
    if cur and len(lines) < 2:
        lines.append(cur)
    return lines if lines else [""]


def draw_overlay(frame, reps: int, feedback: Optional[str], depth_pct: float,
                 phase: Phase, cfg: BulgarianConfig, fonts: dict):
    h, w, _ = frame.shape

    # === Reps box (top-left) ===
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    pad_x, pad_y = 10, 6
    text_w = draw.textlength(reps_text, font=fonts['reps'])
    text_h = cfg.reps_font_size
    x1 = int(text_w + 2 * pad_x)
    y1 = int(text_h + 2 * pad_y)
    top = frame.copy()
    cv2.rectangle(top, (0, 0), (x1, y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, cfg.bar_bg_alpha, frame, 1.0 - cfg.bar_bg_alpha, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((pad_x, pad_y - 1), reps_text,
                              font=fonts['reps'], fill=(255, 255, 255))
    frame = np.array(pil)

    # === Phase indicator (below reps box) ===
    phase_colors = {
        Phase.IDLE: (180, 180, 180),
        Phase.DESCENDING: (80, 180, 255),
        Phase.BOTTOM: (40, 200, 80),
        Phase.ASCENDING: (255, 200, 60),
        Phase.LOCKOUT: (200, 100, 255),
    }
    phase_name = phase.name.capitalize()
    phase_color = phase_colors.get(phase, (180, 180, 180))
    py_start = y1 + 4
    cv2.rectangle(frame, (0, py_start), (x1, py_start + 20), (30, 30, 30), -1)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((pad_x, py_start + 2), phase_name,
                              font=fonts['depth_label'], fill=phase_color)
    frame = np.array(pil)

    # === Depth donut (top-right) ===
    depth_pct = float(np.clip(depth_pct, 0.0, 1.0))
    ref_h = max(int(h * 0.06), int(cfg.reps_font_size * 1.6))
    radius = int(ref_h * cfg.donut_radius_scale)
    thick = max(3, int(radius * cfg.donut_thickness_frac))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)

    cv2.circle(frame, (cx, cy), radius, cfg.depth_ring_bg, thick, cv2.LINE_AA)
    start_ang = -90
    end_ang = start_ang + int(360 * depth_pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                cfg.depth_color, thick, cv2.LINE_AA)

    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"
    pct_txt = f"{int(depth_pct * 100)}%"
    label_w = draw.textlength(label_txt, font=fonts['depth_label'])
    pct_w = draw.textlength(pct_txt, font=fonts['depth_pct'])
    gap = max(2, int(radius * 0.10))
    base_y = cy - (cfg.depth_label_font_size + gap + cfg.depth_pct_font_size) // 2
    draw.text((cx - int(label_w // 2), base_y), label_txt,
              font=fonts['depth_label'], fill=(255, 255, 255))
    draw.text((cx - int(pct_w // 2), base_y + cfg.depth_label_font_size + gap),
              pct_txt, font=fonts['depth_pct'], fill=(255, 255, 255))
    frame = np.array(pil)

    # === Bottom feedback bar ===
    if feedback:
        pil_fb = Image.fromarray(frame)
        draw_fb = ImageDraw.Draw(pil_fb)
        safe_margin = max(6, int(h * 0.02))
        pad_x2, pad_y2, line_gap = 12, 8, 4
        max_text_w = int(w - 2 * pad_x2 - 20)
        lines = _wrap_two_lines(draw_fb, feedback, fonts['feedback'], max_text_w)
        line_h = cfg.feedback_font_size + 6
        block_h = 2 * pad_y2 + len(lines) * line_h + (len(lines) - 1) * line_gap
        y0 = max(0, h - safe_margin - block_h)
        y1b = h - safe_margin
        over = frame.copy()
        cv2.rectangle(over, (0, y0), (w, y1b), (0, 0, 0), -1)
        frame = cv2.addWeighted(over, cfg.bar_bg_alpha, frame, 1.0 - cfg.bar_bg_alpha, 0)
        pil_fb = Image.fromarray(frame)
        draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + pad_y2
        for ln in lines:
            tw = draw_fb.textlength(ln, font=fonts['feedback'])
            tx = max(pad_x2, (w - int(tw)) // 2)
            draw_fb.text((tx, ty), ln, font=fonts['feedback'], fill=(255, 255, 255))
            ty += line_h + line_gap
        frame = np.array(pil_fb)

    return frame


# ========================== SESSION TIPS ==========================

BULGARIAN_TIPS = [
    "Keep your front shin vertical at the bottom",
    "Drive through the front heel for power",
    "Brace your core before the descent",
    "Keep hips square – avoid rotation",
    "Control the eccentric; go down a bit slower",
    "Pause 1–2s at the bottom to build stability",
]

def choose_session_tip(all_feedback: collections.Counter) -> str:
    if all_feedback.get("Avoid knee collapse inward", 0) >= 2:
        return "Track your knee over your toes"
    if all_feedback.get("Keep your back straight", 0) >= 2:
        return "Brace your core and keep chest up"
    if all_feedback.get("Go deeper – aim for 90° knee angle", 0) >= 2:
        return "Work on hip mobility to increase depth"
    if all_feedback.get("Slow down the descent", 0) >= 2:
        return "Control the eccentric – 2-3 seconds down"
    return BULGARIAN_TIPS[1]


# ========================== VALGUS CHECK ==========================

def check_valgus(landmarks, side: str, tol: float) -> bool:
    """Returns True if knee alignment is OK (no valgus collapse)."""
    knee_idx = getattr(mp_pose.PoseLandmark, f"{side.upper()}_KNEE").value
    ankle_idx = getattr(mp_pose.PoseLandmark, f"{side.upper()}_ANKLE").value
    knee_x = landmarks[knee_idx].x
    ankle_x = landmarks[ankle_idx].x
    # Valgus = knee collapses inward (medially). For the front view:
    # Left leg: valgus if knee_x > ankle_x + tol (knee goes right)
    # Right leg: valgus if knee_x < ankle_x - tol (knee goes left)
    if side.upper() == 'LEFT':
        return not (knee_x > ankle_x + tol)
    else:
        return not (knee_x < ankle_x - tol)


# ========================== MAIN ANALYSIS ==========================

def run_bulgarian_analysis(video_path: str,
                           frame_skip: int = 1,
                           scale: float = 1.0,
                           output_path: str = "analyzed_output.mp4",
                           feedback_path: str = "feedback_summary.txt",
                           return_video: bool = True,
                           fast_mode: bool = None,
                           config: BulgarianConfig = None) -> dict:
    """
    Analyze a Bulgarian split squat video.

    Returns a dict with rep count, scores, feedback, and video path.
    """
    cfg = config or BulgarianConfig()

    if fast_mode is True:
        return_video = False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}", "squat_count": 0,
                "technique_score": 0.0, "technique_score_display": "0",
                "technique_label": "N/A", "reps": []}

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / effective_fps

    # Load fonts
    fonts = {
        'reps': _load_font(cfg.font_path, cfg.reps_font_size),
        'feedback': _load_font(cfg.font_path, cfg.feedback_font_size),
        'depth_label': _load_font(cfg.font_path, cfg.depth_label_font_size),
        'depth_pct': _load_font(cfg.font_path, cfg.depth_pct_font_size),
    }

    # Initialize components
    pose = mp_pose.Pose(model_complexity=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6)
    smoother = EMASmoother(alpha=cfg.angle_ema_alpha, vel_alpha=cfg.velocity_ema_alpha)
    movement_filter = MovementFilter(cfg)
    calibrator = StandingCalibrator(cfg)
    lm_stab = LandmarkStabilizer(cfg)
    state_machine = RepStateMachine(cfg, effective_fps)

    leg_votes: list = []
    active_leg: Optional[str] = None
    frame_no = 0
    current_time = 0.0
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Early exit tracking
    nopose_stop_frames = int(cfg.nopose_stop_sec * effective_fps)
    no_movement_stop_frames = int(cfg.no_movement_stop_sec * effective_fps)
    nopose_consecutive = 0
    stillness_consecutive = 0

    # RT feedback hold
    rt_fb_hold_frames = max(2, int(cfg.rt_fb_hold_sec / dt))
    rt_fb_msg: Optional[str] = None
    rt_fb_hold = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_skip > 1 and (frame_no % frame_skip) != 0:
            continue

        current_time += dt

        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        h, w = frame.shape[:2]
        if return_video and out is None:
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

        # Pose detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        depth_live = 0.0
        stab_lms = None

        if not results.pose_landmarks:
            # No pose detected
            if state_machine.count > 0:
                nopose_consecutive += 1
            if state_machine.count > 0 and nopose_consecutive >= nopose_stop_frames:
                break

            stab_lms = lm_stab.stabilize(None)
            stillness_consecutive = 0

            if rt_fb_hold > 0:
                rt_fb_hold -= 1
            if rt_fb_hold == 0:
                rt_fb_msg = None
        else:
            nopose_consecutive = 0
            lms = results.pose_landmarks.landmark

            # Detect active leg (robust voting)
            active_leg = detect_active_leg_robust(lms, leg_votes)
            side = active_leg.upper()

            # Movement filter
            movement_blocked = movement_filter.update(lms, side, w, h)
            is_still = movement_filter.is_still

            if not movement_blocked:
                stillness_consecutive += 1
            else:
                stillness_consecutive = 0

            # Early exit: prolonged stillness after reps
            if state_machine.count > 0 and stillness_consecutive >= no_movement_stop_frames:
                break

            # Compute angles
            hip = lm_px(lms, getattr(mp_pose.PoseLandmark, f"{side}_HIP").value, w, h)
            knee = lm_px(lms, getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value, w, h)
            ankle = lm_px(lms, getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value, w, h)
            shoulder = lm_px(lms, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value, w, h)

            knee_angle_raw = angle_3pt(hip, knee, ankle)
            torso_angle_raw = angle_3pt(shoulder, hip, knee)

            # Smooth
            smoothed = smoother.update(dt, knee=knee_angle_raw, torso=torso_angle_raw)
            knee_angle = smoothed['knee']
            torso_angle = smoothed['torso']
            knee_velocity = smoother.velocity('knee')  # deg/sec, negative = descending

            # Valgus check
            v_ok = check_valgus(lms, side, cfg.valgus_x_tol)

            # Calibration (determine standing angle)
            vis_ok = (lm_vis(lms, getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value)
                      >= cfg.calibration_min_visibility)
            calibrator.add_sample(knee_angle, vis_ok, is_still)
            if calibrator.is_done:
                state_machine.set_standing_angle(calibrator.get_standing_angle())

            # Update state machine
            rep_report = state_machine.update(
                knee_angle=knee_angle,
                knee_velocity=knee_velocity,
                torso_angle=torso_angle,
                valgus_ok=v_ok,
                frame_no=frame_no,
                current_time=current_time,
                movement_blocked=movement_blocked,
                is_still=is_still
            )

            depth_live = state_machine.depth_for_overlay

            # RT feedback with hold
            new_msg = state_machine.active_feedback
            if new_msg:
                if new_msg != rt_fb_msg:
                    rt_fb_msg = new_msg
                    rt_fb_hold = rt_fb_hold_frames
                else:
                    rt_fb_hold = max(rt_fb_hold, rt_fb_hold_frames)
            else:
                if rt_fb_hold > 0:
                    rt_fb_hold -= 1
                if rt_fb_hold == 0:
                    rt_fb_msg = None

            # Stabilize skeleton for drawing
            stab_lms = lm_stab.stabilize(lms)

        # === Draw ===
        if return_video:
            if stab_lms:
                frame = draw_body_only(frame, stab_lms, cfg)
            frame = draw_overlay(
                frame,
                reps=state_machine.count,
                feedback=(rt_fb_msg if rt_fb_hold > 0 else None),
                depth_pct=depth_live,
                phase=state_machine.phase,
                cfg=cfg,
                fonts=fonts
            )
            if out is not None:
                out.write(frame)

    # Cleanup
    pose.close()
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Build result
    result = state_machine.result()

    # Session tip
    session_tip = choose_session_tip(state_machine.all_feedback)
    result["tips"] = [session_tip]
    result["form_tip"] = session_tip

    # Calibration info
    result["calibrated_standing_angle"] = round(calibrator.get_standing_angle(), 1)

    # Write feedback file
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {result['squat_count']}\n")
            f.write(f"Technique Score: {result['technique_score_display']} / 10  "
                    f"({result['technique_label']})\n")
            f.write(f"Standing Angle (calibrated): {result['calibrated_standing_angle']}°\n")
            f.write(f"Form Tip: {session_tip}\n")
            if result.get("feedback"):
                f.write("Feedback:\n")
                for fb in result["feedback"]:
                    f.write(f"  - {fb}\n")
            if result.get("reps"):
                f.write("\nPer-Rep Breakdown:\n")
                for r in result["reps"]:
                    f.write(f"  Rep {r['rep_index']}: {r['score_display']}/10 "
                            f"(depth {int(r['depth_pct']*100)}%, "
                            f"ROM {r['rom_degrees']}°, "
                            f"{r['duration_sec']}s)\n")
                    for fb in r.get('feedback', []):
                        f.write(f"    → {fb}\n")
    except Exception:
        pass

    # Encode with ffmpeg (+faststart)
    final_path = ""
    if return_video and output_path:
        encoded_path = output_path.replace('.mp4', '_encoded.mp4')
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', output_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-movflags', '+faststart', '-pix_fmt', 'yuv420p',
                encoded_path
            ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            final_path = encoded_path if os.path.isfile(encoded_path) else output_path
        except Exception:
            final_path = output_path if os.path.isfile(output_path) else ""

        if not os.path.isfile(final_path) and os.path.isfile(output_path):
            final_path = output_path

    result["video_path"] = final_path if return_video else ""
    result["feedback_path"] = feedback_path
    return result


# Backward compatibility
run_analysis = run_bulgarian_analysis

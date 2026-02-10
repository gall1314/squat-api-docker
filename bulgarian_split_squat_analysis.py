# -*- coding: utf-8 -*-
# bulgarian_split_squat_analysis.py
# ============================================================================
# V3 – Hybrid: uses PROVEN 2-state (up/down) counting logic from original
# that actually works, plus: adaptive calibration, better scoring, walking
# filter, debug logging, clean overlay.
#
# CRITICAL FIX vs V2: Removed velocity-based phase entry that was blocking
# all rep detection. Uses angle-only thresholds like the working original.
# ============================================================================

import os, math, subprocess, collections
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

mp_pose = mp.solutions.pose

# ========================== CONFIGURATION ==========================

@dataclass
class BulgarianConfig:
    # --- Core thresholds (angle-only, like original) ---
    angle_down_thresh: float = 95
    angle_up_thresh: float = 160
    min_range_delta_deg: float = 12
    min_down_frames: int = 5

    # --- Adaptive calibration ---
    use_calibration: bool = True
    calibration_frames: int = 12
    calibration_min_visibility: float = 0.55
    down_offset_from_standing: float = 70
    up_offset_from_standing: float = 12

    # --- Form scoring ---
    good_rep_min_score: float = 8.0
    perfect_min_knee: float = 70
    torso_lean_min: float = 135
    torso_margin_deg: float = 3
    torso_bad_min_frames: int = 4
    valgus_x_tol: float = 0.03
    valgus_bad_min_frames: int = 3

    # --- Smoothing ---
    ema_alpha: float = 0.6

    # --- Debounce ---
    rep_debounce_frames: int = 6

    # --- Walking filter ---
    hip_vel_thresh_pct: float = 0.014
    ankle_vel_thresh_pct: float = 0.017
    motion_ema_alpha: float = 0.65
    movement_clear_frames: int = 2

    # --- Early exit ---
    nopose_stop_sec: float = 1.2
    no_movement_stop_sec: float = 1.3

    # --- Overlay ---
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
    min_visible_fraction: float = 0.55

    # --- Debug ---
    debug_log: bool = True
    debug_log_path: str = "bulgarian_debug.log"


# ========================== GEOMETRY ==========================

def angle_3pt(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def lm_xy(landmarks, idx, w, h):
    return (landmarks[idx].x * w, landmarks[idx].y * h)

def detect_active_leg(landmarks):
    left_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    return 'right' if left_y < right_y else 'left'

def valgus_ok(landmarks, side, tol):
    knee_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x
    ankle_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x
    return not (knee_x < ankle_x - tol)


# ========================== ANGLE EMA ==========================

class AngleEMA:
    def __init__(self, alpha=0.6):
        self.alpha = float(alpha)
        self.knee = None
        self.torso = None
    def update(self, knee_angle, torso_angle):
        ka, ta = float(knee_angle), float(torso_angle)
        if self.knee is None:
            self.knee, self.torso = ka, ta
        else:
            a = self.alpha
            self.knee = a * ka + (1.0 - a) * self.knee
            self.torso = a * ta + (1.0 - a) * self.torso
        return self.knee, self.torso


# ========================== CALIBRATION ==========================

class StandingCalibrator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = []
        self.calibrated = None
        self._done = False

    @property
    def is_done(self):
        return self._done

    def add_sample(self, knee_angle, vis_ok, is_still):
        if self._done: return
        if vis_ok and is_still and knee_angle > 130:
            self.samples.append(knee_angle)
        if len(self.samples) >= self.cfg.calibration_frames:
            self.calibrated = float(np.median(self.samples))
            self._done = True

    def get_standing_angle(self):
        if self.calibrated is not None: return self.calibrated
        if self.samples: return float(np.median(self.samples))
        return 170.0


# ========================== SKELETON ==========================

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
    def __init__(self, cfg):
        self.cfg = cfg
        self.body_points = _BODY_POINTS
        self.last_good = None
        self.hold = 0

    def stabilize(self, lms):
        if lms is None:
            if self.last_good is not None and self.hold < self.cfg.skeleton_max_hold_frames:
                self.hold += 1; return self.last_good
            self.hold = 0; return None
        frac = self._quality(lms)
        if frac < self.cfg.min_visible_fraction:
            if self.last_good is not None and self.hold < self.cfg.skeleton_max_hold_frames:
                self.hold += 1; return self.last_good
            self.hold = 0; return None
        avg_d = self._avg_disp(lms)
        if avg_d > self.cfg.skeleton_jump_thr and frac < 0.8 and self.last_good is not None:
            if self.hold < self.cfg.skeleton_max_hold_frames:
                self.hold += 1; return self.last_good
        self.last_good = [type('P', (), {'x': float(lms[i].x), 'y': float(lms[i].y)}) for i in range(len(lms))]
        self.hold = 0; return self.last_good

    def _quality(self, lms):
        ok = total = 0
        for i in self.body_points:
            if i >= len(lms): continue
            total += 1
            if (getattr(lms[i], 'visibility', 1.0) or 0.0) >= self.cfg.skeleton_quality_thr: ok += 1
        return (ok / max(1, total)) if total else 1.0

    def _avg_disp(self, lms):
        if not self.last_good: return 0.0
        n = min(len(self.last_good), len(lms))
        s = c = 0.0
        for i in self.body_points:
            if i >= n: continue
            s += math.hypot(float(lms[i].x) - float(self.last_good[i].x),
                            float(lms[i].y) - float(self.last_good[i].y)); c += 1
        return s / max(1, c)

def draw_body_only(frame, landmarks, cfg):
    h, w = frame.shape[:2]
    for a, b in _BODY_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks): continue
        pa, pb = landmarks[a], landmarks[b]
        cv2.line(frame, (int(pa.x*w), int(pa.y*h)), (int(pb.x*w), int(pb.y*h)),
                 cfg.skeleton_color, cfg.skeleton_thickness, cv2.LINE_AA)
    for i in _BODY_POINTS:
        if i >= len(landmarks): continue
        p = landmarks[i]
        cv2.circle(frame, (int(p.x*w), int(p.y*h)), cfg.skeleton_point_radius,
                   cfg.skeleton_color, -1, cv2.LINE_AA)
    return frame


# ========================== OVERLAY ==========================

def _load_font(path, size):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

def _display_half(x):
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

def _score_label(s):
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def _wrap_two_lines(draw, text, font, max_width):
    words = text.split()
    if not words: return [""]
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width: cur = trial
        else:
            if cur: lines.append(cur)
            cur = w
        if len(lines) == 2: break
    if cur and len(lines) < 2: lines.append(cur)
    return lines if lines else [""]

def draw_overlay(frame, reps, feedback, depth_pct, cfg, fonts):
    h, w, _ = frame.shape
    depth_pct = float(np.clip(depth_pct, 0.0, 1.0))

    # Reps box
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    px, py = 10, 6
    tw = draw.textlength(reps_text, font=fonts['reps'])
    x1 = int(tw + 2*px); y1 = int(cfg.reps_font_size + 2*py)
    top = frame.copy(); cv2.rectangle(top, (0,0), (x1,y1), (0,0,0), -1)
    frame = cv2.addWeighted(top, cfg.bar_bg_alpha, frame, 1.0-cfg.bar_bg_alpha, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((px, py-1), reps_text, font=fonts['reps'], fill=(255,255,255))
    frame = np.array(pil)

    # Donut
    ref_h = max(int(h*0.06), int(cfg.reps_font_size*1.6))
    radius = int(ref_h * cfg.donut_radius_scale)
    thick = max(3, int(radius * cfg.donut_thickness_frac))
    cx = w - 12 - radius
    cy = max(ref_h + radius//8, radius + thick//2 + 2)
    cv2.circle(frame, (cx,cy), radius, cfg.depth_ring_bg, thick, cv2.LINE_AA)
    cv2.ellipse(frame, (cx,cy), (radius,radius), 0, -90, -90+int(360*depth_pct),
                cfg.depth_color, thick, cv2.LINE_AA)
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    lt = "DEPTH"; pt = f"{int(depth_pct*100)}%"
    lw2 = draw.textlength(lt, font=fonts['depth_label'])
    pw2 = draw.textlength(pt, font=fonts['depth_pct'])
    gap = max(2, int(radius*0.10))
    by = cy - (cfg.depth_label_font_size + gap + cfg.depth_pct_font_size)//2
    draw.text((cx-int(lw2//2), by), lt, font=fonts['depth_label'], fill=(255,255,255))
    draw.text((cx-int(pw2//2), by+cfg.depth_label_font_size+gap), pt, font=fonts['depth_pct'], fill=(255,255,255))
    frame = np.array(pil)

    # Feedback
    if feedback:
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        safe = max(6, int(h*0.02)); mx = int(w - 24 - 20)
        lines = _wrap_two_lines(draw_fb, feedback, fonts['feedback'], mx)
        lh = cfg.feedback_font_size + 6
        bh = 16 + len(lines)*lh + (len(lines)-1)*4
        y0 = max(0, h - safe - bh)
        over = frame.copy(); cv2.rectangle(over, (0,y0), (w,h-safe), (0,0,0), -1)
        frame = cv2.addWeighted(over, cfg.bar_bg_alpha, frame, 1.0-cfg.bar_bg_alpha, 0)
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + 8
        for ln in lines:
            tww = draw_fb.textlength(ln, font=fonts['feedback'])
            draw_fb.text((max(12, (w-int(tww))//2), ty), ln, font=fonts['feedback'], fill=(255,255,255))
            ty += lh + 4
        frame = np.array(pil_fb)
    return frame


# ========================== REP COUNTER ==========================

class BulgarianRepCounter:
    """
    PROVEN 2-state (up/down) logic from original code.
    + adaptive thresholds, better scoring, debug logging.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.count = 0
        self.stage = None
        self.rep_reports = []
        self.rep_index = 1
        self.rep_start_frame = None
        self.good_reps = 0
        self.bad_reps = 0
        self.all_feedback = collections.Counter()
        self._start_knee_angle = None
        self._curr_min_knee = 999.0
        self._curr_max_knee = -999.0
        self._curr_min_torso = 999.0
        self._curr_valgus_bad = 0
        self._torso_bad_frames = 0
        self._valgus_bad_frames = 0
        self._down_frames = 0
        self._last_depth_for_ui = 0.0
        self._last_rep_end_frame = -10
        self._down_thresh = cfg.angle_down_thresh
        self._up_thresh = cfg.angle_up_thresh
        self._debug_lines = []

    def _log(self, msg):
        self._debug_lines.append(msg)

    def set_calibrated_thresholds(self, standing_angle):
        cfg = self.cfg
        new_down = np.clip(standing_angle - cfg.down_offset_from_standing, 70, 120)
        new_up = np.clip(standing_angle - cfg.up_offset_from_standing, 140, 175)
        self._log(f"CALIBRATE: standing={standing_angle:.1f}, "
                  f"down: {self._down_thresh:.1f}→{new_down:.1f}, "
                  f"up: {self._up_thresh:.1f}→{new_up:.1f}")
        self._down_thresh = float(new_down)
        self._up_thresh = float(new_up)

    def _start_rep(self, frame_no, start_knee):
        if frame_no - self._last_rep_end_frame < self.cfg.rep_debounce_frames:
            self._log(f"F{frame_no}: DEBOUNCE skip"); return False
        self.rep_start_frame = frame_no
        self._start_knee_angle = float(start_knee)
        self._curr_min_knee = 999.0; self._curr_max_knee = -999.0
        self._curr_min_torso = 999.0; self._curr_valgus_bad = 0
        self._torso_bad_frames = 0; self._valgus_bad_frames = 0
        self._down_frames = 0
        self._log(f"F{frame_no}: REP START knee={start_knee:.1f}")
        return True

    def _finish_rep(self, frame_no, score, feedback, extra=None):
        score_q = round(float(score) * 2) / 2.0
        if score_q >= self.cfg.good_rep_min_score: self.good_reps += 1
        else: self.bad_reps += 1
        for fb in (feedback or []): self.all_feedback[fb] += 1
        report = {
            "rep_index": self.rep_index,
            "score": float(score_q),
            "score_display": _display_half(score_q),
            "feedback": feedback or [],
            "start_frame": self.rep_start_frame or 0,
            "end_frame": frame_no,
            "start_knee_angle": round(float(self._start_knee_angle or 0), 2),
            "min_knee_angle": round(self._curr_min_knee, 2),
            "max_knee_angle": round(self._curr_max_knee, 2),
            "torso_min_angle": round(self._curr_min_torso, 2),
        }
        if extra: report.update(extra)
        self.rep_reports.append(report)
        self._log(f"F{frame_no}: ✓ REP #{self.rep_index} score={score_q}, "
                  f"min_knee={self._curr_min_knee:.1f}, down_frames={self._down_frames}")
        self.rep_index += 1
        self.rep_start_frame = None; self._start_knee_angle = None
        self._last_depth_for_ui = 0.0; self._last_rep_end_frame = frame_no

    def evaluate_form(self, start_knee, min_knee, min_torso, valgus_bad):
        feedback = []; score = 10.0
        denom = max(10.0, start_knee - self.cfg.perfect_min_knee)
        depth_pct = np.clip((start_knee - min_knee) / denom, 0, 1)
        if depth_pct < 0.6:
            feedback.append("Go deeper – aim for 90° knee angle"); score -= 3
        elif depth_pct < 0.8:
            feedback.append("Go a bit deeper"); score -= 1.5
        elif depth_pct < 0.9:
            score -= 0.5
        if self._torso_bad_frames >= self.cfg.torso_bad_min_frames:
            feedback.append("Keep your back straight"); score -= 2
        if valgus_bad >= self.cfg.valgus_bad_min_frames:
            feedback.append("Avoid knee collapse"); score -= 2
        return float(np.clip(score, 0, 10)), feedback, float(depth_pct)

    def update(self, knee_angle, torso_angle, valgus_ok_flag, frame_no):
        """Core 2-state logic – same as original that works."""
        if knee_angle < self._down_thresh:
            if self.stage != 'down':
                self.stage = 'down'
                if not self._start_rep(frame_no, knee_angle):
                    self.stage = 'up'; return
            self._down_frames += 1
        elif knee_angle > self._up_thresh and self.stage == 'down':
            depth_delta = (self._start_knee_angle or 0) - (self._curr_min_knee if self._curr_min_knee < 900 else 0)
            ok_move = depth_delta >= self.cfg.min_range_delta_deg
            ok_frames = self._down_frames >= self.cfg.min_down_frames
            if ok_frames and ok_move:
                score, fb, depth = self.evaluate_form(
                    float(self._start_knee_angle or knee_angle),
                    float(self._curr_min_knee if self._curr_min_knee < 900 else knee_angle),
                    float(self._curr_min_torso if self._curr_min_torso < 900 else 180),
                    self._curr_valgus_bad)
                self.count += 1
                self._finish_rep(frame_no, score, fb, extra={"depth_pct": float(depth)})
            else:
                self._log(f"F{frame_no}: REJECT move={ok_move}(Δ{depth_delta:.1f}°) frames={ok_frames}({self._down_frames})")
                self._last_depth_for_ui = 0.0
                self.rep_start_frame = None; self._start_knee_angle = None
            self.stage = 'up'

        if self.stage == 'down' and self.rep_start_frame:
            self._curr_min_knee = min(self._curr_min_knee, knee_angle)
            self._curr_max_knee = max(self._curr_max_knee, knee_angle)
            self._curr_min_torso = min(self._curr_min_torso, torso_angle)
            if torso_angle < (self.cfg.torso_lean_min - self.cfg.torso_margin_deg):
                self._torso_bad_frames += 1
            else:
                self._torso_bad_frames = 0
            if not valgus_ok_flag:
                self._valgus_bad_frames += 1; self._curr_valgus_bad += 1
            else:
                self._valgus_bad_frames = 0
            denom = max(10.0, self._start_knee_angle - self.cfg.perfect_min_knee)
            self._last_depth_for_ui = float(np.clip(
                (self._start_knee_angle - self._curr_min_knee) / denom, 0, 1))

    def depth_for_overlay(self):
        return float(self._last_depth_for_ui)

    def result(self):
        avg = np.mean([float(r["score"]) for r in self.rep_reports]) if self.rep_reports else 0.0
        ts = round(float(avg) * 2) / 2.0
        return {
            "squat_count": self.count,
            "technique_score": float(ts),
            "technique_score_display": _display_half(ts),
            "technique_label": _score_label(ts),
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "feedback": list(self.all_feedback.elements()) if self.bad_reps > 0 else ["Great form! Keep it up 💪"],
            "reps": self.rep_reports,
        }


# ========================== TIPS ==========================

BULGARIAN_TIPS = [
    "Keep your front shin vertical at the bottom",
    "Drive through the front heel for power",
    "Brace your core before the descent",
    "Keep hips square – avoid rotation",
]

def choose_session_tip(counter):
    if counter.all_feedback.get("Avoid knee collapse", 0) >= 2:
        return "Track your knee over your toes"
    if counter.all_feedback.get("Keep your back straight", 0) >= 2:
        return "Brace your core and keep chest up"
    return BULGARIAN_TIPS[1]


# ========================== MAIN ==========================

def run_bulgarian_analysis(video_path, frame_skip=1, scale=1.0,
                           output_path="analyzed_output.mp4",
                           feedback_path="feedback_summary.txt",
                           return_video=True, fast_mode=None, config=None):
    cfg = config or BulgarianConfig()
    if fast_mode is True: return_video = False
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Debug log
    debug_file = None
    if cfg.debug_log:
        dp = cfg.debug_log_path
        if os.path.dirname(output_path):
            dp = os.path.join(os.path.dirname(output_path), dp)
        try:
            debug_file = open(dp, "w", encoding="utf-8")
            debug_file.write("=== Bulgarian Split Squat Debug Log V3 ===\n\n")
        except: pass
    def dlog(msg):
        if debug_file:
            try: debug_file.write(msg + "\n")
            except: pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open: {video_path}", "squat_count": 0,
                "technique_score": 0.0, "technique_score_display": "0",
                "technique_label": "N/A", "reps": []}

    counter = BulgarianRepCounter(cfg)
    frame_no = 0; active_leg = None; out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    ema = AngleEMA(alpha=cfg.ema_alpha)
    lm_stab = LandmarkStabilizer(cfg)
    calibrator = StandingCalibrator(cfg) if cfg.use_calibration else None
    calibration_applied = False

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    dlog(f"Video: {video_path}")
    dlog(f"FPS={fps_in}, frame_skip={frame_skip}, effective_fps={effective_fps:.1f}")
    dlog(f"Thresholds: down={cfg.angle_down_thresh}, up={cfg.angle_up_thresh}")
    dlog(f"Calibration={'ON' if cfg.use_calibration else 'OFF'}")
    dlog(f"min_ROM={cfg.min_range_delta_deg}°, min_down_frames={cfg.min_down_frames}\n")

    fonts = {
        'reps': _load_font(cfg.font_path, cfg.reps_font_size),
        'feedback': _load_font(cfg.font_path, cfg.feedback_font_size),
        'depth_label': _load_font(cfg.font_path, cfg.depth_label_font_size),
        'depth_pct': _load_font(cfg.font_path, cfg.depth_pct_font_size),
    }

    NOPOSE_STOP = int(cfg.nopose_stop_sec * effective_fps)
    NOMOVE_STOP = int(cfg.no_movement_stop_sec * effective_fps)
    nopose_since_rep = 0; no_move_frames = 0
    RT_HOLD = max(2, int(cfg.rt_fb_hold_sec / dt))
    rt_fb_msg = None; rt_fb_hold = 0

    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    move_free = 0; stand_knee_ema = None
    pose_ok = 0; pose_fail = 0

    def _eu(a, b, n): return math.hypot(a[0]-b[0], a[1]-b[1]) / max(1, n)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1
        if frame_skip > 1 and (frame_no % frame_skip) != 0: continue
        if scale != 1.0: frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

        h, w = frame.shape[:2]
        if return_video and out is None:
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_live = 0.0; stab_lms = None

        if not results.pose_landmarks:
            pose_fail += 1
            if counter.count > 0: nopose_since_rep += 1
            else: nopose_since_rep = 0
            if counter.count > 0 and nopose_since_rep >= NOPOSE_STOP:
                dlog(f"F{frame_no}: EXIT no pose"); break
            if rt_fb_hold > 0: rt_fb_hold -= 1
            no_move_frames = 0
            stab_lms = lm_stab.stabilize(None)
        else:
            pose_ok += 1; nopose_since_rep = 0
            lms = results.pose_landmarks.landmark
            if active_leg is None:
                active_leg = detect_active_leg(lms)
                dlog(f"F{frame_no}: leg={active_leg}")
            side = "RIGHT" if active_leg == "right" else "LEFT"

            # Walking filter
            hp = (lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x*w,
                  lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y*h)
            la = (lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].x*w,
                  lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].y*h)
            ra = (lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x*w,
                  lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y*h)
            if prev_hip is None: prev_hip, prev_la, prev_ra = hp, la, ra
            nm = max(h, w)
            hv = _eu(hp, prev_hip, nm); av = max(_eu(la, prev_la, nm), _eu(ra, prev_ra, nm))
            hip_vel_ema = cfg.motion_ema_alpha*hv + (1-cfg.motion_ema_alpha)*hip_vel_ema
            ankle_vel_ema = cfg.motion_ema_alpha*av + (1-cfg.motion_ema_alpha)*ankle_vel_ema
            prev_hip, prev_la, prev_ra = hp, la, ra
            blocked = hip_vel_ema > cfg.hip_vel_thresh_pct or ankle_vel_ema > cfg.ankle_vel_thresh_pct
            if blocked: move_free = 0; no_move_frames = 0
            else: move_free = min(cfg.movement_clear_frames, move_free+1); no_move_frames += 1
            if counter.count > 0 and no_move_frames >= NOMOVE_STOP:
                dlog(f"F{frame_no}: EXIT still"); break

            # Angles
            hip_pt = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_HIP").value, w, h)
            knee_pt = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value, w, h)
            ankle_pt = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value, w, h)
            shoulder_pt = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value, w, h)
            ka_raw = angle_3pt(hip_pt, knee_pt, ankle_pt)
            ta_raw = angle_3pt(shoulder_pt, hip_pt, knee_pt)
            ka, ta = ema.update(ka_raw, ta_raw)
            vok = valgus_ok(lms, side, cfg.valgus_x_tol)

            # Calibration
            if calibrator and not calibration_applied:
                vis = getattr(lms[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value], 'visibility', 0) or 0
                calibrator.add_sample(ka, vis >= cfg.calibration_min_visibility, not blocked)
                if calibrator.is_done:
                    counter.set_calibrated_thresholds(calibrator.get_standing_angle())
                    calibration_applied = True

            # Log
            if frame_no % 10 == 0:
                dlog(f"F{frame_no}: ka_raw={ka_raw:.1f} ka={ka:.1f} ta={ta:.1f} "
                     f"stage={counter.stage} cnt={counter.count} blk={blocked} "
                     f"down_t={counter._down_thresh:.1f} up_t={counter._up_thresh:.1f}")

            # Update counter (only if not blocked)
            if not blocked or move_free >= cfg.movement_clear_frames:
                counter.update(ka, ta, vok, frame_no)
            else:
                if frame_no % 10 == 0: dlog(f"F{frame_no}: BLOCKED by walk filter")

            # Live depth
            if ka > counter._up_thresh - 3 and move_free >= 1:
                stand_knee_ema = ka if stand_knee_ema is None else (0.3*ka + 0.7*stand_knee_ema)
            if stand_knee_ema is not None:
                depth_live = float(np.clip((stand_knee_ema - ka) / max(10, stand_knee_ema - cfg.perfect_min_knee), 0, 1))
            else:
                depth_live = counter.depth_for_overlay()

            # RT feedback
            msgs = []
            if counter.stage == 'down':
                if counter._torso_bad_frames >= cfg.torso_bad_min_frames: msgs.append("Keep your back straight")
                if counter._valgus_bad_frames >= cfg.valgus_bad_min_frames: msgs.append("Avoid knee collapse")
            nm2 = " | ".join(msgs) if msgs else None
            if nm2:
                if nm2 != rt_fb_msg: rt_fb_msg = nm2; rt_fb_hold = RT_HOLD
                else: rt_fb_hold = max(rt_fb_hold, RT_HOLD)
            else:
                if rt_fb_hold > 0: rt_fb_hold -= 1
                if rt_fb_hold == 0: rt_fb_msg = None

            stab_lms = lm_stab.stabilize(lms)

        # Draw
        if return_video:
            if stab_lms: frame = draw_body_only(frame, stab_lms, cfg)
            frame = draw_overlay(frame, counter.count, rt_fb_msg if rt_fb_hold > 0 else None,
                                 depth_live, cfg, fonts)
            if out: out.write(frame)

    pose.close(); cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    result = counter.result()
    tip = choose_session_tip(counter)
    result["tips"] = [tip]; result["form_tip"] = tip
    if calibrator: result["calibrated_standing_angle"] = round(calibrator.get_standing_angle(), 1)

    # Debug summary
    dlog(f"\n{'='*50}\nFINAL: frames={frame_no} pose_ok={pose_ok} pose_fail={pose_fail}")
    dlog(f"leg={active_leg} cal={'done' if calibration_applied else 'no'}")
    if calibrator: dlog(f"standing={calibrator.get_standing_angle():.1f}°")
    dlog(f"thresholds: down={counter._down_thresh:.1f} up={counter._up_thresh:.1f}")
    dlog(f"count={counter.count} good={counter.good_reps} bad={counter.bad_reps}")
    dlog(f"\nCounter log:")
    for l in counter._debug_lines: dlog(f"  {l}")
    if debug_file:
        try: debug_file.close()
        except: pass

    result["_debug"] = {
        "total_frames": frame_no, "pose_ok": pose_ok, "pose_fail": pose_fail,
        "active_leg": active_leg,
        "calibration_done": calibration_applied,
        "standing_angle": round(calibrator.get_standing_angle(), 1) if calibrator else None,
        "final_down_thresh": round(counter._down_thresh, 1),
        "final_up_thresh": round(counter._up_thresh, 1),
    }

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {result['squat_count']}\n")
            f.write(f"Technique Score: {result['technique_score_display']} / 10  ({result['technique_label']})\n")
            f.write(f"Form Tip: {tip}\n")
            if result.get("feedback"):
                f.write("Feedback:\n")
                for fb in result["feedback"]: f.write(f"- {fb}\n")
    except: pass

    final_path = ""
    if return_video and output_path:
        enc = output_path.replace('.mp4', '_encoded.mp4')
        try:
            subprocess.run(['ffmpeg','-y','-i',output_path,'-c:v','libx264','-preset','fast',
                            '-movflags','+faststart','-pix_fmt','yuv420p',enc],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            final_path = enc if os.path.isfile(enc) else output_path
        except: final_path = output_path if os.path.isfile(output_path) else ""
        if not os.path.isfile(final_path) and os.path.isfile(output_path): final_path = output_path

    result["video_path"] = final_path if return_video else ""
    result["feedback_path"] = feedback_path
    return result

# Backward compatibility
run_analysis = run_bulgarian_analysis

# -*- coding: utf-8 -*-
# bulgarian_split_squat_analysis.py
# ============================================================================
# Production-ready Bulgarian Split Squat analyzer.
#
# DESIGNED TO WORK WITH: frame_skip=3, scale=0.4 (from app.py)
# This means: ~8-10 effective FPS, small image → noisy landmarks.
# Thresholds are tuned for these real-world conditions.
# ============================================================================

import os, math, subprocess, collections
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

mp_pose = mp.solutions.pose

# ========================== CONSTANTS ==========================
# --- These are tuned for frame_skip=3, scale=0.4 ---

# Rep detection thresholds
ANGLE_DOWN_THRESH   = 105     # knee angle below this = "down" (wider than 95 for noisy data)
ANGLE_UP_THRESH     = 150     # knee angle above this = "up" (lower than 160 for noisy data)
MIN_RANGE_DELTA_DEG = 10      # min ROM to count (lower than 12 for skip=3)
MIN_DOWN_FRAMES     = 2       # with skip=3 at 30fps → only ~3-5 frames in descent
FAST_REP_MIN_DEPTH_DELTA = 18 # allow faster reps if there is clear ROM

# Form scoring
GOOD_REP_MIN_SCORE  = 8.0
PERFECT_MIN_KNEE    = 70
TORSO_LEAN_MIN      = 112
TORSO_MARGIN_DEG    = 3
TORSO_BAD_MIN_FRAMES = 8       # much stricter persistence to avoid natural-lean false positives
VALGUS_X_TOL        = 0.03
VALGUS_BAD_MIN_FRAMES = 2      # lower for skip=3

# Smoothing
EMA_ALPHA           = 0.7     # higher = more responsive (less lag with few frames)

# Debounce
REP_DEBOUNCE_FRAMES = 3       # lower for skip=3 (each frame = 3 real frames)

# Walking filter
HIP_VEL_THRESH_PCT   = 0.018  # more lenient (noisy at low res)
ANKLE_VEL_THRESH_PCT  = 0.022
MOTION_EMA_ALPHA      = 0.55
MOVEMENT_CLEAR_FRAMES = 1     # just 1 clean frame needed at low fps

# Early exit
NOPOSE_STOP_SEC       = 1.5
NO_MOVEMENT_STOP_SEC  = 1.5

# Overlay style
BAR_BG_ALPHA         = 0.55
REPS_FONT_SIZE       = 28
FEEDBACK_FONT_SIZE   = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE  = 18
FONT_PATH            = "Roboto-VariableFont_wdth,wght.ttf"
RT_FB_HOLD_SEC       = 0.8
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)
DEPTH_RING_BG        = (70, 70, 70)


# ========================== GEOMETRY ==========================

def angle_3pt(a, b, c):
    """Angle at point b formed by segments ba and bc. Returns degrees [0, 360)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def lm_xy(landmarks, idx, w, h):
    return (landmarks[idx].x * w, landmarks[idx].y * h)

def detect_active_leg(landmarks):
    """Front (working) leg has LOWER ankle in frame (higher y)."""
    left_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    return 'right' if left_y < right_y else 'left'

def check_valgus(landmarks, side, tol):
    knee_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x
    ankle_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x
    return not (knee_x < ankle_x - tol)


# ========================== EMA ==========================

class AngleEMA:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = float(alpha)
        self.knee = None
        self.torso = None

    def update(self, knee_angle, torso_angle):
        ka, ta = float(knee_angle), float(torso_angle)
        if self.knee is None:
            self.knee, self.torso = ka, ta
        else:
            a = self.alpha
            self.knee  = a * ka + (1.0 - a) * self.knee
            self.torso = a * ta + (1.0 - a) * self.torso
        return self.knee, self.torso


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
    def __init__(self, quality_thr=0.55, min_fraction=0.6, jump_thr=0.12, max_hold=6):
        self.body_points = _BODY_POINTS
        self.quality_thr = quality_thr
        self.min_fraction = min_fraction
        self.jump_thr = jump_thr
        self.max_hold = max_hold
        self.last_good = None
        self.hold = 0

    def stabilize(self, lms):
        if lms is None:
            if self.last_good is not None and self.hold < self.max_hold:
                self.hold += 1; return self.last_good
            self.hold = 0; return None

        frac = self._quality(lms)
        if frac < self.min_fraction:
            if self.last_good is not None and self.hold < self.max_hold:
                self.hold += 1; return self.last_good
            self.hold = 0; return None

        avg_d = self._avg_disp(lms)
        if avg_d > self.jump_thr and frac < 0.8 and self.last_good is not None:
            if self.hold < self.max_hold:
                self.hold += 1; return self.last_good

        self.last_good = [type('P', (), {'x': float(lms[i].x), 'y': float(lms[i].y)})
                          for i in range(len(lms))]
        self.hold = 0
        return self.last_good

    def _quality(self, lms):
        ok = total = 0
        for i in self.body_points:
            if i >= len(lms): continue
            total += 1
            if (getattr(lms[i], 'visibility', 1.0) or 0.0) >= self.quality_thr:
                ok += 1
        return (ok / max(1, total)) if total else 1.0

    def _avg_disp(self, lms):
        if not self.last_good: return 0.0
        n = min(len(self.last_good), len(lms))
        s = c = 0.0
        for i in self.body_points:
            if i >= n: continue
            s += math.hypot(float(lms[i].x) - float(self.last_good[i].x),
                            float(lms[i].y) - float(self.last_good[i].y))
            c += 1
        return s / max(1, c)


def draw_body_only(frame, landmarks, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    for a, b in _BODY_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks): continue
        pa, pb = landmarks[a], landmarks[b]
        cv2.line(frame, (int(pa.x*w), int(pa.y*h)), (int(pb.x*w), int(pb.y*h)),
                 color, 2, cv2.LINE_AA)
    for i in _BODY_POINTS:
        if i >= len(landmarks): continue
        p = landmarks[i]
        cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, color, -1, cv2.LINE_AA)
    return frame


# ========================== OVERLAY ==========================

def _load_font(path, size):
    try: return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

REPS_FONT = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)


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
    leftover = len(words) - sum(len(l.split()) for l in lines)
    if leftover > 0 and len(lines) >= 2:
        last = lines[-1] + "…"
        while draw.textlength(last, font=font) > max_width and len(last) > 1:
            last = last[:-2] + "…"
        lines[-1] = last
    return lines if lines else [""]


def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    h, w, _ = frame.shape
    depth_pct = float(np.clip(depth_pct, 0.0, 1.0))

    # Reps box (top-left)
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    px, py = 10, 6
    tw = draw.textlength(reps_text, font=REPS_FONT)
    x1 = int(tw + 2*px); y1 = int(REPS_FONT_SIZE + 2*py)
    top = frame.copy(); cv2.rectangle(top, (0,0), (x1,y1), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1.0-BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((px, py-1), reps_text, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Depth donut (top-right)
    ref_h = max(int(h*0.06), int(REPS_FONT_SIZE*1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick = max(3, int(radius * DONUT_THICKNESS_FRAC))
    cx = w - 12 - radius
    cy = max(ref_h + radius//8, radius + thick//2 + 2)
    cv2.circle(frame, (cx,cy), radius, DEPTH_RING_BG, thick, cv2.LINE_AA)
    cv2.ellipse(frame, (cx,cy), (radius,radius), 0, -90, -90+int(360*depth_pct),
                DEPTH_COLOR, thick, cv2.LINE_AA)
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    lt = "DEPTH"; pt = f"{int(depth_pct*100)}%"
    lw = draw.textlength(lt, font=DEPTH_LABEL_FONT)
    pw = draw.textlength(pt, font=DEPTH_PCT_FONT)
    gap = max(2, int(radius*0.10))
    by = cy - (DEPTH_LABEL_FONT_SIZE + gap + DEPTH_PCT_FONT_SIZE)//2
    draw.text((cx-int(lw//2), by), lt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx-int(pw//2), by+DEPTH_LABEL_FONT_SIZE+gap), pt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Feedback bar (bottom)
    if feedback:
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        safe = max(6, int(h*0.02))
        max_tw = int(w - 44)
        lines = _wrap_two_lines(draw_fb, feedback, FEEDBACK_FONT, max_tw)
        lh = FEEDBACK_FONT_SIZE + 6
        bh = 16 + len(lines)*lh + (len(lines)-1)*4
        y0 = max(0, h - safe - bh)
        over = frame.copy(); cv2.rectangle(over, (0,y0), (w,h-safe), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0-BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + 8
        for ln in lines:
            tww = draw_fb.textlength(ln, font=FEEDBACK_FONT)
            draw_fb.text((max(12, (w-int(tww))//2), ty), ln, font=FEEDBACK_FONT, fill=(255,255,255))
            ty += lh + 4
        frame = np.array(pil_fb)
    return frame


# ========================== REP COUNTER ==========================

class BulgarianRepCounter:
    """
    Proven 2-state (up/down) rep counter.
    Tuned for frame_skip=3, scale=0.4 conditions.
    """

    def __init__(self):
        self.count = 0
        self.stage = None          # None, 'down', or 'up'
        self.rep_reports = []
        self.rep_index = 1
        self.rep_start_frame = None
        self.good_reps = 0
        self.bad_reps = 0
        self.all_feedback = collections.Counter()

        # Per-rep tracking
        self._start_knee_angle = None
        self._curr_min_knee = 999.0
        self._curr_max_knee = -999.0
        self._curr_min_torso = 999.0
        self._curr_valgus_bad = 0
        self._torso_bad_frames = 0
        self._valgus_bad_frames = 0
        self._down_frames = 0
        self._last_depth_for_ui = 0.0
        self._last_rep_end_frame = -100
        self._last_standing_knee = 170.0

        # Adaptive thresholds (start with defaults, may be calibrated)
        self._down_thresh = ANGLE_DOWN_THRESH
        self._up_thresh = ANGLE_UP_THRESH

        # Calibration
        self._cal_samples = []
        self._cal_done = False
        self._standing_angle = None

    def calibrate_standing(self, knee_angle, is_good_frame):
        """Collect standing angle samples for first ~10 good frames."""
        if self._cal_done:
            return
        if is_good_frame and knee_angle > 130:
            self._cal_samples.append(knee_angle)
        if len(self._cal_samples) >= 10:
            self._standing_angle = float(np.median(self._cal_samples))
            # Adjust thresholds based on actual standing angle
            new_down = self._standing_angle - 65   # e.g. 170-65=105
            new_up = self._standing_angle - 15     # e.g. 170-15=155
            # Clamp to sane range
            self._down_thresh = float(np.clip(new_down, 75, 120))
            self._up_thresh = float(np.clip(new_up, 140, 170))
            self._cal_done = True

    def _start_rep(self, frame_no, start_knee_angle):
        if frame_no - self._last_rep_end_frame < REP_DEBOUNCE_FRAMES:
            return False
        self.rep_start_frame = frame_no
        self._start_knee_angle = float(start_knee_angle)
        self._curr_min_knee = 999.0
        self._curr_max_knee = -999.0
        self._curr_min_torso = 999.0
        self._curr_valgus_bad = 0
        self._torso_bad_frames = 0
        self._valgus_bad_frames = 0
        self._down_frames = 0
        return True

    def _finish_rep(self, frame_no, score, feedback, extra=None):
        score_q = round(float(score) * 2) / 2.0
        if score_q >= GOOD_REP_MIN_SCORE:
            self.good_reps += 1
        else:
            self.bad_reps += 1
        for fb in (feedback or []):
            self.all_feedback[fb] += 1

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
        if extra:
            report.update(extra)
        self.rep_reports.append(report)
        self.rep_index += 1
        self.rep_start_frame = None
        self._start_knee_angle = None
        self._last_depth_for_ui = 0.0
        self._last_rep_end_frame = frame_no

    def evaluate_form(self):
        feedback = []
        score = 10.0
        start = self._start_knee_angle or 170
        min_k = self._curr_min_knee if self._curr_min_knee < 900 else start
        denom = max(10.0, start - PERFECT_MIN_KNEE)
        depth_pct = float(np.clip((start - min_k) / denom, 0, 1))

        if depth_pct < 0.6:
            feedback.append("Go deeper – aim for 90° knee angle"); score -= 3
        elif depth_pct < 0.8:
            feedback.append("Go a bit deeper"); score -= 1.5
        elif depth_pct < 0.9:
            feedback.append("Try to reach a bit more depth for full ROM"); score -= 0.5

        if self._torso_bad_frames >= TORSO_BAD_MIN_FRAMES:
            feedback.append("Keep your back straight"); score -= 2

        if self._curr_valgus_bad >= VALGUS_BAD_MIN_FRAMES:
            feedback.append("Avoid knee collapse"); score -= 2

        score = float(np.clip(score, 0, 10))
        if score < 10.0 and not feedback:
            feedback.append("Good rep, but there is still room for cleaner form")
        return score, feedback, depth_pct

    def update(self, knee_angle, torso_angle, valgus_ok_flag, frame_no):
        """
        Core 2-state rep counting.
        knee_angle < down_thresh → entering descent
        knee_angle > up_thresh AND was down → rep complete (if valid)
        """
        if knee_angle > self._up_thresh:
            self._last_standing_knee = max(self._last_standing_knee, float(knee_angle))

        # === ENTER DOWN ===
        if knee_angle < self._down_thresh:
            if self.stage != 'down':
                self.stage = 'down'
                if not self._start_rep(frame_no, self._last_standing_knee):
                    self.stage = 'up'
                    return
            self._down_frames += 1

        # === ENTER UP (potential rep complete) ===
        elif knee_angle > (self._up_thresh - 5) and self.stage == 'down':
            start_angle = self._start_knee_angle or 0
            min_knee = self._curr_min_knee if self._curr_min_knee < 900 else 0
            depth_delta = start_angle - min_knee
            moved_enough = depth_delta >= MIN_RANGE_DELTA_DEG
            enough_frames = (
                self._down_frames >= MIN_DOWN_FRAMES
                or (self._down_frames >= 1 and depth_delta >= FAST_REP_MIN_DEPTH_DELTA)
            )

            if enough_frames and moved_enough:
                score, fb, depth = self.evaluate_form()
                self.count += 1
                self._finish_rep(frame_no, score, fb, extra={"depth_pct": float(depth)})
                self._last_standing_knee = float(knee_angle)
            else:
                # Reset without counting
                self._last_depth_for_ui = 0.0
                self.rep_start_frame = None
                self._start_knee_angle = None
            self.stage = 'up'

        # === TRACK FORM DURING DOWN ===
        if self.stage == 'down' and self.rep_start_frame:
            self._curr_min_knee = min(self._curr_min_knee, knee_angle)
            self._curr_max_knee = max(self._curr_max_knee, knee_angle)
            self._curr_min_torso = min(self._curr_min_torso, torso_angle)

            if torso_angle < (TORSO_LEAN_MIN - TORSO_MARGIN_DEG):
                self._torso_bad_frames += 1
            else:
                self._torso_bad_frames = 0

            if not valgus_ok_flag:
                self._valgus_bad_frames += 1
                self._curr_valgus_bad += 1
            else:
                self._valgus_bad_frames = 0

            denom = max(10.0, self._start_knee_angle - PERFECT_MIN_KNEE)
            self._last_depth_for_ui = float(np.clip(
                (self._start_knee_angle - self._curr_min_knee) / denom, 0, 1))

    def depth_for_overlay(self):
        return float(self._last_depth_for_ui)

    def result(self):
        avg = np.mean([float(r["score"]) for r in self.rep_reports]) if self.rep_reports else 0.0
        ts = round(float(avg) * 2) / 2.0
        aggregated_feedback = list(self.all_feedback.keys())
        return {
            "squat_count": self.count,
            "technique_score": float(ts),
            "technique_score_display": _display_half(ts),
            "technique_label": _score_label(ts),
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "feedback": aggregated_feedback if aggregated_feedback else ["Great form! Keep it up 💪"],
            "reps": self.rep_reports,
        }


# ========================== TIPS ==========================

BULGARIAN_TIPS = [
    "Keep your front shin vertical at the bottom",
    "Drive through the front heel for power",
    "Brace your core before the descent",
    "Keep hips square – avoid rotation",
    "Control the eccentric; go down a bit slower",
    "Pause 1–2s at the bottom to build stability",
]

def choose_session_tip(counter):
    if counter.all_feedback.get("Avoid knee collapse", 0) >= 2:
        return "Track your knee over your toes"
    if counter.all_feedback.get("Keep your back straight", 0) >= 2:
        return "Brace your core and keep chest up"
    return BULGARIAN_TIPS[1]


# ========================== MAIN ANALYSIS ==========================

def run_bulgarian_analysis(video_path, frame_skip=1, scale=1.0,
                           output_path="analyzed_output.mp4",
                           feedback_path="feedback_summary.txt",
                           return_video=True,
                           fast_mode=None):
    if fast_mode is True:
        return_video = False
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open: {video_path}", "squat_count": 0,
                "technique_score": 0.0, "technique_score_display": "0",
                "technique_label": "N/A", "reps": []}

    counter = BulgarianRepCounter()
    frame_no = 0
    active_leg = None
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    ema = AngleEMA(alpha=EMA_ALPHA)
    lm_stab = LandmarkStabilizer()

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    # Early exit
    NOPOSE_STOP_FRAMES = int(NOPOSE_STOP_SEC * effective_fps)
    NO_MOVEMENT_STOP_FRAMES = int(NO_MOVEMENT_STOP_SEC * effective_fps)
    nopose_since_rep = 0
    no_movement_frames = 0

    # Live depth
    stand_knee_ema = None
    STAND_KNEE_ALPHA = 0.30

    # RT feedback
    RT_FB_HOLD_FRAMES = max(2, int(RT_FB_HOLD_SEC / dt))
    rt_fb_msg = None
    rt_fb_hold = 0

    # Walking filter
    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    movement_free_streak = 0

    def _euclid(a, b, norm):
        return math.hypot(a[0]-b[0], a[1]-b[1]) / max(1, norm)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_skip > 1 and (frame_no % frame_skip) != 0:
            continue
        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        h, w = frame.shape[:2]
        if return_video and out is None:
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        depth_live = 0.0

        if not results.pose_landmarks:
            if counter.count > 0:
                nopose_since_rep += 1
            else:
                nopose_since_rep = 0
            if counter.count > 0 and nopose_since_rep >= NOPOSE_STOP_FRAMES:
                break
            if rt_fb_hold > 0:
                rt_fb_hold -= 1
            no_movement_frames = 0
            stab_lms = lm_stab.stabilize(None)
        else:
            nopose_since_rep = 0
            lms = results.pose_landmarks.landmark

            if active_leg is None:
                active_leg = detect_active_leg(lms)
            side = "RIGHT" if active_leg == "right" else "LEFT"

            # --- Walking filter ---
            hip_px = (lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x * w,
                      lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y * h)
            l_ankle_px = (lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                          lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h)
            r_ankle_px = (lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                          lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
            if prev_hip is None:
                prev_hip, prev_la, prev_ra = hip_px, l_ankle_px, r_ankle_px
            norm = max(h, w)
            hip_vel = _euclid(hip_px, prev_hip, norm)
            an_vel = max(_euclid(l_ankle_px, prev_la, norm), _euclid(r_ankle_px, prev_ra, norm))
            hip_vel_ema = MOTION_EMA_ALPHA * hip_vel + (1 - MOTION_EMA_ALPHA) * hip_vel_ema
            ankle_vel_ema = MOTION_EMA_ALPHA * an_vel + (1 - MOTION_EMA_ALPHA) * ankle_vel_ema
            prev_hip, prev_la, prev_ra = hip_px, l_ankle_px, r_ankle_px

            movement_block = (hip_vel_ema > HIP_VEL_THRESH_PCT) or (ankle_vel_ema > ANKLE_VEL_THRESH_PCT)
            if movement_block:
                movement_free_streak = 0
                no_movement_frames = 0
            else:
                movement_free_streak = min(MOVEMENT_CLEAR_FRAMES, movement_free_streak + 1)
                no_movement_frames += 1

            if counter.count > 0 and no_movement_frames >= NO_MOVEMENT_STOP_FRAMES:
                break

            # --- Angles ---
            hip = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_HIP").value, w, h)
            knee = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value, w, h)
            ankle = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value, w, h)
            shoulder = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value, w, h)
            knee_angle_raw = angle_3pt(hip, knee, ankle)
            torso_angle_raw = angle_3pt(shoulder, hip, knee)
            knee_angle, torso_angle = ema.update(knee_angle_raw, torso_angle_raw)
            v_ok = check_valgus(lms, side, VALGUS_X_TOL)

            # --- Calibration (first ~10 stable frames) ---
            vis = getattr(lms[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value], 'visibility', 0) or 0
            counter.calibrate_standing(knee_angle, vis > 0.5)

            # --- Update counter ---
            # IMPORTANT: do not block rep counting by the walking filter.
            # In real videos, hip/ankle velocity can stay above threshold even during valid reps,
            # which previously led to never calling `counter.update(...)` and reporting 0 reps.
            counter.update(knee_angle, torso_angle, v_ok, frame_no)

            # --- Live depth ---
            if knee_angle > counter._up_thresh - 5 and movement_free_streak >= 1:
                stand_knee_ema = knee_angle if stand_knee_ema is None else (
                    STAND_KNEE_ALPHA * knee_angle + (1 - STAND_KNEE_ALPHA) * stand_knee_ema)
            if stand_knee_ema is not None:
                denom_live = max(10.0, stand_knee_ema - PERFECT_MIN_KNEE)
                depth_live = float(np.clip((stand_knee_ema - knee_angle) / denom_live, 0, 1))
            else:
                depth_live = counter.depth_for_overlay()

            # --- RT feedback ---
            live_msgs = []
            if counter.stage == 'down':
                if counter._torso_bad_frames >= TORSO_BAD_MIN_FRAMES:
                    live_msgs.append("Keep your back straight")
                if counter._valgus_bad_frames >= VALGUS_BAD_MIN_FRAMES:
                    live_msgs.append("Avoid knee collapse")
            new_msg = " | ".join(live_msgs) if live_msgs else None
            if new_msg:
                if new_msg != rt_fb_msg:
                    rt_fb_msg = new_msg; rt_fb_hold = RT_FB_HOLD_FRAMES
                else:
                    rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold > 0: rt_fb_hold -= 1
                if rt_fb_hold == 0: rt_fb_msg = None

            stab_lms = lm_stab.stabilize(lms)

        # === Draw ===
        if return_video:
            if 'stab_lms' in locals() and stab_lms:
                frame = draw_body_only(frame, stab_lms)
            frame = draw_overlay(frame, reps=counter.count,
                                 feedback=(rt_fb_msg if rt_fb_hold > 0 else None),
                                 depth_pct=depth_live)
            if out is not None:
                out.write(frame)

    pose.close()
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    result = counter.result()

    # Session tip
    session_tip = choose_session_tip(counter)
    result["tips"] = [session_tip]
    result["form_tip"] = session_tip

    # Feedback file
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {result['squat_count']}\n")
            f.write(f"Technique Score: {result['technique_score_display']} / 10  ({result['technique_label']})\n")
            f.write(f"Form Tip: {session_tip}\n")
            if result.get("feedback"):
                f.write("Feedback:\n")
                for fb in result["feedback"]:
                    f.write(f"- {fb}\n")
    except:
        pass

    # ffmpeg encode
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
        except:
            final_path = output_path if os.path.isfile(output_path) else ""
        if not os.path.isfile(final_path) and os.path.isfile(output_path):
            final_path = output_path

    result["video_path"] = final_path if return_video else ""
    result["feedback_path"] = feedback_path
    return result


# Backward compatibility
run_analysis = run_bulgarian_analysis

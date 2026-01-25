# -*- coding: utf-8 -*-
# bulgarian_split_squat_analysis.py
# Overlay אחיד כמו בסקוואט + תאימות JSON + Freeze-on-drop ליציבות שלד (ללא לג).
# - לוגיקת ספירה/ספים נשארת כמו קודם.
# - technique_score (double) לשימוש חישובי + technique_score_display (string) להצגה.
# - בכל חזרה: score (double) + score_display (string).
# - technique_label מילולי; form tip אחד לסשן.
# - חסימת ספירה בזמן הליכה (soft-start + דרישת שקט לסיום) כמו בסקוואט.
# - פלט וידאו אמין עם ffmpeg (+faststart).

import os, math, subprocess, collections
import cv2, numpy as np
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== פרמטרים ללוגיקת בולגרי (כמו אצלך) =====================
ANGLE_DOWN_THRESH   = 95      # כניסה לירידה
ANGLE_UP_THRESH     = 160     # יציאה לעלייה
MIN_RANGE_DELTA_DEG = 12      # שינוי מינימלי Top->Bottom כדי לספור
MIN_DOWN_FRAMES     = 5       # מינ' פריימים בירידה

GOOD_REP_MIN_SCORE  = 8.0
TORSO_LEAN_MIN      = 135     # גב פחות מזה -> הערה
VALGUS_X_TOL        = 0.03    # סטייה מותרת X בין ברך לקרסול
PERFECT_MIN_KNEE    = 70      # יעד לנרמול עומק

# יציבות חישובית (לוגיקה)
EMA_ALPHA             = 0.6
TORSO_MARGIN_DEG      = 3
TORSO_BAD_MIN_FRAMES  = 4
VALGUS_BAD_MIN_FRAMES = 3
REP_DEBOUNCE_FRAMES   = 6

# ===================== חסימת תנועה גלובלית (כמו בסקוואט) =====================
HIP_VEL_THRESH_PCT    = 0.014
ANKLE_VEL_THRESH_PCT  = 0.017
MOTION_EMA_ALPHA      = 0.65
MOVEMENT_CLEAR_FRAMES = 2

# ===================== סטייל אחיד כמו בסקוואט =====================
BAR_BG_ALPHA         = 0.55
REPS_FONT_SIZE       = 28
FEEDBACK_FONT_SIZE   = 22
DEPTH_LABEL_FONT_SIZE= 14
DEPTH_PCT_FONT_SIZE  = 18
FONT_PATH            = "Roboto-VariableFont_wdth,wght.ttf"
RT_FB_HOLD_SEC       = 0.8

DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)   # BGR
DEPTH_RING_BG        = (70, 70, 70)

mp_pose = mp.solutions.pose

# ===================== פונטים =====================
def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

REPS_FONT        = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

# ===================== BODY-ONLY (לצייר שלד בלי פנים) =====================
_FACE_LMS = {
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
    mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
    mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
}
_BODY_CONNECTIONS = tuple(
    (a, b) for (a, b) in mp_pose.POSE_CONNECTIONS
    if a not in _FACE_LMS and b not in _FACE_LMS
)
_BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

def draw_body_only(frame, landmarks, color=(255,255,255)):
    h, w = frame.shape[:2]
    for a, b in _BODY_CONNECTIONS:
        pa, pb = landmarks[a], landmarks[b]
        ax, ay = int(pa.x * w), int(pa.y * h)
        bx, by = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = landmarks[i]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
    return frame

# ===================== LandmarkStabilizer (ללא החלקה/לג) =====================
class LandmarkStabilizer:
    """Freeze-on-drop: כשיש זיהוי איכותי מציגים מיד; בעת איבוד איכות מחזיקים שלד אחרון לכמה פריימים ואז מסתירים."""
    def __init__(self, quality_thr=0.55, min_fraction=0.6, jump_thr=0.12, max_hold=6, body_points=None):
        # לא נסמוך על _BODY_POINTS חיצוני – נבנה גוף-בלבד מתוך mediapipe אם צריך:
        if body_points is None:
            FACE_LMS = _FACE_LMS
            body_connections = _BODY_CONNECTIONS
            self.body_points = tuple(sorted({i for (a,b) in body_connections for i in (a,b)}))
        else:
            self.body_points = tuple(body_points)
        self.quality_thr = float(quality_thr)
        self.min_fraction = float(min_fraction)
        self.jump_thr = float(jump_thr)
        self.max_hold = int(max_hold)
        self.last_good = None  # רשימת נק' עם x,y
        self.hold = 0

    def _quality(self, lms):
        ok = 0; total = 0
        for i in self.body_points:
            if i >= len(lms): continue
            total += 1
            vis = getattr(lms[i], 'visibility', 1.0) or 0.0
            if vis >= self.quality_thr:
                ok += 1
        return (ok / max(1, total)) if total else 1.0

    def _avg_disp(self, lms):
        if not self.last_good: return 0.0
        n = min(len(self.last_good), len(lms))
        if n == 0: return 0.0
        s = 0.0; c = 0
        for i in self.body_points:
            if i >= n: continue
            dx = float(lms[i].x) - float(self.last_good[i].x)
            dy = float(lms[i].y) - float(self.last_good[i].y)
            s += (dx*dx + dy*dy) ** 0.5
            c += 1
        return (s / max(1, c))

    def _copy_xy(self, lms):
        out = []
        for i in range(len(lms)):
            out.append(type('P', (), {'x': float(lms[i].x), 'y': float(lms[i].y)}))
        return out

    def stabilize(self, lms):
        # אין שלד: החזק אחרון לכמה פריימים ואז הסתר
        if lms is None:
            if self.last_good is not None and self.hold < self.max_hold:
                self.hold += 1
                return self.last_good
            self.hold = 0
            return None

        # בדיקת איכות
        frac = self._quality(lms)
        if frac < self.min_fraction:
            if self.last_good is not None and self.hold < self.max_hold:
                self.hold += 1
                return self.last_good
            self.hold = 0
            return None

        # נטרול קפיצה חריגה
        avg_d = self._avg_disp(lms)
        if avg_d > self.jump_thr and frac < 0.8 and self.last_good is not None:
            if self.hold < self.max_hold:
                self.hold += 1
                return self.last_good
            self.hold = 0
            # נצמד מידית מכאן כדי לא ליצור לג

        # הצמדה מידית (ללא החלקה)
        self.last_good = self._copy_xy(lms)
        self.hold = 0
        return self.last_good

# ===================== עזרי ציור לאוברליי =====================
def draw_depth_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius = int(radius); thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def _wrap_two_lines(draw, text, font, max_width):
    words = text.split()
    if not words: return [""]
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
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
    return lines

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    h, w, _ = frame.shape
    # Reps box בפינה (0,0)
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    inner_pad_x, inner_pad_y = 10, 6
    text_w = draw.textlength(reps_text, font=REPS_FONT)
    text_h = REPS_FONT_SIZE
    x0, y0 = 0, 0
    x1 = int(text_w + 2*inner_pad_x); y1 = int(text_h + 2*inner_pad_y)
    top = frame.copy(); cv2.rectangle(top, (x0, y0), (x1, y1), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((x0 + inner_pad_x, y0 + inner_pad_y - 1), reps_text, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Donut (ימין-עליון)
    ref_h = max(int(h * 0.06), int(REPS_FONT_SIZE * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)
    frame = draw_depth_donut(frame, (cx, cy), radius, thick, float(np.clip(depth_pct,0,1)))

    pil  = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"; pct_txt = f"{int(float(np.clip(depth_pct,0,1))*100)}%"
    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt,   font=DEPTH_PCT_FONT)
    gap     = max(2, int(radius * 0.10))
    base_y  = cy - (DEPTH_LABEL_FONT_SIZE + gap + DEPTH_PCT_FONT_SIZE) // 2
    draw.text((cx - int(label_w // 2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pct_w // 2), base_y + DEPTH_LABEL_FONT_SIZE + gap), pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # פידבק תחתון (wrap עד 2 שורות, safe area)
    if feedback:
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        safe_margin = max(6, int(h * 0.02)); pad_x, pad_y, line_gap = 12, 8, 4
        max_text_w = int(w - 2*pad_x - 20)
        lines = _wrap_two_lines(draw_fb, feedback, FEEDBACK_FONT, max_text_w)
        line_h = FEEDBACK_FONT_SIZE + 6
        block_h = (2*pad_y) + len(lines)*line_h + (len(lines)-1)*line_gap
        y0 = max(0, h - safe_margin - block_h); y1 = h - safe_margin
        over = frame.copy(); cv2.rectangle(over, (0, y0), (w, y1), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + pad_y
        for ln in lines:
            tw = draw_fb.textlength(ln, font=FEEDBACK_FONT)
            tx = max(pad_x, (w - int(tw)) // 2)
            draw_fb.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255,255,255))
            ty += line_h + line_gap
        frame = np.array(pil_fb)
    return frame

# ===================== עזר/גאומטריה =====================
def calculate_angle(a, b, c):
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

def valgus_ok(landmarks, side):
    knee_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x
    ankle_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x
    return not (knee_x < ankle_x - VALGUS_X_TOL)

# ===================== החלקת זוויות ללוגיקה =====================
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

# ===================== תצוגה/תיוג =====================
def score_label(s):
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    if abs(q - round(q)) < 1e-9:
        return str(int(round(q)))  # "10"
    return f"{q:.1f}"            # "9.5"

# ===================== מונה חזרות (לוגיקה מקורית) =====================
class BulgarianRepCounter:
    def __init__(self):
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
        score_q = round(float(score) * 2) / 2.0  # double
        if score_q >= GOOD_REP_MIN_SCORE: self.good_reps += 1
        else: self.bad_reps += 1
        for fb in (feedback or []):
            self.all_feedback[fb] += 1
        report = {
            "rep_index": self.rep_index,
            "score": float(score_q),                      # double לתאימות
            "score_display": display_half_str(score_q),   # תצוגה
            "feedback": feedback or [],
            "start_frame": self.rep_start_frame or 0,
            "end_frame": frame_no,
            "start_knee_angle": round(float(self._start_knee_angle or 0), 2),
            "min_knee_angle": round(self._curr_min_knee, 2),
            "max_knee_angle": round(self._curr_max_knee, 2),
            "torso_min_angle": round(self._curr_min_torso, 2)
        }
        if extra: report.update(extra)
        self.rep_reports.append(report)
        self.rep_index += 1
        self.rep_start_frame = None
        self._start_knee_angle = None
        self._last_depth_for_ui = 0.0
        self._last_rep_end_frame = frame_no

    def evaluate_form(self, start_knee_angle, min_knee_angle, min_torso_angle, valgus_bad_frames):
        feedback = []
        score = 10.0
        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE))
        depth_pct = np.clip((start_knee_angle - min_knee_angle) / denom, 0, 1)
        if self._torso_bad_frames >= TORSO_BAD_MIN_FRAMES:
            feedback.append("Keep your back straight"); score -= 2
        if valgus_bad_frames >= VALGUS_BAD_MIN_FRAMES:
            feedback.append("Avoid knee collapse");     score -= 2
        if depth_pct < 0.8:
            feedback.append("Go a bit deeper");         score -= 1
        return score, feedback, float(depth_pct)

    def update(self, knee_angle, torso_angle, valgus_ok_flag, frame_no):
        if knee_angle < ANGLE_DOWN_THRESH:
            if self.stage != 'down':
                self.stage = 'down'
                started = self._start_rep(frame_no, knee_angle)
                if not started:
                    self.stage = 'up'; return
            self._down_frames += 1
        elif knee_angle > ANGLE_UP_THRESH and self.stage == 'down':
            depth_delta = (self._start_knee_angle or 0) - (self._curr_min_knee or 0)
            did_move_enough = depth_delta >= MIN_RANGE_DELTA_DEG
            if self._down_frames >= MIN_DOWN_FRAMES and did_move_enough:
                score, fb, depth = self.evaluate_form(float(self._start_knee_angle or knee_angle),
                                                      float(self._curr_min_knee or knee_angle),
                                                      float(self._curr_min_torso or 180.0),
                                                      self._curr_valgus_bad)
                self.count += 1
                self._finish_rep(frame_no, score, fb, extra={"depth_pct": float(depth)})
            else:
                self._last_depth_for_ui = 0.0
                self.rep_start_frame = None
                self._start_knee_angle = None
            self.stage = 'up'

        if self.stage == 'down' and self.rep_start_frame:
            self._curr_min_knee = min(self._curr_min_knee, knee_angle)
            self._curr_max_knee = max(self._curr_max_knee, knee_angle)
            self._curr_min_torso = min(self._curr_min_torso, torso_angle)
            if torso_angle < (TORSO_LEAN_MIN - TORSO_MARGIN_DEG): self._torso_bad_frames += 1
            else: self._torso_bad_frames = 0
            if not valgus_ok_flag:
                self._valgus_bad_frames += 1; self._curr_valgus_bad += 1
            else:
                self._valgus_bad_frames = 0
            denom = max(10.0, (self._start_knee_angle - PERFECT_MIN_KNEE))
            self._last_depth_for_ui = float(np.clip((self._start_knee_angle - self._curr_min_knee) / denom, 0, 1))

    def depth_for_overlay(self):
        return float(self._last_depth_for_ui)

    def result(self):
        avg = np.mean([float(r["score"]) for r in self.rep_reports]) if self.rep_reports else 0.0
        technique_score = round(float(avg) * 2) / 2.0  # double
        return {
            "squat_count": self.count,
            "technique_score": float(technique_score),                 # double לתאימות
            "technique_score_display": display_half_str(technique_score),  # string להצגה
            "technique_label": score_label(technique_score),
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "feedback": list(self.all_feedback.elements()) if self.bad_reps > 0 else ["Great form! Keep it up 💪"],
            "reps": self.rep_reports
        }

# ===================== טיפים לבולגרי (לא משפיע על הציון) =====================
BULGARIAN_TIPS = [
    "Keep your front shin vertical at the bottom",
    "Drive through the front heel for power",
    "Brace your core before the descent",
    "Keep hips square – avoid rotation",
    "Control the eccentric; go down a bit slower",
    "Pause 1–2s at the bottom to build stability",
]

def choose_session_tip(counter: BulgarianRepCounter):
    if counter.all_feedback.get("Avoid knee collapse", 0) >= 2:
        return "Track your knee over your toes"
    if counter.all_feedback.get("Keep your back straight", 0) >= 2:
        return "Brace your core and keep chest up"
    return BULGARIAN_TIPS[1]  # Drive through the front heel

# ===================== ריצה =====================
def run_bulgarian_analysis(video_path, frame_skip=1, scale=1.0,
                           output_path="analyzed_output.mp4",
                           feedback_path="feedback_summary.txt"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    counter = BulgarianRepCounter()
    frame_no = 0
    active_leg = None
    out = None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    ema = AngleEMA(alpha=EMA_ALPHA)
    lm_stab = LandmarkStabilizer()

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
dt = 1.0 / float(effective_fps)

# === EXIT CONDITIONS (כמו בפול-אפ) ===
NOPOSE_STOP_SEC = 1.2
NOPOSE_STOP_FRAMES = int(NOPOSE_STOP_SEC * effective_fps)
nopose_frames_since_any_rep = 0

NO_MOVEMENT_STOP_SEC = 1.3
NO_MOVEMENT_STOP_FRAMES = int(NO_MOVEMENT_STOP_SEC * effective_fps)
no_movement_frames = 0


    # עומק "לייב" דו-כיווני (גם בירידה וגם בעלייה)
    stand_knee_ema = None
    STAND_KNEE_ALPHA = 0.30

    # RT feedback hold (~0.8s)
    RT_FB_HOLD_FRAMES = max(2, int(RT_FB_HOLD_SEC / dt))
    rt_fb_msg = None
    rt_fb_hold = 0

    # תנועה גלובלית
    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    movement_free_streak = 0

    def _euclid(a, b, norm):
        return math.hypot(a[0]-b[0], a[1]-b[1]) / max(1, norm)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1
        if frame_skip > 1 and (frame_no % frame_skip) != 0: continue
        if scale != 1.0: frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        h, w = frame.shape[:2]
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        depth_live = 0.0

        if not results.pose_landmarks:
    if counter.count > 0:
        nopose_frames_since_any_rep += 1
    else:
        nopose_frames_since_any_rep = 0

    if counter.count > 0 and nopose_frames_since_any_rep >= NOPOSE_STOP_FRAMES:
        break

    if rt_fb_hold > 0:
        rt_fb_hold -= 1

    stab_lms = lm_stab.stabilize(None)
else:
    nopose_frames_since_any_rep = 0
    lms = results.pose_landmarks.landmark
            if active_leg is None:
                active_leg = detect_active_leg(lms)
            side = "RIGHT" if active_leg == "right" else "LEFT"

            # === תנועה גלובלית (למנוע ספירה בזמן הליכה) ===
            hip_px    = (lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x * w,
                         lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y * h)
            l_ankle_px= (lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                         lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h)
            r_ankle_px= (lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                         lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
            if prev_hip is None:
                prev_hip, prev_la, prev_ra = hip_px, l_ankle_px, r_ankle_px
            norm = max(h, w)
            hip_vel = _euclid(hip_px, prev_hip, norm)
            an_vel  = max(_euclid(l_ankle_px, prev_la, norm), _euclid(r_ankle_px, prev_ra, norm))
            hip_vel_ema   = MOTION_EMA_ALPHA*hip_vel + (1-MOTION_EMA_ALPHA)*hip_vel_ema
            ankle_vel_ema = MOTION_EMA_ALPHA*an_vel  + (1-MOTION_EMA_ALPHA)*ankle_vel_ema
            prev_hip, prev_la, prev_ra = hip_px, l_ankle_px, r_ankle_px
            movement_block = (hip_vel_ema > HIP_VEL_THRESH_PCT) or (ankle_vel_ema > ANKLE_VEL_THRESH_PCT)
            if movement_block: movement_free_streak = 0
            else:              movement_free_streak = min(MOVEMENT_CLEAR_FRAMES, movement_free_streak + 1)
                ifif movement_block:
    movement_free_streak = 0
    no_movement_frames = 0
else:
    movement_free_streak = min(MOVEMENT_CLEAR_FRAMES, movement_free_streak + 1)
    no_movement_frames += 1

if counter.count > 0 and no_movement_frames >= NO_MOVEMENT_STOP_FRAMES:
    break



            # === זוויות ===
            hip = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_HIP").value, w, h)
            knee = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value, w, h)
            ankle= lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value, w, h)
            shoulder = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value, w, h)
            knee_angle_raw = calculate_angle(hip, knee, ankle)
            torso_angle_raw= calculate_angle(shoulder, hip, knee)
            knee_angle, torso_angle = ema.update(knee_angle_raw, torso_angle_raw)
            v_ok = valgus_ok(lms, side)

            # === soft-start כמו בסקוואט: מתחילים רק כששקט יחסית ===
            soft_start_ok = (hip_vel_ema < HIP_VEL_THRESH_PCT * 1.25) and (ankle_vel_ema < ANKLE_VEL_THRESH_PCT * 1.25)
            if (knee_angle < ANGLE_DOWN_THRESH) and (counter.stage != 'down') and soft_start_ok:
                pass  # הסטארט בפועל מתבצע במחלקה; אין שינוי לוגי

            # === עדכון מונה (לוגיקה מקורית) + תנאי שקט לסיום ===
            prev_stage = counter.stage
            counter.update(knee_angle, torso_angle, v_ok, frame_no)
            if prev_stage == 'down' and counter.stage == 'up' and movement_free_streak < MOVEMENT_CLEAR_FRAMES:
                counter.stage = 'down'  # נחכה לשקט כדי לסיים חזרה

            # === עומק "לייב" דו-כיווני ===
            if knee_angle > ANGLE_UP_THRESH - 3 and movement_free_streak >= 1:
                stand_knee_ema = knee_angle if stand_knee_ema is None else (STAND_KNEE_ALPHA*knee_angle + (1-STAND_KNEE_ALPHA)*stand_knee_ema)
            if stand_knee_ema is not None:
                denom_live = max(10.0, (stand_knee_ema - PERFECT_MIN_KNEE))
                depth_live = float(np.clip((stand_knee_ema - knee_angle) / denom_live, 0, 1))
            else:
                depth_live = counter.depth_for_overlay()

            # === RT feedback עם HOLD ~0.8s (בלי לשנות לוגיקה בסיסית) ===
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

            # === ייצוב שלד לציור בלבד (ללא לג) ===
            stab_lms = lm_stab.stabilize(lms)

        # === ציור ===
        if 'stab_lms' in locals() and stab_lms:
            frame = draw_body_only(frame, stab_lms)
        frame = draw_overlay(frame, reps=counter.count, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
        out.write(frame)

    pose.close(); cap.release();
    if out: out.release()
    cv2.destroyAllWindows()

    result = counter.result()

    # הוספת form tip אחד (לא משפיע על ציון)
    session_tip = choose_session_tip(counter)
    result["tips"] = [session_tip]
    result["form_tip"] = session_tip

    # קובץ תקציר
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {result['squat_count']}\n")
            f.write(f"Technique Score: {result['technique_score_display']} / 10  ({result['technique_label']})\n")
            f.write(f"Form Tip: {session_tip}\n")
            if result.get("feedback"):
                f.write("Feedback:\n")
                for fb in result["feedback"]:
                    f.write(f"- {fb}\n")
    except Exception:
        pass

    # קידוד faststart אמין + fallback
    encoded_path = output_path.replace('.mp4', '_encoded.mp4')
    try:
        subprocess.run([
            'ffmpeg','-y','-i', output_path,
            '-c:v','libx264','-preset','fast','-movflags','+faststart','-pix_fmt','yuv420p',
            encoded_path
        ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final_path = encoded_path if os.path.isfile(encoded_path) else output_path
    except Exception:
        final_path = output_path

    if not os.path.isfile(final_path) and os.path.isfile(output_path):
        final_path = output_path

    result["video_path"] = final_path
    result["feedback_path"] = feedback_path
    return result

# תאימות
run_analysis = run_bulgarian_analysis

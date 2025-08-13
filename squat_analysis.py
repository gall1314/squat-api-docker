# combined_analysis.py  (אפשר גם לשמור כ-squat_analysis.py)
import os
import cv2
import math
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / FONTS =====================
TOP_BAR_MIN_FRAC     = 0.065    # גיבוי אם המסך ענק
BAR_BG_ALPHA         = 0.55

# דונאט – גרסה מוגדלת ולא נחתכת
DONUT_RADIUS_SCALE   = 0.90     # היה 0.70
DONUT_THICKNESS_FRAC = 0.32     # היה 0.28
DEPTH_COLOR          = (40, 200, 80)
DEPTH_RING_BG        = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE   = 18

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

REPS_FONT        = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

mp_pose = mp.solutions.pose

# ===================== TEXT WRAP (BOTTOM FEEDBACK 2 LINES) =====================
def wrap_text_to_width(draw, text, font, max_width, max_lines=2, ellipsis="…"):
    words = (text or "").split()
    if not words: return []
    lines, cur = [], ""
    i = 0
    while i < len(words):
        test = (cur + " " + words[i]).strip()
        if draw.textlength(test, font=font) <= max_width:
            cur = test
            i += 1
        else:
            if cur:
                lines.append(cur)
                cur = ""
            else:
                # מילה בודדת ארוכה מדי — חתוך אותה
                word = words[i]
                while word and draw.textlength(word + ellipsis, font=font) > max_width:
                    word = word[:-1]
                lines.append((word + ellipsis) if word else "")
                i += 1
            if len(lines) == max_lines - 1:
                rest = " ".join(words[i:])
                while rest and draw.textlength(rest + ellipsis, font=font) > max_width:
                    rest = rest[:-1]
                if rest: lines.append(rest + ellipsis)
                return lines
    if cur: lines.append(cur)
    return lines[:max_lines]

# ===================== OVERLAY =====================
def draw_depth_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius   = int(radius)
    thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """
    - פס עליון שקוף קטן מאחורי טקסט Reps בלבד (לא לכל הרוחב).
    - דונאט DEPTH מוגדל ומונמך כך שלא ייחתך.
    - פס תחתון עם עיטוף עד 2 שורות (פונט קבוע).
    """
    h, w, _ = frame.shape

    # ----- TOP: רק מאחורי הטקסט -----
    # גובה פס נגזר מגודל הפונט כדי שלא ייחתך
    bar_h_top = max(int(h * TOP_BAR_MIN_FRAC), int(REPS_FONT.size * 1.8))

    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    text_w = draw.textlength(reps_text, font=REPS_FONT)
    text_h = REPS_FONT.size

    pad_x, pad_y = 12, 8
    x0 = 12
    y0 = max(2, int(0.5 * (bar_h_top - text_h) - 2))
    x1 = min(int(x0 + text_w + 2 * pad_x), w - 1)
    y1 = int(y0 + text_h + 2 * pad_y)

    top = frame.copy()
    cv2.rectangle(top, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)

    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    draw.text((x0 + pad_x, y0 + pad_y - 1), reps_text, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # ----- DONUT: גדול יותר ומונמך -----
    radius = int(bar_h_top * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 14
    cx = w - margin - radius
    cy = max(int(bar_h_top * 0.75), radius + thick // 2 + 2)
    cy = min(cy, max(bar_h_top + radius, radius + thick // 2 + 2))
    frame = draw_depth_donut(frame, (cx, cy), radius, thick, float(np.clip(depth_pct,0,1)))

    pil  = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"
    pct_txt   = f"{int(float(np.clip(depth_pct,0,1))*100)}%"
    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt, font=DEPTH_PCT_FONT)
    gap     = max(2, int(radius * 0.10))
    block_h = DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size
    base_y  = cy - block_h // 2
    draw.text((cx - int(label_w // 2), base_y),                    label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pct_w   // 2), base_y + DEPTH_LABEL_FONT.size + gap),
              pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # ----- BOTTOM FEEDBACK: עד 2 שורות -----
    if feedback:
        line_h = FEEDBACK_FONT.size + 6
        bottom_h = max(int(h * 0.08), 2 * line_h + 14)
        over = frame.copy()
        cv2.rectangle(over, (0, h - bottom_h), (w, h), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)

        pil2 = Image.fromarray(frame)
        draw2 = ImageDraw.Draw(pil2)
        side_pad = 18
        max_w = w - 2 * side_pad
        lines = wrap_text_to_width(draw2, feedback, FEEDBACK_FONT, max_w, max_lines=2)
        total_h = len(lines) * line_h
        y = h - bottom_h + (bottom_h - total_h) // 2
        for line in lines:
            tw = draw2.textlength(line, font=FEEDBACK_FONT)
            x  = (w - int(tw)) // 2
            draw2.text((x, y), line, font=FEEDBACK_FONT, fill=(255,255,255))
            y += line_h
        frame = np.array(pil2)

    return frame

# ===================== BODY-ONLY SKELETON =====================
# נסנן חיבורים ונקודות פנים: NOSE/EYES/EARS/MOUTH
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
# אינדקסים של נקודות גוף שברצוננו להציג כנקודות (ללא פני/ראש)
_BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

def draw_body_only(frame, landmarks, color=(255,255,255)):
    """
    מצייר רק חיבורים ונקודות של הגוף (ללא ראש/פנים).
    """
    h, w = frame.shape[:2]
    # חיבורים
    for a, b in _BODY_CONNECTIONS:
        pa = landmarks[a]; pb = landmarks[b]
        ax, ay = int(pa.x * w), int(pa.y * h)
        bx, by = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    # נקודות
    for i in _BODY_POINTS:
        p = landmarks[i]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
    return frame

# ===================== GEOMETRY =====================
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

# ===================== BULGARIAN =====================
ANGLE_DOWN_THRESH   = 95
ANGLE_UP_THRESH     = 160
MIN_RANGE_DELTA_DEG = 12
MIN_DOWN_FRAMES     = 5
GOOD_REP_MIN_SCORE  = 8.0
TORSO_LEAN_MIN      = 135
VALGUS_X_TOL        = 0.03
PERFECT_MIN_KNEE_BG = 70
EMA_ALPHA           = 0.6
TORSO_MARGIN_DEG    = 3
TORSO_BAD_MIN_FRAMES  = 4
VALGUS_BAD_MIN_FRAMES = 3
REP_DEBOUNCE_FRAMES   = 6

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
            self.knee  = a * ka + (1 - a) * self.knee
            self.torso = a * ta + (1 - a) * self.torso
        return self.knee, self.torso

def valgus_ok(landmarks, side):
    knee_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x
    ankle_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x
    return not (knee_x < ankle_x - VALGUS_X_TOL)

class BulgarianRepCounter:
    def __init__(self):
        self.count = 0
        self.stage = None
        self.rep_reports = []
        self.rep_index = 1
        self.rep_start_frame = None
        self.good_reps = 0
        self.bad_reps = 0
        self.all_feedback = set()
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
        if score >= GOOD_REP_MIN_SCORE:
            self.good_reps += 1
        else:
            self.bad_reps += 1
            if feedback: self.all_feedback.update(feedback)
        report = {
            "rep_index": self.rep_index,
            "score": round(score, 1),
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
        feedback, score = [], 10.0
        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE_BG))
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
                if not self._start_rep(frame_no, knee_angle):
                    self.stage = 'up'; return
            self._down_frames += 1
        elif knee_angle > ANGLE_UP_THRESH and self.stage == 'down':
            depth_delta = (self._start_knee_angle or 0) - (self._curr_min_knee or 0)
            if self._down_frames >= MIN_DOWN_FRAMES and depth_delta >= MIN_RANGE_DELTA_DEG:
                score, fb, depth = self.evaluate_form(
                    float(self._start_knee_angle or knee_angle),
                    float(self._curr_min_knee or knee_angle),
                    float(self._curr_min_torso or 180.0),
                    self._curr_valgus_bad
                )
                self.count += 1
                self._finish_rep(frame_no, score, fb, extra={"depth_pct": depth})
            else:
                self._last_depth_for_ui = 0.0
                self.rep_start_frame = None
                self._start_knee_angle = None
            self.stage = 'up'

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
            denom = max(10.0, (self._start_knee_angle - PERFECT_MIN_KNEE_BG))
            self._last_depth_for_ui = float(np.clip(
                (self._start_knee_angle - self._curr_min_knee) / denom, 0, 1
            ))

    def depth_for_overlay(self):
        return float(self._last_depth_for_ui)

    def result(self):
        avg = np.mean([r["score"] for r in self.rep_reports]) if self.rep_reports else 0.0
        technique_score = round(round(avg * 2) / 2, 2)
        return {
            "squat_count": self.count,
            "technique_score": technique_score if self.count else 0.0,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "feedback": list(self.all_feedback) if self.bad_reps > 0 else ["Great form! Keep it up 💪"],
            "reps": self.rep_reports
        }

def run_bulgarian_analysis(video_path, frame_skip=1, scale=1.0,
                           output_path="analyzed_output.mp4",
                           feedback_path="feedback_summary.txt"):
    cap = cv2.VideoCapture(video_path)
    counter = BulgarianRepCounter()
    frame_no = 0
    active_leg = None
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    ema = AngleEMA(alpha=EMA_ALPHA)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))

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
        if not results.pose_landmarks:
            frame = draw_overlay(frame, reps=counter.count, feedback=None, depth_pct=0.0)
            out.write(frame); continue

        landmarks = results.pose_landmarks.landmark
        if active_leg is None:
            active_leg = detect_active_leg(landmarks)

        side = "RIGHT" if active_leg == "right" else "LEFT"
        hip = lm_xy(landmarks, getattr(mp_pose.PoseLandmark, f"{side}_HIP").value, w, h)
        knee = lm_xy(landmarks, getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value, w, h)
        ankle = lm_xy(landmarks, getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value, w, h)
        shoulder = lm_xy(landmarks, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value, w, h)

        knee_angle_raw = calculate_angle(hip, knee, ankle)
        torso_angle_raw = calculate_angle(shoulder, hip, knee)
        knee_angle, torso_angle = ema.update(knee_angle_raw, torso_angle_raw)
        v_ok = valgus_ok(landmarks, side)

        counter.update(knee_angle, torso_angle, v_ok, frame_no)

        # שלד – גוף בלבד
        frame = draw_body_only(frame, landmarks)

        feedbacks = []
        if counter.stage == "down":
            if counter._torso_bad_frames >= TORSO_BAD_MIN_FRAMES:
                feedbacks.append("Keep your back straight")
            if counter._valgus_bad_frames >= VALGUS_BAD_MIN_FRAMES:
                feedbacks.append("Avoid knee collapse")
        feedback = " | ".join(feedbacks) if feedbacks else ""

        depth_live = counter.depth_for_overlay()
        frame = draw_overlay(frame, reps=counter.count, feedback=feedback, depth_pct=depth_live)
        out.write(frame)

    pose.close()
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    result = counter.result()
    with open(feedback_path, "w", encoding="utf-8") as f:
        f.write(f"Total Reps: {result['squat_count']}\n")
        f.write(f"Technique Score: {result['technique_score']}/10\n")
        if result["feedback"]:
            f.write("Feedback:\n")
            for fb in result["feedback"]:
                f.write(f"- {fb}\n")

    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", output_path,
        "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
        encoded_path
    ], check=False)
    if os.path.exists(output_path): os.remove(output_path)

    return {**result, "video_path": encoded_path, "feedback_path": feedback_path}

# ===================== SQUAT (BODYWEIGHT) =====================
PERFECT_MIN_KNEE_SQ = 60.0
STAND_KNEE_ANGLE    = 160.0
MIN_FRAMES_BETWEEN_REPS_SQ = 10
DEPTH_ALPHA_SQ      = 0.35

def run_squat_analysis(video_path,
                       frame_skip=3,
                       scale=0.4,
                       output_path="squat_analyzed.mp4",
                       feedback_path="squat_feedback.txt"):
    mp_pose_mod = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0, "good_reps": 0, "bad_reps": 0,
            "feedback": ["Could not open video"], "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    counter = 0
    good_reps = 0
    bad_reps = 0
    overall_feedback = set()
    rep_reports = []
    all_scores = []

    stage = None
    frame_idx = 0
    last_rep_frame = -999

    start_knee_angle = None
    rep_min_knee_angle = 180.0
    rep_max_knee_angle = -999.0
    rep_min_torso_angle = 999.0
    rep_start_frame = None

    depth_smooth = 0.0
    peak_hold = 0
    def update_depth(dt, target):
        nonlocal depth_smooth, peak_hold
        depth_smooth = DEPTH_ALPHA_SQ * target + (1 - DEPTH_ALPHA_SQ) * depth_smooth
        if peak_hold > 0: peak_hold -= 1
        else: depth_smooth *= 0.985
        depth_smooth = float(np.clip(depth_smooth, 0.0, 1.0))
        return depth_smooth

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    with mp_pose_mod.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue
            if scale != 1.0: frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=update_depth(dt, 0.0))
                out.write(frame); continue

            try:
                lm = results.pose_landmarks.landmark
                R = mp_pose_mod.PoseLandmark
                hip      = np.array([lm[R.RIGHT_HIP.value].x,      lm[R.RIGHT_HIP.value].y])
                knee     = np.array([lm[R.RIGHT_KNEE.value].x,     lm[R.RIGHT_KNEE.value].y])
                ankle    = np.array([lm[R.RIGHT_ANKLE.value].x,    lm[R.RIGHT_ANKLE.value].y])
                shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])
                heel_y   = lm[R.RIGHT_HEEL.value].y

                knee_angle   = calculate_angle(hip, knee, ankle)
                torso_angle  = calculate_angle(shoulder, hip, knee)

                # start down
                if knee_angle < 100:
                    if stage != "down":
                        start_knee_angle = float(knee_angle)
                        rep_min_knee_angle = 180.0
                        rep_max_knee_angle = -999.0
                        rep_min_torso_angle = 999.0
                        rep_start_frame = frame_idx
                    stage = "down"

                if stage == "down":
                    rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                    rep_max_knee_angle = max(rep_max_knee_angle, knee_angle)
                    rep_min_torso_angle = min(rep_min_torso_angle, torso_angle)
                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE_SQ))
                        depth_target = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))
                        update_depth(dt, depth_target)

                # end rep
                if knee_angle > STAND_KNEE_ANGLE and stage == "down":
                    feedbacks = []
                    penalty = 0.0
                    hip_to_heel_dist = abs(hip[1] - heel_y)
                    if hip_to_heel_dist > 0.48: feedbacks.append("Try to squat deeper");            penalty += 3
                    elif hip_to_heel_dist > 0.45: feedbacks.append("Almost there — go a bit lower");  penalty += 1.5
                    elif hip_to_heel_dist > 0.43: feedbacks.append("Looking good — just a bit more depth"); penalty += 0.5
                    if torso_angle < 140:
                        feedbacks.append("Try to keep your back a bit straighter"); penalty += 1.0
                    if knee_angle < 160:
                        feedbacks.append("Finish with knees fully extended");        penalty += 1.0

                    score = 10.0 if penalty == 0 else round(max(4, 10 - min(penalty,6)) * 2) / 2
                    depth_pct = 0.0
                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE_SQ))
                        depth_pct = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))

                    rep_reports.append({
                        "rep_index": counter + 1,
                        "score": round(float(score), 1),
                        "feedback": feedbacks[:],
                        "start_frame": rep_start_frame or 0,
                        "end_frame": frame_idx,
                        "start_knee_angle": round(float(start_knee_angle or knee_angle), 2),
                        "min_knee_angle": round(float(rep_min_knee_angle), 2),
                        "max_knee_angle": round(float(rep_max_knee_angle), 2),
                        "torso_min_angle": round(float(rep_min_torso_angle), 2),
                        "depth_pct": depth_pct
                    })

                    if feedbacks: overall_feedback.update(feedbacks)
                    start_knee_angle = None
                    stage = "up"

                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS_SQ:
                        counter += 1
                        last_rep_frame = frame_idx
                        if score >= 9.5: good_reps += 1
                        else: bad_reps += 1
                        all_scores.append(score)

                # שלד – גוף בלבד
                frame = draw_body_only(frame, lm)

                # overlay
                frame = draw_overlay(
                    frame, reps=counter,
                    feedback=" | ".join(list(overall_feedback)[-2:]) if overall_feedback else None,
                    depth_pct=depth_smooth
                )
                out.write(frame)

            except Exception:
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=update_depth(dt, 0.0))
                if out is not None: out.write(frame)
                continue

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    feedback_list = list(overall_feedback) if bad_reps > 0 else ["Great form! Keep it up 💪"]

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list: f.write(f"- {fb}\n")
    except Exception:
        pass

    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
            encoded_path
        ], check=False)
        if os.path.exists(output_path) and os.path.exists(encoded_path):
            os.remove(output_path)
    except Exception:
        pass
    final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")

    return {
        "squat_count": counter,
        "technique_score": technique_score,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": feedback_list,
        "reps": rep_reports,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }

# ===== Alias לשמירה על תאימות ל-app.py =====
def run_analysis(*args, **kwargs):
    return run_squat_analysis(*args, **kwargs)


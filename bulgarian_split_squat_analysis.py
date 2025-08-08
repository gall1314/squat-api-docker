import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from PIL import ImageFont, ImageDraw, Image

# ===================== קבועים =====================
ANGLE_DOWN_THRESH   = 95     # כניסה לירידה
ANGLE_UP_THRESH     = 160    # יציאה לעלייה
MIN_RANGE_DELTA_DEG = 12     # שינוי מינימלי (Top->Bottom) כדי לספור חזרה (הוקל מ-15)
MIN_DOWN_FRAMES     = 5      # מינ' פריימים בירידה (היסטרזיס)

GOOD_REP_MIN_SCORE  = 8.0
TORSO_LEAN_MIN      = 135    # גב פחות מזה -> הערה
VALGUS_X_TOL        = 0.03   # סטייה מותרת X בין ברך לקרסול
PERFECT_MIN_KNEE    = 70     # יעד "מושלם" לנרמול עומק (לא לספירה)

# === שיפורי יציבות ===
EMA_ALPHA             = 0.6   # החלקת זוויות (0=הרבה החלקה, 1=בלי)
TORSO_MARGIN_DEG      = 3     # חוצץ קטן לפני הערת גב
TORSO_BAD_MIN_FRAMES  = 4     # חייבים X פריימים רצופים לפני הערת גב
VALGUS_BAD_MIN_FRAMES = 3     # חייבים X פריימים רצופים לוואלגוס
REP_DEBOUNCE_FRAMES   = 6     # מרווח מינ' פריימים בין חזרות

# פונטים
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14     # "DEPTH"
DEPTH_PCT_FONT_SIZE   = 18     # "72%"

try:
    REPS_FONT = ImageFont.truetype(FONT_PATH, REPS_FONT_SIZE)
    FEEDBACK_FONT = ImageFont.truetype(FONT_PATH, FEEDBACK_FONT_SIZE)
    DEPTH_LABEL_FONT = ImageFont.truetype(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
    DEPTH_PCT_FONT   = ImageFont.truetype(FONT_PATH, DEPTH_PCT_FONT_SIZE)
except Exception:
    REPS_FONT = ImageFont.load_default()
    FEEDBACK_FONT = ImageFont.load_default()
    DEPTH_LABEL_FONT = ImageFont.load_default()
    DEPTH_PCT_FONT = ImageFont.load_default()

# סגנון דונאט
DEPTH_RADIUS_SCALE   = 0.70   # יחס לגובה הפס העליון (bar_h)
DEPTH_THICKNESS_FRAC = 0.28   # עובי הטבעת ביחס לרדיוס
DEPTH_COLOR          = (40, 200, 80)  # BGR ירוק
DEPTH_RING_BG        = (70, 70, 70)   # רקע טבעת אפור

mp_pose = mp.solutions.pose

# ===================== החלקת זוויות (EMA) =====================
class AngleEMA:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = float(alpha)
        self.knee = None
        self.torso = None

    def update(self, knee_angle, torso_angle):
        ka = float(knee_angle)
        ta = float(torso_angle)
        if self.knee is None:
            self.knee = ka
            self.torso = ta
        else:
            a = self.alpha
            self.knee  = a * ka + (1.0 - a) * self.knee
            self.torso = a * ta + (1.0 - a) * self.torso
        return self.knee, self.torso

# ===================== עזרי ציור =====================
def draw_plain_text(pil_img, xy, text, font, color=(255,255,255)):
    ImageDraw.Draw(pil_img).text((int(xy[0]), int(xy[1])), text, font=font, fill=color)
    return np.array(pil_img)

def draw_depth_donut(frame, center, radius, thickness, pct):
    """
    טבעת DEPTH: pct ∈ [0..1], 0=ריק, 1=מלא. מתחיל מלמעלה (-90°) עם כיוון השעון.
    """
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius   = int(radius)
    thickness = int(thickness)

    # טבעת רקע אפורה
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    # קשת מילוי ירוקה
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """
    פס עליון: Reps + Depth donut קטן (מוצג תמיד) ; פס תחתון: פידבק.
    """
    h, w, _ = frame.shape
    bar_h = int(h * 0.065)

    # פס עליון שקוף
    top = frame.copy()
    cv2.rectangle(top, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, 0.55, frame, 0.45, 0)

    # Reps
    pil = Image.fromarray(frame)
    frame = draw_plain_text(pil, (16, int(bar_h*0.22)), f"Reps: {reps}", REPS_FONT)

    # DEPTH donut בפינה ימין-עליון
    margin = 12
    radius = int(bar_h * DEPTH_RADIUS_SCALE)
    thick  = max(3, int(radius * DEPTH_THICKNESS_FRAC))
    cx = w - margin - radius
    cy = int(bar_h * 0.52)

    frame = draw_depth_donut(frame, (cx, cy), radius, thick, depth_pct)

    # טקסטים בתוך הדונאט: "DEPTH" למעלה, אחוז מתחת – ממורכזים
    pil  = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)

    label_txt = "DEPTH"
    pct_txt   = f"{int(depth_pct*100)}%"

    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt,   font=DEPTH_PCT_FONT)

    gap    = max(2, int(radius * 0.10))
    block_h = DEPTH_LABEL_FONT_SIZE + gap + DEPTH_PCT_FONT_SIZE
    base_y  = cy - block_h // 2

    lx = cx - int(label_w // 2)
    ly = base_y
    draw.text((lx, ly), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))

    px = cx - int(pct_w // 2)
    py = ly + DEPTH_LABEL_FONT_SIZE + gap
    draw.text((px, py), pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))

    frame = np.array(pil)

    # פידבק תחתון
    if feedback:
        bottom_h = int(h * 0.07)
        over = frame.copy()
        cv2.rectangle(over, (0, h-bottom_h), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(over, 0.55, frame, 0.45, 0)

        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        tw = draw.textlength(feedback, font=FEEDBACK_FONT)
        tx = (w - int(tw)) // 2
        ty = h - bottom_h + 6
        draw.text((tx, ty), feedback, font=FEEDBACK_FONT, fill=(255,255,255))
        frame = np.array(pil)

    return frame

# ===================== גאומטריה =====================
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

# ===================== מונה חזרות =====================
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

        self._start_knee_angle = None  # Top angle בתחילת החזרה
        self._curr_min_knee = 999.0
        self._curr_max_knee = -999.0
        self._curr_min_torso = 999.0
        self._curr_valgus_bad = 0
        self._torso_bad_frames = 0        # חדש: סופרים רצף גב עקום
        self._valgus_bad_frames = 0       # חדש: סופרים רצף וואלגוס
        self._down_frames = 0
        self._last_depth_for_ui = 0.0     # להצגה תמידית
        self._last_rep_end_frame = -10    # חדש: ל-debounce

    def _start_rep(self, frame_no, start_knee_angle):
        # אל תתחיל חזרה אם לא עבר debounce
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
            if feedback:
                self.all_feedback.update(feedback)

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
        if extra:
            report.update(extra)
        self.rep_reports.append(report)
        self.rep_index += 1
        self.rep_start_frame = None
        self._start_knee_angle = None
        self._last_depth_for_ui = 0.0
        self._last_rep_end_frame = frame_no  # חשוב ל-debounce

    def evaluate_form(self, start_knee_angle, min_knee_angle, min_torso_angle, valgus_bad_frames):
        feedback = []
        score = 10.0

        # עומק יחסי: כמה ירדתי מהטופ עד המינימום, מנורמל לסף "מושלם"
        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE))
        depth_pct = np.clip((start_knee_angle - min_knee_angle) / denom, 0, 1)

        # דרוש רצף פריימים אמיתי לפני הערות
        if self._torso_bad_frames >= TORSO_BAD_MIN_FRAMES:
            feedback.append("Keep your back straight"); score -= 2
        if valgus_bad_frames >= VALGUS_BAD_MIN_FRAMES:
            feedback.append("Avoid knee collapse");     score -= 2
        if depth_pct < 0.8:
            feedback.append("Go a bit deeper");         score -= 1

        return score, feedback, float(depth_pct)

    def update(self, knee_angle, torso_angle, valgus_ok_flag, frame_no):
        # כניסה לירידה
        if knee_angle < ANGLE_DOWN_THRESH:
            if self.stage != 'down':
                self.stage = 'down'
                started = self._start_rep(frame_no, knee_angle)
                if not started:
                    # עדיין ב-debounce: אל תספור חזרה חדשה
                    self.stage = 'up'
                    return
            self._down_frames += 1

        # יציאה לעלייה -> בדיקת חזרה
        elif knee_angle > ANGLE_UP_THRESH and self.stage == 'down':
            depth_delta = (self._start_knee_angle or 0) - (self._curr_min_knee or 0)
            did_move_enough = depth_delta >= MIN_RANGE_DELTA_DEG

            if self._down_frames >= MIN_DOWN_FRAMES and did_move_enough:
                score, fb, depth = self.evaluate_form(
                    float(self._start_knee_angle or knee_angle),
                    float(self._curr_min_knee or knee_angle),
                    float(self._curr_min_torso or 180.0),
                    self._curr_valgus_bad
                )
                self.count += 1
                self._finish_rep(frame_no, score, fb, extra={"depth_pct": depth})
            else:
                # חזרה לא תקינה – אפס עומק תצוגה
                self._last_depth_for_ui = 0.0
                self.rep_start_frame = None
                self._start_knee_angle = None

            self.stage = 'up'

        # אגרגציה בזמן ירידה + סופרי "רצף" להערות
        if self.stage == 'down' and self.rep_start_frame:
            self._curr_min_knee = min(self._curr_min_knee, knee_angle)
            self._curr_max_knee = max(self._curr_max_knee, knee_angle)
            self._curr_min_torso = min(self._curr_min_torso, torso_angle)

            # גב: דרוש מתחת לסף-מרווח כדי להיחשב "רע"
            if torso_angle < (TORSO_LEAN_MIN - TORSO_MARGIN_DEG):
                self._torso_bad_frames += 1
            else:
                self._torso_bad_frames = 0

            # ברך: דרוש רצף פריימים
            if not valgus_ok_flag:
                self._valgus_bad_frames += 1
                self._curr_valgus_bad += 1
            else:
                self._valgus_bad_frames = 0

            # עומק להצגה חיה
            denom = max(10.0, (self._start_knee_angle - PERFECT_MIN_KNEE))
            self._last_depth_for_ui = float(np.clip(
                (self._start_knee_angle - self._curr_min_knee) / denom, 0, 1
            ))

    def depth_for_overlay(self):
        """ערך להצגה תמידית: 0 כשאין ירידה כרגע."""
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

# ===================== ריצה =====================
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
        if not ret:
            break

        frame_no += 1
        if frame_skip > 1 and (frame_no % frame_skip) != 0:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        h, w = frame.shape[:2]
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            # הצג overlay גם כשאין זיהוי: עומק 0
            frame = draw_overlay(frame, reps=counter.count, feedback=None, depth_pct=0.0)
            out.write(frame)
            continue

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

        # החלקת זוויות
        knee_angle, torso_angle = ema.update(knee_angle_raw, torso_angle_raw)

        v_ok = valgus_ok(landmarks, side)

        counter.update(knee_angle, torso_angle, v_ok, frame_no)

        # שלד
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # פידבק בזמן אמת – רק אם יש רצף קצר של פריימים "רע"
        feedbacks = []
        if counter.stage == "down":
            if counter._torso_bad_frames >= TORSO_BAD_MIN_FRAMES:
                feedbacks.append("Keep your back straight")
            if counter._valgus_bad_frames >= VALGUS_BAD_MIN_FRAMES:
                feedbacks.append("Avoid knee collapse")
        feedback = " | ".join(feedbacks) if feedbacks else ""

        # עומק להצגה תמידית
        depth_live = counter.depth_for_overlay()
        frame = draw_overlay(frame, reps=counter.count, feedback=feedback, depth_pct=depth_live)

        out.write(frame)

    pose.close()
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    result = counter.result()

    # קובץ תקציר
    with open(feedback_path, "w", encoding="utf-8") as f:
        f.write(f"Total Reps: {result['squat_count']}\n")
        f.write(f"Technique Score: {result['technique_score']}/10\n")
        f.write("Depth = relative squat depth per rep (vs. your top)\n")
        if result["feedback"]:
            f.write("Feedback:\n")
            for fb in result["feedback"]:
                f.write(f"- {fb}\n")

    # קידוד יצוא
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        encoded_path
    ])
    if os.path.exists(output_path):
        os.remove(output_path)

    return {
        **result,
        "video_path": encoded_path,
        "feedback_path": feedback_path
    }



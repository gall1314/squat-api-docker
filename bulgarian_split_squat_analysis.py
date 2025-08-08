import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from PIL import ImageFont, ImageDraw, Image

# ===== קבועים =====
ANGLE_DOWN_THRESH   = 95    # זווית ברך שמתחתיה נכנסים לשלב "ירידה"
ANGLE_UP_THRESH     = 160   # זווית ברך שמעליה יוצאים חזרה ל"למעלה"
MIN_RANGE_DELTA_DEG = 15    # שינוי זווית מינימלי (Top->Bottom) כדי לספור חזרה
MIN_DOWN_FRAMES     = 5     # מינימום פריימים בשלב הירידה (היסטרזיס)

GOOD_REP_MIN_SCORE  = 8.0   # סף ציון לחזרה "טובה" (לא משפיע על ספירה)
TORSO_LEAN_MIN      = 135   # זווית גב מינימלית (פחות מזה -> גב נשבר)
VALGUS_X_TOL        = 0.03  # סטייה מותרת X בין ברך לקרסול (ולגוס)

# יעד איכות (לא לספירה): כמה רחוק מה-Top נחשב "מושלם" לצורך נרמול עומק
PERFECT_MIN_KNEE    = 70

# פונטים
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22

try:
    REPS_FONT = ImageFont.truetype(FONT_PATH, REPS_FONT_SIZE)
    FEEDBACK_FONT = ImageFont.truetype(FONT_PATH, FEEDBACK_FONT_SIZE)
except Exception:
    REPS_FONT = ImageFont.load_default()
    FEEDBACK_FONT = ImageFont.load_default()

mp_pose = mp.solutions.pose

# ===== ציור =====
def draw_plain_text(pil_img, xy, text, font, color=(255,255,255)):
    ImageDraw.Draw(pil_img).text((int(xy[0]), int(xy[1])), text, font=font, fill=color)
    return np.array(pil_img)

def draw_depth_donut(frame, center, radius, thickness, pct):
    """
    טבעת DEPTH: pct ∈ [0..1], 0=ריק, 1=מלא. התחלה מלמעלה, עם כיוון השעון.
    """
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius = int(radius)
    thickness = int(thickness)

    # רקע הטבעת (אפור כהה)
    cv2.circle(frame, (cx, cy), radius, (60, 60, 60), thickness, lineType=cv2.LINE_AA)
    # קשת מילוי (לבן)
    end_angle = -90 + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, end_angle,
                (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, depth_pct=None):
    """
    פס עליון עם Reps + Depth donut קטן; פס תחתון לפידבק (אם יש).
    """
    h, w, _ = frame.shape
    bar_h = int(h * 0.065)

    # פס עליון שקוף
    top = frame.copy()
    cv2.rectangle(top, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, 0.55, frame, 0.45, 0)

    # Reps (טקסט נקי)
    pil = Image.fromarray(frame)
    frame = draw_plain_text(pil, (16, int(bar_h*0.2)), f"Reps: {reps}", REPS_FONT)

    # Depth donut קטן בפינה ימין-עליון
    if depth_pct is not None:
        margin = 12
        radius = int(bar_h * 0.45)
        thick  = max(3, int(radius * 0.35))
        cx = w - margin - radius
        cy = int(bar_h * 0.5)
        frame = draw_depth_donut(frame, (cx, cy), radius, thick, depth_pct)

        # אחוז במרכז הטבעת
        pct_txt = f"{int(depth_pct*100)}%"
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        tw = draw.textlength(pct_txt, font=REPS_FONT)
        frame = draw_plain_text(pil, (cx - int(tw)//2, cy - REPS_FONT_SIZE//2), pct_txt, REPS_FONT)

    # פידבק תחתון (אם יש)
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

# ===== גאומטריה =====
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

# ===== ספירה ואיכות =====
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

        self._start_knee_angle = None  # ה"טופ" של אותה חזרה
        self._curr_min_knee = None
        self._curr_max_knee = None
        self._curr_min_torso = None
        self._curr_valgus_bad = 0
        self._down_frames = 0

    def _start_rep(self, frame_no, start_knee_angle):
        self.rep_start_frame = frame_no
        self._start_knee_angle = float(start_knee_angle)
        self._curr_min_knee = 999.0
        self._curr_max_knee = -999.0
        self._curr_min_torso = 999.0
        self._curr_valgus_bad = 0
        self._down_frames = 0

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

    def evaluate_form(self, start_knee_angle, min_knee_angle, min_torso_angle, valgus_bad_frames):
        """
        איכות/עומק נמדדים יחסית ל-Top של אותה חזרה.
        """
        feedback = []
        score = 10.0

        # עומק יחסי: כמה ירדתי מהטופ עד המינימום, מנורמל ליעד "מושלם"
        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE))  # הגנה על חילוק קטן
        depth_pct = np.clip((start_knee_angle - min_knee_angle) / denom, 0, 1)

        if min_torso_angle < TORSO_LEAN_MIN:
            feedback.append("Keep your back straight"); score -= 2
        if valgus_bad_frames > 0:
            feedback.append("Avoid knee collapse");     score -= 2
        if depth_pct < 0.8:
            feedback.append("Go a bit deeper");         score -= 1

        return score, feedback, float(depth_pct)

    def update(self, knee_angle, torso_angle, valgus_ok_flag, frame_no):
        # כניסה לירידה
        if knee_angle < ANGLE_DOWN_THRESH:
            if self.stage != 'down':
                self.stage = 'down'
                self._start_rep(frame_no, knee_angle)
            self._down_frames += 1

        # מעבר למעלה -> סיום חזרה (אם באמת זזנו)
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

            self.stage = 'up'

        # אגרגציה בזמן הירידה
        if self.stage == 'down' and self.rep_start_frame:
            self._curr_min_knee = min(self._curr_min_knee, knee_angle)
            self._curr_max_knee = max(self._curr_max_knee, knee_angle)
            self._curr_min_torso = min(self._curr_min_torso, torso_angle)
            if not valgus_ok_flag:
                self._curr_valgus_bad += 1

    def current_depth_pct(self):
        """עומק Live בזמן ירידה, יחסית ל-Top של אותה חזרה."""
        if self.stage != 'down' or self._curr_min_knee in (None, 999.0) or self._start_knee_angle is None:
            return None
        denom = max(10.0, (self._start_knee_angle - PERFECT_MIN_KNEE))
        return float(np.clip((self._start_knee_angle - self._curr_min_knee) / denom, 0, 1))

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

# ===== ריצה =====
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

        knee_angle = calculate_angle(hip, knee, ankle)
        torso_angle = calculate_angle(shoulder, hip, knee)
        v_ok = valgus_ok(landmarks, side)

        counter.update(knee_angle, torso_angle, v_ok, frame_no)

        # שלד
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # פידבק בזמן אמת בזמן ירידה
        feedbacks = []
        if counter.stage == "down":
            if torso_angle < TORSO_LEAN_MIN: feedbacks.append("Keep your back straight")
            if not v_ok: feedbacks.append("Avoid knee collapse")
        feedback = " | ".join(feedbacks) if feedbacks else ""

        # עומק ל‑donut בזמן אמת
        depth_live = counter.current_depth_pct()
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


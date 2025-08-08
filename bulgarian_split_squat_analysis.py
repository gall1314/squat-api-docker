import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from PIL import ImageFont, ImageDraw, Image

# ===== קבועים =====
ANGLE_DOWN_THRESH = 95
ANGLE_UP_THRESH   = 160
GOOD_REP_MIN_SCORE = 8.0
TORSO_LEAN_MIN     = 135

VALGUS_X_TOL = 0.03
MIN_VALID_KNEE   = 90   # עומק מינימלי לספירה (ככל שקטן יותר -> עמוק יותר)
PERFECT_MIN_KNEE = 70   # עומק "מושלם" לחישוב ROM=100%
MIN_DOWN_FRAMES  = 5    # מינימום פריימים בשלב ירידה כדי להיחשב חזרה

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
LEGEND_FONT_SIZE = 18

# ===== טעינת פונט פעם אחת =====
try:
    REPS_FONT = ImageFont.truetype(FONT_PATH, REPS_FONT_SIZE)
    FEEDBACK_FONT = ImageFont.truetype(FONT_PATH, FEEDBACK_FONT_SIZE)
    LEGEND_FONT = ImageFont.truetype(FONT_PATH, LEGEND_FONT_SIZE)
except Exception:
    REPS_FONT = ImageFont.load_default()
    FEEDBACK_FONT = ImageFont.load_default()
    LEGEND_FONT = ImageFont.load_default()

mp_pose = mp.solutions.pose


# ===== ציור =====
def draw_text_with_outline(draw, xy, text, font, fill=(255, 255, 255)):
    x, y = xy
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0,0,0))
    draw.text((x, y), text, font=font, fill=fill)


def draw_rom_badge(frame, pil_img, bar_height, rom_pct):
    """ROM badge נקיה בפינה הימנית העליונה: 'ROM 85%'"""
    h, w, _ = frame.shape
    draw = ImageDraw.Draw(pil_img)

    rom_pct = float(np.clip(rom_pct, 0.0, 1.0))
    txt = f"{int(rom_pct*100)}%"

    label = "ROM"
    label_w = draw.textlength(label, font=REPS_FONT)
    txt_w   = draw.textlength(txt,   font=REPS_FONT)
    pad_x, pad_y = 10, 6
    spacing      = 8

    total_w = int(label_w + spacing + txt_w + pad_x*2)
    total_h = int(REPS_FONT_SIZE + pad_y*2)

    x1 = int(w - 20)
    x0 = int(x1 - total_w)
    y0 = int(bar_height * 0.18)
    y1 = int(y0 + total_h)

    # קלמפ לגבולות הפריים
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w-1, x1); y1 = min(h-1, y1)

    frame_np = np.array(pil_img)
    overlay  = frame_np.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame_np = cv2.addWeighted(overlay, 0.35, frame_np, 0.65, 0)
    cv2.rectangle(frame_np, (x0, y0), (x1, y1), (255, 255, 255), 2)

    pil_img = Image.fromarray(frame_np)
    draw = ImageDraw.Draw(pil_img)
    draw_text_with_outline(draw, (x0+pad_x, y0+pad_y), label, REPS_FONT)
    draw_text_with_outline(draw, (x0+pad_x+int(label_w)+spacing, y0+pad_y), txt, REPS_FONT)
    return np.array(pil_img)


def draw_overlay(frame, reps=0, feedback=None, rom_pct=None, show_legend=False):
    h, w, _ = frame.shape
    bar_height = int(h * 0.07)

    # פס עליון שקוף
    top = frame.copy()
    cv2.rectangle(top, (0, 0), (w, bar_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, 0.6, frame, 0.4, 0)

    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # Reps
    draw_text_with_outline(draw, (20, int(bar_height * 0.2)), f"Reps: {reps}", REPS_FONT)

    # ROM badge
    if rom_pct is not None:
        frame = draw_rom_badge(frame, pil_img, bar_height, rom_pct)
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

    # הסבר קצר בתחילת הווידאו (~2 שניות)
    if show_legend:
        legend = "ROM = Range of Motion (depth per rep)"
        tw = draw.textlength(legend, font=LEGEND_FONT)
        pad_x, pad_y = 10, 6

        x0 = 20
        y0 = int(bar_height + 10)
        x1 = int(x0 + tw + pad_x*2)
        y1 = int(y0 + LEGEND_FONT_SIZE + pad_y*2)

        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(w-1, x1); y1 = min(h-1, y1)

        frame_np = np.array(pil_img)
        overlay  = frame_np.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        frame_np = cv2.addWeighted(overlay, 0.35, frame_np, 0.65, 0)
        cv2.rectangle(frame_np, (x0, y0), (x1, y1), (255, 255, 255), 1)

        pil_img = Image.fromarray(frame_np)
        draw = ImageDraw.Draw(pil_img)
        draw_text_with_outline(draw, (x0+pad_x, y0+pad_y), legend, LEGEND_FONT)

    # פידבק תחתון
    if feedback:
        frame = np.array(pil_img)
        bottom = frame.copy()
        cv2.rectangle(bottom, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(bottom, 0.6, frame, 0.4, 0)
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        max_width = int(w * 0.9)
        words = feedback.split()
        lines, current = [], ""
        for word in words:
            test = (current + " " + word).strip()
            if draw.textlength(test, font=FEEDBACK_FONT) <= max_width:
                current = test
            else:
                lines.append(current); current = word
        if current:
            lines.append(current)

        y = h - bar_height + 5
        for line in lines:
            tw = draw.textlength(line, font=FEEDBACK_FONT)
            x = (w - int(tw)) // 2
            draw_text_with_outline(draw, (x, y), line, FEEDBACK_FONT)
            y += FEEDBACK_FONT_SIZE + 4

    return np.array(pil_img)


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


# ===== מונה חזרות =====
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

        self._curr_min_knee = None
        self._curr_max_knee = None
        self._curr_min_torso = None
        self._curr_valgus_bad = 0
        self._down_frames = 0

    def _start_rep(self, frame_no):
        self.rep_start_frame = frame_no
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
            "min_knee_angle": round(self._curr_min_knee, 2),
            "max_knee_angle": round(self._curr_max_knee, 2),
            "torso_min_angle": round(self._curr_min_torso, 2)
        }
        if extra:
            report.update(extra)

        self.rep_reports.append(report)
        self.rep_index += 1
        self.rep_start_frame = None

    def evaluate_form(self, min_knee_angle, min_torso_angle, valgus_bad_frames):
        feedback = []
        score = 10.0

        rom_pct = np.clip(
            (MIN_VALID_KNEE - min_knee_angle) / (MIN_VALID_KNEE - PERFECT_MIN_KNEE),
            0, 1
        )

        if min_torso_angle < TORSO_LEAN_MIN:
            feedback.append("Keep your back straight"); score -= 2
        if valgus_bad_frames > 0:
            feedback.append("Avoid knee collapse");     score -= 2
        if rom_pct < 0.8:
            feedback.append("Go a bit deeper");         score -= 1

        return score, feedback, float(rom_pct)

    def update(self, knee_angle, torso_angle, valgus_ok_flag, frame_no):
        if knee_angle < ANGLE_DOWN_THRESH:
            if self.stage != 'down':
                self.stage = 'down'
                self._start_rep(frame_no)
            self._down_frames += 1
        elif knee_angle > ANGLE_UP_THRESH and self.stage == 'down':
            if self._down_frames >= MIN_DOWN_FRAMES and self._curr_min_knee <= MIN_VALID_KNEE:
                score, fb, rom = self.evaluate_form(
                    self._curr_min_knee, self._curr_min_torso, self._curr_valgus_bad
                )
                self.count += 1
                self._finish_rep(frame_no, score, fb, extra={"rom_pct": rom})
            self.stage = 'up'

        if self.stage == 'down' and self.rep_start_frame:
            self._curr_min_knee = min(self._curr_min_knee, knee_angle)
            self._curr_max_knee = max(self._curr_max_knee, knee_angle)
            self._curr_min_torso = min(self._curr_min_torso, torso_angle)
            if not valgus_ok_flag:
                self._curr_valgus_bad += 1

    def current_rom_pct(self):
        if self.stage != 'down' or self._curr_min_knee in (None, 999.0):
            return None
        return float(np.clip(
            (MIN_VALID_KNEE - self._curr_min_knee) / (MIN_VALID_KNEE - PERFECT_MIN_KNEE), 0, 1
        ))

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

        rom_live = counter.current_rom_pct()
        show_legend = (frame_no / effective_fps) < 2.0

        frame = draw_overlay(frame, reps=counter.count, feedback=feedback,
                             rom_pct=rom_live, show_legend=show_legend)

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
        f.write("ROM = Range of Motion (depth per rep)\n")
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



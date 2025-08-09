import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from utils import calculate_angle, calculate_body_angle
from PIL import ImageFont, ImageDraw, Image

# ========= Fonts =========
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 24         # קטן יותר
FEEDBACK_FONT_SIZE = 24     # אחיד וקריא
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE   = 18

def _load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

REPS_FONT        = _load_font(REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(DEPTH_PCT_FONT_SIZE)

# ========= Top bar & Donut layout (קבועים) =========
TOP_BAR_HEIGHT_FRAC = 0.09   # גובה הפס העליון
TOP_BAR_WIDTH_FRAC  = 0.30   # רוחב הפס העליון (שמאל) – קבוע, לא קופץ
TOP_BAR_ALPHA       = 0.55   # שקיפות

DONUT_BASE_FRAC     = 0.11   # בסיס גודל הדונאט מתוך גובה הווידאו

# ========= Donut style =========
DEPTH_RADIUS_SCALE   = 0.70
DEPTH_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)   # BGR
DEPTH_RING_BG        = (70, 70, 70)    # BGR
PERFECT_MIN_KNEE     = 70
DEPTH_SMOOTH_TAU_SEC = 0.30
DEPTH_PEAK_HOLD_SEC  = 0.18
DEPTH_FADE_SEC       = 0.65

# ========= Skeleton (בלי פנים) =========
BODY_LANDMARKS = {11,12,13,14,15,16,23,24,25,26,27,28}
BODY_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
]
SKELETON_COLOR = (255,255,255)
SKELETON_THICK = 2
JOINT_RADIUS   = 3

def draw_body_skeleton(frame, landmarks, w, h):
    for a, b in BODY_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), SKELETON_COLOR, SKELETON_THICK, cv2.LINE_AA)
    for idx in BODY_LANDMARKS:
        if idx < len(landmarks):
            x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (x, y), JOINT_RADIUS, SKELETON_COLOR, -1, cv2.LINE_AA)
    return frame

# ========= Text helpers =========
def fit_text_to_width(draw: ImageDraw.ImageDraw, text: str, max_width: int,
                      base_px: int, min_px: int) -> ImageFont.FreeTypeFont:
    size = int(base_px)
    while size >= int(min_px):
        try:
            f = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            f = ImageFont.load_default()
        if draw.textlength(text, font=f) <= max_width:
            return f
        size -= 1
    return ImageFont.load_default()

def wrap_text_by_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
    words = text.split()
    if not words:
        return []
    lines, line = [], ""
    for w in words:
        test = (line + " " + w).strip()
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
                line = w
            else:
                lines.append(w)  # מילה בודדת ארוכה
                line = ""
    if line:
        lines.append(line)
    return lines

# ========= Donut =========
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

# ========= Overlay =========
def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """
    - פס עליון קבוע (רוחב/גובה) מתחת ל-Reps בלבד.
    - הדונאט חופשי בצד ימין, בגודל יציב, לא מכוסה ולא נחתך.
    - פידבק בתחתית: פונט אחיד + עיטוף שורות + גובה דינמי.
    """
    h, w, _ = frame.shape
    PADDING = 14

    # --- פס עליון קבוע ---
    bar_h = int(h * TOP_BAR_HEIGHT_FRAC)
    bar_w = int(w * TOP_BAR_WIDTH_FRAC)
    top = frame.copy()
    cv2.rectangle(top, (0, 0), (bar_w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, TOP_BAR_ALPHA, frame, 1 - TOP_BAR_ALPHA, 0)

    # Reps
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    rx = PADDING
    ry = max(4, (bar_h - REPS_FONT.size) // 2)
    draw.text((rx, ry), reps_text, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # --- דונאט חופשי ---
    donut_base = int(h * DONUT_BASE_FRAC)
    radius = int(donut_base * DEPTH_RADIUS_SCALE)
    thick  = max(3, int(radius * DEPTH_THICKNESS_FRAC))

    margin = 12
    cx = w - margin - radius
    cy = max(radius + 2, int(donut_base * 0.84))  # בטוח לא נחתך
    frame = draw_depth_donut(frame, (cx, cy), radius, thick, depth_pct)

    # טקסטים בתוך הדונאט
    pil  = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"
    pct_txt   = f"{int(depth_pct*100)}%"
    gap    = max(2, int(radius * 0.10))
    block_h = DEPTH_LABEL_FONT_SIZE + gap + DEPTH_PCT_FONT_SIZE
    base_y  = cy - block_h // 2
    lw = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pw = draw.textlength(pct_txt,   font=DEPTH_PCT_FONT)
    draw.text((cx - int(lw // 2), base_y),                            label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pw // 2), base_y + DEPTH_LABEL_FONT_SIZE + gap), pct_txt,   font=DEPTH_PCT_FONT,   fill=(255,255,255))
    frame = np.array(pil)

    # --- פס תחתון לפידבק עם עיטוף שורות ---
    if feedback:
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        max_w = w - 2 * 16
        lines = wrap_text_by_width(draw, feedback, FEEDBACK_FONT, max_w)

        line_h = FEEDBACK_FONT.size + 2
        needed_h = 10 + len(lines) * line_h + 10
        bottom_h = max(int(h * 0.10), needed_h)
        over = np.array(pil).copy()
        cv2.rectangle(over, (0, h-bottom_h), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(over, 0.55, np.array(pil), 0.45, 0)

        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        y = h - bottom_h + (bottom_h - len(lines) * line_h) // 2
        for line in lines:
            tw_line = draw.textlength(line, font=FEEDBACK_FONT)
            tx = (w - int(tw_line)) // 2
            draw.text((tx, y), line, font=FEEDBACK_FONT, fill=(255,255,255))
            y += line_h
        frame = np.array(pil)

    return frame

# ========= Depth smoother =========
class DepthSmoother:
    def __init__(self, tau_sec=DEPTH_SMOOTH_TAU_SEC):
        self.tau = float(tau_sec)
        self.display = 0.0
        self.target = 0.0
        self.hold_timer = 0.0
        self.fade_timer = 0.0
    def set_target(self, val): self.target = float(np.clip(val, 0.0, 1.0))
    def peak_hold(self, seconds): self.hold_timer = max(self.hold_timer, float(seconds)); self.fade_timer = 0.0
    def start_fade(self, seconds): self.hold_timer = 0.0; self.fade_timer = float(seconds)
    def update(self, dt):
        if self.hold_timer > 0:
            self.hold_timer = max(0.0, self.hold_timer - dt); return self.display
        if self.fade_timer > 0:
            if self.fade_timer <= dt:
                self.display = 0.0; self.fade_timer = 0.0
            else:
                self.display = max(0.0, self.display - self.display * (dt / self.fade_timer))
                self.fade_timer -= dt
            return self.display
        k = dt / (self.tau + dt) if self.tau > 1e-6 else 1.0
        self.display += k * (self.target - self.display)
        return self.display

# ========= Utils =========
def _format_score_value(x: float):
    x = round(x * 2) / 2
    return int(x) if float(x).is_integer() else round(x, 1)

# ========= Main =========
def run_analysis(video_path, frame_skip=3, scale=0.4,
                 output_path="squat_output.mp4",
                 feedback_path="squat_feedback.txt"):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    counter = good_reps = bad_reps = 0
    all_scores, problem_reps, overall_feedback = [], [], []
    any_feedback_session = False

    stage = None
    start_knee_angle = None
    rep_min_knee_angle = 180.0
    last_rep_feedback = []
    depth_smoother = DepthSmoother()
    last_depth_for_ui = 0.0

    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    with mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue
            if scale != 1.0: frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None: out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                last_depth_for_ui = depth_smoother.update(dt)
                overlay_text = " | ".join(last_rep_feedback) if last_rep_feedback else None
                frame = draw_overlay(frame, reps=counter, feedback=overlay_text, depth_pct=last_depth_for_ui)
                out.write(frame)
                continue

            try:
                lm = results.pose_landmarks.landmark
                frame = draw_body_skeleton(frame, lm, w, h)

                r = mp_pose.PoseLandmark
                hip      = [lm[r.RIGHT_HIP.value].x,      lm[r.RIGHT_HIP.value].y]
                knee     = [lm[r.RIGHT_KNEE.value].x,     lm[r.RIGHT_KNEE.value].y]
                ankle    = [lm[r.RIGHT_ANKLE.value].x,    lm[r.RIGHT_ANKLE.value].y]
                shoulder = [lm[r.RIGHT_SHOULDER.value].x, lm[r.RIGHT_SHOULDER.value].y]
                heel_y   = lm[r.RIGHT_HEEL.value].y
                foot_y   = lm[r.RIGHT_FOOT_INDEX.value].y

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)
                body_angle = calculate_body_angle(shoulder, hip)
                heel_lifted = (foot_y - heel_y) > 0.03

                # התחלת ירידה
                if knee_angle < 90:
                    if stage != "down":
                        last_rep_feedback = []
                        start_knee_angle = float(knee_angle)
                        rep_min_knee_angle = 180.0
                    stage = "down"

                # יעד עומק לדונאט בזמן ירידה
                if stage == "down":
                    rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE))
                        depth_target = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))
                        depth_smoother.set_target(depth_target)

                # עדכון הדונאט בכל פריים
                last_depth_for_ui = depth_smoother.update(dt)

                # סוף חזרה: עלייה מעל 160 אחרי ירידה
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    feedbacks = []
                    penalty = 0

                    # עומק (hip->heel) לפידבק
                    hip_to_heel_dist = abs(hip[1] - heel_y)
                    if hip_to_heel_dist > 0.48:
                        feedbacks.append("Too shallow — squat lower"); penalty += 3
                    elif hip_to_heel_dist > 0.45:
                        feedbacks.append("Almost there — go a bit lower"); penalty += 1.5
                    elif hip_to_heel_dist > 0.43:
                        feedbacks.append("Looking good — just a bit more depth"); penalty += 0.5

                    if back_angle < 140:
                        feedbacks.append("Try to straighten your back more at the top"); penalty += 1
                    if heel_lifted:
                        feedbacks.append("Keep your heels down"); penalty += 1
                    if knee_angle < 160:
                        feedbacks.append("Incomplete lockout"); penalty += 1

                    if not feedbacks:
                        score = 10.0
                    else:
                        any_feedback_session = True
                        penalty = min(penalty, 6)
                        score = round(max(4, 10 - penalty) * 2) / 2

                    for f in feedbacks:
                        if f not in overall_feedback: overall_feedback.append(f)
                    last_rep_feedback = feedbacks[:]

                    depth_smoother.peak_hold(DEPTH_PEAK_HOLD_SEC)
                    depth_smoother.start_fade(DEPTH_FADE_SEC)
                    start_knee_angle = None

                    counter += 1
                    if score >= 9.5: good_reps += 1
                    else:
                        bad_reps += 1
                        problem_reps.append(counter)
                    all_scores.append(score)

            except Exception:
                pass

            overlay_text = " | ".join(last_rep_feedback) if last_rep_feedback else None
            frame = draw_overlay(frame, reps=counter, feedback=overlay_text, depth_pct=last_depth_for_ui)
            out.write(frame)

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    # ========== תוצאות ==========
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0.0
    if any_feedback_session and technique_score == 10.0:
        technique_score = 9.5
    technique_score_display = _format_score_value(technique_score)

    if not overall_feedback:
        overall_feedback.append("Great form! Keep it up 💪")

    # קובץ פידבק
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score_display}/10\n")
            if overall_feedback:
                f.write("Feedback:\n")
                for fb in overall_feedback:
                    f.write(f"- {fb}\n")
    except Exception:
        pass

    # קידוד MP4 (faststart) + Fallback בטוח
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg","-y","-i",output_path,
            "-c:v","libx264","-preset","fast","-movflags","+faststart","-pix_fmt","yuv420p",
            encoded_path
        ], check=False)
        if os.path.exists(encoded_path):
            if os.path.exists(output_path): os.remove(output_path)
    except Exception:
        pass

    final_video_path = encoded_path if os.path.exists(encoded_path) else (
        output_path if os.path.exists(output_path) else ""
    )

    # ==== החזרה – סכימה שעובדת ====
    return {
        "technique_score": technique_score_display,
        "squat_count": counter,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": overall_feedback,
        "problem_reps": problem_reps,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }


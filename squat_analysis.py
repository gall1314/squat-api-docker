import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from utils import calculate_angle, calculate_body_angle
from PIL import ImageFont, ImageDraw, Image

# ================== THEME (אחיד ונקי) ==================
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"

# פס עליון/תחתון
TOP_BAR_FRAC      = 0.095
BOTTOM_BAR_FRAC   = 0.08
BAR_BG_OPACITY    = 0.58

# טיפוגרפיה
REPS_FONT_PX      = 30
DEPTH_LABEL_PX    = 16
DEPTH_PCT_PX      = 22
FEEDBACK_BASE_PX  = 26
FEEDBACK_MIN_PX   = 18
PADDING           = 14

# דונאט
DEPTH_RADIUS_SCALE   = 0.78
DEPTH_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)   # BGR
DEPTH_RING_BG        = (70, 70, 70)    # BGR

# דינמיקת דונאט
PERFECT_MIN_KNEE     = 70
DEPTH_SMOOTH_TAU_SEC = 0.30
DEPTH_PEAK_HOLD_SEC  = 0.18
DEPTH_FADE_SEC       = 0.65

# שלד – גוף בלבד
BODY_LANDMARKS = {11,12,13,14,15,16,23,24,25,26,27,28}
BODY_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),
]
SKELETON_COLOR = (255,255,255)
SKELETON_THICK = 2
JOINT_RADIUS   = 3

# ספירת חזרות – היסטרזיס
KNEE_DOWN_TRIG     = 140
KNEE_BOTTOM_ZONE   = 95
KNEE_UP_TRIG       = 165
MIN_FRAMES_DOWN    = 3
MIN_FRAMES_AT_TOP  = 2

# ========= Fonts helpers =========
def _load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

REPS_FONT       = _load_font(REPS_FONT_PX)
DEPTH_LABEL_FONT= _load_font(DEPTH_LABEL_PX)
DEPTH_PCT_FONT  = _load_font(DEPTH_PCT_PX)

def fit_text_to_width(draw: ImageDraw.ImageDraw, text: str, max_width: int,
                      base_px: int, min_px: int) -> ImageFont.FreeTypeFont:
    size = base_px
    while size >= min_px:
        f = _load_font(size)
        if draw.textlength(text, font=f) <= max_width:
            return f
        size -= 1
    return _load_font(min_px)

# ========= Smoothers =========
class DepthSmoother:
    def __init__(self, tau_sec=DEPTH_SMOOTH_TAU_SEC):
        self.tau = float(tau_sec)
        self.display = 0.0
        self.target = 0.0
        self.hold_timer = 0.0
        self.fade_timer = 0.0

    def set_target(self, val):
        self.target = float(np.clip(val, 0.0, 1.0))

    def peak_hold(self, seconds):
        self.hold_timer = max(self.hold_timer, float(seconds))
        self.fade_timer = 0.0

    def start_fade(self, seconds):
        self.hold_timer = 0.0
        self.fade_timer = float(seconds)

    def update(self, dt):
        if self.hold_timer > 0:
            self.hold_timer = max(0.0, self.hold_timer - dt)
            return self.display
        if self.fade_timer > 0:
            if self.fade_timer <= dt:
                self.display = 0.0
                self.fade_timer = 0.0
            else:
                step = self.display * (dt / self.fade_timer)
                self.display = max(0.0, self.display - step)
                self.fade_timer -= dt
            return self.display
        if self.tau <= 1e-6:
            self.display = self.target
        else:
            k = dt / (self.tau + dt)
            self.display += k * (self.target - self.display)
        return self.display

class LandmarkSmoother:
    """Low-pass EMA על נקודות שלד כדי לצמצם 'קפיצות'."""
    def __init__(self, alpha=0.70):
        self.alpha = float(alpha)
        self.prev = None  # np.ndarray [N,2]

    def apply(self, lm, w, h):
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        if self.prev is None:
            self.prev = pts
            return pts
        smoothed = self.alpha * self.prev + (1.0 - self.alpha) * pts
        self.prev = smoothed
        return smoothed

# ========= Drawing =========
def draw_body_skeleton(frame, pts):
    for a, b in BODY_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            x1, y1 = int(pts[a,0]), int(pts[a,1])
            x2, y2 = int(pts[b,0]), int(pts[b,1])
            cv2.line(frame, (x1,y1), (x2,y2), SKELETON_COLOR, SKELETON_THICK, cv2.LINE_AA)
    for idx in BODY_LANDMARKS:
        if idx < len(pts):
            x, y = int(pts[idx,0]), int(pts[idx,1])
            cv2.circle(frame, (x,y), JOINT_RADIUS, SKELETON_COLOR, -1, cv2.LINE_AA)
    return frame

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
    h, w, _ = frame.shape
    top_h    = int(h * TOP_BAR_FRAC)
    bottom_h = int(h * BOTTOM_BAR_FRAC)

    # פס עליון
    top = frame.copy()
    cv2.rectangle(top, (0,0), (w, top_h), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_OPACITY, frame, 1-BAR_BG_OPACITY, 0)

    # Reps
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    draw.text((PADDING, int(top_h*0.22)), f"Reps: {reps}", font=REPS_FONT, fill=(255,255,255))

    # דונאט
    radius = int(top_h * DEPTH_RADIUS_SCALE / 2)
    thick  = max(3, int(radius * DEPTH_THICKNESS_FRAC))
    cx = w - PADDING - radius
    cy = int(top_h * 0.60)
    frame = np.array(pil)
    frame = draw_depth_donut(frame, (cx, cy), radius, thick, depth_pct)

    # טקסטים בתוך הדונאט
    pil  = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"
    pct_txt   = f"{int(depth_pct*100)}%"
    max_text_w = int(radius * 1.6)
    label_font = fit_text_to_width(draw, label_txt, max_text_w, DEPTH_LABEL_PX, 12)
    pct_font   = fit_text_to_width(draw, pct_txt,   max_text_w, DEPTH_PCT_PX, 14)

    label_w = draw.textlength(label_txt, font=label_font)
    pct_w   = draw.textlength(pct_txt,   font=pct_font)
    gap    = max(2, int(radius * 0.10))
    block_h = label_font.size + gap + pct_font.size
    base_y  = cy - block_h // 2
    draw.text((cx - int(label_w//2), base_y), label_txt, font=label_font, fill=(255,255,255))
    draw.text((cx - int(pct_w//2),   base_y + label_font.size + gap), pct_txt, font=pct_font, fill=(255,255,255))
    frame = np.array(pil)

    # פס תחתון לפידבק
    if feedback:
        over = frame.copy()
        cv2.rectangle(over, (0, h-bottom_h), (w, h), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_OPACITY, frame, 1-BAR_BG_OPACITY, 0)
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        max_w = w - 2*PADDING
        fb_text = feedback
        fb_font = fit_text_to_width(draw, fb_text, max_w, FEEDBACK_BASE_PX, FEEDBACK_MIN_PX)
        tw = draw.textlength(fb_text, font=fb_font)
        tx = (w - int(tw)) // 2
        ty = h - bottom_h + max(4, (bottom_h - fb_font.size)//2)
        draw.text((tx, ty), fb_text, font=fb_font, fill=(255,255,255))
        frame = np.array(pil)

    return frame

# ========= Helpers =========
def _format_score_value(x: float):
    x = round(x * 2) / 2
    return int(x) if float(x).is_integer() else round(x, 1)

# ================== Main ==================
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

    # לפר־חזרה (לתיקון הפופ־אפ)
    rep_reports = []

    stage = "top"  # "down" / "bottom" / "top"
    frames_down = frames_top = 0

    rep_min_knee_angle = 180.0
    start_knee_angle   = None
    last_rep_feedback = []

    depth_smoother = DepthSmoother()
    lmk_smoother   = LandmarkSmoother(alpha=0.70)

    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # כשאין שלד – Overlay בלבד
            if not results.pose_landmarks:
                depth_smoother.update(dt)
                overlay_text = " | ".join(last_rep_feedback) if last_rep_feedback else None
                frame = draw_overlay(frame, reps=counter, feedback=overlay_text, depth_pct=depth_smoother.display)
                out.write(frame)
                continue

            try:
                lm = results.pose_landmarks.landmark
                pts = lmk_smoother.apply(lm, w, h)
                frame = draw_body_skeleton(frame, pts)

                r = mp_pose.PoseLandmark
                hip_y     = pts[r.RIGHT_HIP.value,1]/h
                knee_y    = pts[r.RIGHT_KNEE.value,1]/h
                ankle_y   = pts[r.RIGHT_ANKLE.value,1]/h
                shoulder_y= pts[r.RIGHT_SHOULDER.value,1]/h
                heel_y    = results.pose_landmarks.landmark[r.RIGHT_HEEL.value].y
                foot_y    = results.pose_landmarks.landmark[r.RIGHT_FOOT_INDEX.value].y

                hip = [results.pose_landmarks.landmark[r.RIGHT_HIP.value].x, hip_y]
                knee= [results.pose_landmarks.landmark[r.RIGHT_KNEE.value].x, knee_y]
                ankle=[results.pose_landmarks.landmark[r.RIGHT_ANKLE.value].x, ankle_y]
                shoulder=[results.pose_landmarks.landmark[r.RIGHT_SHOULDER.value].x, shoulder_y]

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)
                body_angle = calculate_body_angle(shoulder, hip)
                heel_lifted = (foot_y - heel_y) > 0.03

                # ========= מצב מכונה =========
                if stage in ("top","bottom") and knee_angle < KNEE_DOWN_TRIG:
                    stage = "down"
                    frames_down = 0
                    last_rep_feedback = []
                    start_knee_angle = float(knee_angle)
                    rep_min_knee_angle = 180.0

                if stage == "down":
                    frames_down += 1
                    rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)

                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE))
                        depth_target = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))
                        depth_smoother.set_target(depth_target)

                    if knee_angle <= KNEE_BOTTOM_ZONE:
                        stage = "bottom"

                elif stage == "bottom" and knee_angle > KNEE_UP_TRIG:
                    stage = "top"
                    frames_top = 0

                    # חזרה מסתיימת רק אם היינו מספיק פריימים ב-down
                    if frames_down >= MIN_FRAMES_DOWN:
                        feedbacks = []
                        penalty = 0

                        # עומק (hip->heel)
                        hip_to_heel_dist = abs(hip[1] - heel_y)
                        if hip_to_heel_dist > 0.48:
                            feedbacks.append("Too shallow — squat lower")
                            penalty += 3
                        elif hip_to_heel_dist > 0.45:
                            feedbacks.append("Almost there — go a bit lower")
                            penalty += 1.5
                        elif hip_to_heel_dist > 0.43:
                            feedbacks.append("Looking good — just a bit more depth")
                            penalty += 0.5

                        # גב למעלה
                        if back_angle < 140:
                            feedbacks.append("Try to straighten your back more at the top")
                            penalty += 1

                        # עקבים
                        if heel_lifted:
                            feedbacks.append("Keep your heels down")
                            penalty += 1

                        # נעילה מלאה
                        if knee_angle < KNEE_UP_TRIG:
                            feedbacks.append("Incomplete lockout")
                            penalty += 1

                        # ניקוד
                        if not feedbacks:
                            score = 10.0
                        else:
                            any_feedback_session = True
                            penalty = min(penalty, 6)
                            score = round(max(4, 10 - penalty) * 2) / 2

                        # צבירת פידבק ייחודי לסשן
                        for f in feedbacks:
                            if f not in overall_feedback:
                                overall_feedback.append(f)
                        last_rep_feedback = feedbacks[:]

                        # אפקט דונאט
                        depth_smoother.peak_hold(DEPTH_PEAK_HOLD_SEC)
                        depth_smoother.start_fade(DEPTH_FADE_SEC)

                        # ====== NEW: הוספת רשומת חזרה ל־UI ======
                        if start_knee_angle is not None:
                            denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE))
                            depth_final = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))
                        else:
                            depth_final = float(depth_smoother.display)

                        rep_reports.append({
                            "rep_index": int(counter + 1),                 # לפני הגדלת המונה
                            "score": float(round(score, 1)),
                            "feedback": last_rep_feedback[:] if last_rep_feedback else [],
                            "depth_pct": depth_final
                        })
                        # ========================================

                        # מונים
                        counter += 1
                        if score >= 9.5:
                            good_reps += 1
                        else:
                            bad_reps += 1
                            problem_reps.append(counter)
                        all_scores.append(score)

                        start_knee_angle = None

                elif stage == "top":
                    frames_top += 1

                # עדכון הדונאט
                depth_smoother.update(dt)

            except Exception:
                # לא מפיל את ההרצה על פריים "רע"
                pass

            # Overlay
            overlay_text = " | ".join(last_rep_feedback) if last_rep_feedback else None
            frame = draw_overlay(frame, reps=counter, feedback=overlay_text, depth_pct=depth_smoother.display)
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # ========== תוצאות ==========
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0.0
    # הצגה עקבית כ-float:
    technique_score_display = float(_format_score_value(technique_score))
    if any_feedback_session and technique_score_display == 10.0:
        technique_score_display = 9.5

    if not overall_feedback:
        overall_feedback.append("Great form! Keep it up 💪")

    # קובץ תקציר
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {int(counter)}\n")
            f.write(f"Technique Score: {technique_score_display}/10\n")
            if overall_feedback:
                f.write("Feedback:\n")
                for fb in overall_feedback:
                    f.write(f"- {fb}\n")
    except Exception:
        pass

    # קידוד MP4 (faststart) עם fallback בטוח (למקרה שתצטרך)
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    video_path_return = output_path
    try:
        proc = subprocess.run([
            "ffmpeg","-y","-i",output_path,
            "-c:v","libx264","-preset","fast","-movflags","+faststart","-pix_fmt","yuv420p",
            encoded_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and os.path.exists(encoded_path) and os.path.getsize(encoded_path) > 0:
            video_path_return = encoded_path
            if os.path.exists(output_path):
                os.remove(output_path)
        else:
            if os.path.exists(encoded_path) and os.path.getsize(encoded_path) == 0:
                os.remove(encoded_path)
    except Exception:
        pass

    # החזרה: שדות בטיפוסים "בטוחים" + reps מלא, לא ריק
    return {
        "technique_score": float(technique_score_display),
        "squat_count": int(counter),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": overall_feedback or ["Great form! Keep it up 💪"],
        "problem_reps": [int(i) for i in problem_reps],
        "reps": rep_reports,                   # <<< קריטי לפופ־אפ
        "video_path": video_path_return,       # לא משפיע על הפופ־אפ, אבל בטוח יותר
        "feedback_path": feedback_path
    }


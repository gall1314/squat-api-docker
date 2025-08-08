import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from utils import calculate_angle, calculate_body_angle
from PIL import ImageFont, ImageDraw, Image

# ====== סגנון גרפי ======
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22

try:
    REPS_FONT = ImageFont.truetype(FONT_PATH, REPS_FONT_SIZE)
    FEEDBACK_FONT = ImageFont.truetype(FONT_PATH, FEEDBACK_FONT_SIZE)
except Exception:
    REPS_FONT = ImageFont.load_default()
    FEEDBACK_FONT = ImageFont.load_default()

def draw_plain_text(pil_img, xy, text, font, color=(255,255,255)):
    ImageDraw.Draw(pil_img).text((int(xy[0]), int(xy[1])), text, font=font, fill=color)
    return np.array(pil_img)

def draw_overlay(frame, reps=0, feedback=None):
    """פס עליון עם ספירה, פס תחתון עם פידבק (אם יש)."""
    h, w, _ = frame.shape
    bar_h = int(h * 0.065)

    # פס עליון שקוף + Reps
    top = frame.copy()
    cv2.rectangle(top, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, 0.55, frame, 0.45, 0)
    pil = Image.fromarray(frame)
    frame = draw_plain_text(pil, (16, int(bar_h*0.22)), f"Reps: {reps}", REPS_FONT)

    # פס תחתון לפידבק (רק אם יש)
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

# עיגול לחצאים והצגת מספר שלם בלי .0
def _format_score_value(x: float):
    x = round(x * 2) / 2
    return int(x) if float(x).is_integer() else round(x, 1)

def run_analysis(video_path, frame_skip=3, scale=0.4,
                 output_path="squat_output.mp4",
                 feedback_path="squat_feedback.txt"):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []          # רשימת הערות ייחודיות לכל האימון
    depth_distances = []           # לשיקלול עומק ממוצע אם לא היו הערות כלל
    stage = None

    # מעקבים שונים (נשארים כמו בקוד שלך, לשימור הלוגיקה)
    rep_min_knee_angle = 180
    max_lean_down = 0
    top_back_angle = 0

    # הפידבק שיוצג על המסך – תמיד זה של סוף החזרה האחרונה (או ריק)
    last_rep_feedback = []

    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                overlay_text = " | ".join(last_rep_feedback) if last_rep_feedback else None
                frame = draw_overlay(frame, reps=counter, feedback=overlay_text)
                out.write(frame)
                continue

            try:
                lm = results.pose_landmarks.landmark
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                heel_y = lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
                foot_y = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)
                body_angle = calculate_body_angle(shoulder, hip)
                heel_lifted = foot_y - heel_y > 0.03

                # שלבי תנועה
                if knee_angle < 90:
                    stage = "down"
                if stage == "down":
                    rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                    max_lean_down = max(max_lean_down, body_angle)
                if stage == "up":
                    top_back_angle = max(top_back_angle, body_angle)

                # סוף חזרה
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    feedbacks = []
                    penalty = 0

                    # עומק יחסי לפי לוגיקה המקורית שלך
                    hip_to_heel_dist = abs(hip[1] - heel_y)
                    depth_distances.append(hip_to_heel_dist)

                    if hip_to_heel_dist > 0.48:
                        feedbacks.append("Too shallow — squat lower")
                        depth_penalty = 3
                    elif hip_to_heel_dist > 0.45:
                        feedbacks.append("Almost there — go a bit lower")
                        depth_penalty = 1.5
                    elif hip_to_heel_dist > 0.43:
                        feedbacks.append("Looking good — just a bit more depth")
                        depth_penalty = 0.5
                    else:
                        depth_penalty = 0
                    penalty += depth_penalty

                    # גב למעלה
                    if back_angle < 140:
                        feedbacks.append("Try to straighten your back more at the top")
                        penalty += 1

                    # עקבים
                    if heel_lifted:
                        feedbacks.append("Keep your heels down")
                        penalty += 1

                    # נעילה מלאה
                    if knee_angle < 160:
                        feedbacks.append("Incomplete lockout")
                        penalty += 1

                    # ניקוד חזרה
                    if not feedbacks:
                        score = 10.0
                    else:
                        penalty = min(penalty, 6)
                        score = round(max(4, 10 - penalty) * 2) / 2

                    # עדכון דו"ח: אוסף הערות ייחודי
                    for f in feedbacks:
                        if f not in overall_feedback:
                            overall_feedback.append(f)

                    # מה שמופיע על המסך עד החזרה הבאה
                    last_rep_feedback = feedbacks[:]  # אם ריק – לא יוצג טקסט

                    # עדכוני מונים
                    counter += 1
                    if score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                        problem_reps.append(counter)
                    all_scores.append(score)

                    # אפס מעקבים לחזרה הבאה
                    rep_min_knee_angle = 180
                    max_lean_down = 0

            except Exception:
                # לא מפיל ריצה על פריים בעייתי
                pass

            # ציור Overlay עם הפידבק של החזרה האחרונה בלבד
            overlay_text = " | ".join(last_rep_feedback) if last_rep_feedback else None
            frame = draw_overlay(frame, reps=counter, feedback=overlay_text)
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # ציון סופי
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0.0
    if counter > 0 and not overall_feedback:
        # לא היו הערות בכלל -> 10
        technique_score = 10.0
    technique_score_display = _format_score_value(technique_score)

    # אם לא היו הערות בכלל, אפשר עדיין לתת רמז עומק ממוצע (אופציונלי)
    if depth_distances and not overall_feedback:
        avg_depth = np.mean(depth_distances)
        # לא מוסיפים overall_feedback כדי לשמור על "ללא הערות" = 10,
        # אבל אם תרצה להציג טיפ לא-מוריד-ציון, אפשר להחזיר בשדה tips נוסף.

    if not overall_feedback:
        # מסר כללי חיובי כשאין הערות
        overall_feedback.append("Great form! Keep it up 💪")

    # כתיבת קובץ פידבק
    with open(feedback_path, "w", encoding="utf-8") as f:
        f.write(f"Total Reps: {counter}\n")
        f.write(f"Technique Score: {technique_score_display}/10\n")
        if overall_feedback:
            f.write("Feedback:\n")
            for fb in overall_feedback:
                f.write(f"- {fb}\n")

    # קידוד ל-mp4 תואם
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
        "technique_score": technique_score_display,  # כבר בפורמט 10 / 8.5 וכו'
        "squat_count": counter,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": overall_feedback,
        "problem_reps": problem_reps,
        "video_path": encoded_path,
        "feedback_path": feedback_path
    }


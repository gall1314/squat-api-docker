import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from utils import calculate_angle, calculate_body_angle
from PIL import ImageFont, ImageDraw, Image

# ========= Debug / Version tag =========
VERSION_TAG = "SQUAT_v3_sync_2025-08-08"

# ========= Fonts / Overlay =========
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

def draw_overlay(frame, reps=0, feedback=None, debug_tag=None):
    """
    פס עליון: Reps ; פס תחתון: פידבק החזרה האחרונה (אם יש).
    debug_tag: טקסט קטן בפינה שמאל-עליון (לבדיקת גרסה שרצה).
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

    # תג גרסה קטן לשמאל-עליון (ויזואלי, לא מפריע)
    if debug_tag:
        cv2.putText(frame, debug_tag, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1, cv2.LINE_AA)

    return frame

# עיגול תצוגה לחצאים והסרת .0 כששלם
def _format_score_value(x: float):
    x = round(x * 2) / 2
    return int(x) if float(x).is_integer() else round(x, 1)

def run_analysis(video_path, frame_skip=3, scale=0.4,
                 output_path="squat_output.mp4",
                 feedback_path="squat_feedback.txt"):
    print(f"[{VERSION_TAG}] run_analysis() start, video={video_path}")
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{VERSION_TAG}] ERROR: Could not open video")
        return {"error": "Could not open video", "debug_version": VERSION_TAG}

    # מונים / צבירת תוצאות
    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []   # הערות ייחודיות לכל האימון
    depth_distances = []    # לשימוש עתידי במידת הצורך
    stage = None

    # מעקבים (משאירים כמו המקור – לא שוברים לוגיקה)
    rep_min_knee_angle = 180
    max_lean_down = 0
    top_back_angle = 0

    # מה שיוצג בסרטון – תמיד פידבק של סוף החזרה האחרונה בלבד
    last_rep_feedback = []

    # וידאו יצוא
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
                frame = draw_overlay(frame, reps=counter, feedback=overlay_text, debug_tag=VERSION_TAG)
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

                # התחלת ירידה – מנקים פידבק למסך עד סוף החזרה
                if knee_angle < 90:
                    if stage != "down":
                        last_rep_feedback = []
                    stage = "down"

                # מעקבים (כמו במקור)
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

                    # עומק (לפי המקור)
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

                    # דו"ח – הערות ייחודיות
                    for f in feedbacks:
                        if f not in overall_feedback:
                            overall_feedback.append(f)

                    # זה מה שיוצג בווידאו עד סוף החזרה הבאה
                    last_rep_feedback = feedbacks[:]  # ריק -> לא יוצג טקסט

                    # מונים
                    counter += 1
                    if score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                        problem_reps.append(counter)
                    all_scores.append(score)

                    print(f"[{VERSION_TAG}] rep {counter}: score={score}, fb={feedbacks}")

                    # reset
                    rep_min_knee_angle = 180
                    max_lean_down = 0

            except Exception as e:
                print(f"[{VERSION_TAG}] WARN frame error: {e}")

            # ציור Overlay – רק פידבק סוף-חזרה אחרונה
            overlay_text = " | ".join(last_rep_feedback) if last_rep_feedback else None
            frame = draw_overlay(frame, reps=counter, feedback=overlay_text, debug_tag=VERSION_TAG)
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # ציון סופי (חצאים) + כפייה ל-10 אם אין הערות בכלל
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0.0
    if counter > 0 and not overall_feedback:
        technique_score = 10.0
    technique_score_display = _format_score_value(technique_score)

    # מסר חיובי כשאין הערות
    if not overall_feedback:
        overall_feedback.append("Great form! Keep it up 💪")

    # קובץ תקציר
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Version: {VERSION_TAG}\n")
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score_display}/10\n")
            if overall_feedback:
                f.write("Feedback:\n")
                for fb in overall_feedback:
                    f.write(f"- {fb}\n")
    except Exception as e:
        print(f"[{VERSION_TAG}] WARN writing feedback file: {e}")

    # קידוד MP4 (faststart)
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            encoded_path
        ], check=False)
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception as e:
        print(f"[{VERSION_TAG}] WARN ffmpeg encode: {e}")

    print(f"[{VERSION_TAG}] done. reps={counter}, score={technique_score_display}, video={encoded_path}")

    return {
        "technique_score": technique_score_display,  # 10 / 9.5 / 9 / ...
        "squat_count": counter,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": overall_feedback,
        "problem_reps": problem_reps,
        "video_path": encoded_path,
        "feedback_path": feedback_path,
        "debug_version": VERSION_TAG
    }


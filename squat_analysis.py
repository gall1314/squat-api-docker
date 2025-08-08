import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from utils import calculate_angle, calculate_body_angle
from PIL import ImageFont, ImageDraw, Image

# ========= Debug / Version tag =========
VERSION_TAG = "SQUAT_v4_bulgarian_ui_sync"

# ========= Fonts / Overlay =========
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE   = 18

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

# ========= Donut style (כמו בבולגרי) =========
DEPTH_RADIUS_SCALE   = 0.70
DEPTH_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)   # BGR
DEPTH_RING_BG        = (70, 70, 70)    # BGR

# עומק יעד “מושלם” לנרמול ברך (כמו בבולגרי)
PERFECT_MIN_KNEE = 70

def draw_plain_text(pil_img, xy, text, font, color=(255,255,255)):
    ImageDraw.Draw(pil_img).text((int(xy[0]), int(xy[1])), text, font=font, fill=color)
    return np.array(pil_img)

def draw_depth_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius   = int(radius)
    thickness = int(thickness)

    # טבעת רקע
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    # מילוי
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0, debug_tag=None):
    """
    פס עליון: Reps + DEPTH donut ; פס תחתון: פידבק של סוף-חזרה אחרונה (אם יש).
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

    # טקסטים בתוך הדונאט
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

    # תג גרסה קטן לשמאל-עליון
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

    # מונים / תוצאות
    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []   # הערות ייחודיות לכל האימון
    any_feedback_session = False

    # שלבים ומדדים
    stage = None
    rep_min_knee_angle = 180
    max_lean_down = 0
    top_back_angle = 0

    # עומק חי להצגה (כמו בבולגרי)
    start_knee_angle = None
    last_depth_for_ui = 0.0

    # הפידבק שמוצג על המסך – של סוף החזרה האחרונה בלבד
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

            # כשאין שלד – שומרים עומק 0 וממשיכים
            if not results.pose_landmarks:
                overlay_text = " | ".join(last_rep_feedback) if last_rep_feedback else None
                frame = draw_overlay(frame, reps=counter, feedback=overlay_text,
                                     depth_pct=0.0, debug_tag=VERSION_TAG)
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

                # התחלת ירידה – איפוס פידבק למסך והגדרת baseline עומק
                if knee_angle < 90:
                    if stage != "down":
                        last_rep_feedback = []
                        start_knee_angle = float(knee_angle)
                        rep_min_knee_angle = 180
                        last_depth_for_ui = 0.0
                    stage = "down"

                # מעקבים (כמו המקור)
                if stage == "down":
                    rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                    max_lean_down = max(max_lean_down, body_angle)

                    # עומק חי להצגה (כמו בבולגרי)
                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE))
                        last_depth_for_ui = float(np.clip(
                            (start_knee_angle - rep_min_knee_angle) / denom, 0, 1
                        ))

                if stage == "up":
                    top_back_angle = max(top_back_angle, body_angle)

                # סוף חזרה
                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    feedbacks = []
                    penalty = 0

                    # עומק לפי הקוד שלך (hip->heel) – לדרוג פידבק; הדונאט מוצג לפי זווית ברך
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
                    if knee_angle < 160:
                        feedbacks.append("Incomplete lockout")
                        penalty += 1

                    # ניקוד חזרה
                    if not feedbacks:
                        score = 10.0
                    else:
                        any_feedback_session = True
                        penalty = min(penalty, 6)
                        score = round(max(4, 10 - penalty) * 2) / 2

                    # דו"ח – הערות ייחודיות
                    for f in feedbacks:
                        if f not in overall_feedback:
                            overall_feedback.append(f)

                    # מה שיוצג בווידאו עד סוף החזרה הבאה
                    last_rep_feedback = feedbacks[:]  # ריק -> אין טקסט
                    # אפס עומק להצגה (כדי לא להיתקע על 100% בין חזרות)
                    last_depth_for_ui = 0.0
                    start_knee_angle = None

                    # מונים
                    counter += 1
                    if score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                        problem_reps.append(counter)

                    all_scores.append(score)
                    print(f"[{VERSION_TAG}] rep {counter}: score={score}, fb={feedbacks}")

            except Exception as e:
                print(f"[{VERSION_TAG}] WARN frame error: {e}")

            # ציור Overlay – דונאט עומק חי + פידבק של סוף-חזרה אחרונה בלבד
            overlay_text = " | ".join(last_rep_feedback) if last_rep_feedback else None
            frame = draw_overlay(frame, reps=counter, feedback=overlay_text,
                                 depth_pct=last_depth_for_ui, debug_tag=VERSION_TAG)
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # ציון סופי (חצאים) + אל תשאיר 10 אם היו הערות כלשהן
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0.0
    if (len(all_scores) > 0) and (not any_feedback_session) and technique_score < 10.0:
        # אם באמת לא היו הערות אבל ממוצע לא 10 (למשל 9.5) – נשאיר ככה
        pass
    if any_feedback_session and technique_score == 10.0:
        technique_score = 9.5  # אין 10 אם היו הערות כלשהן

    technique_score_display = _format_score_value(technique_score)

    if not overall_feedback:
        overall_feedback.append("Great form! Keep it up 💪")

    # כתיבת קובץ פידבק
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

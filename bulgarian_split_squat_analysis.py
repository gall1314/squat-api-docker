import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
import time
from PIL import ImageFont, ImageDraw, Image

# ========== הגדרות ==========
FONT_PATH = "C:/Users/lenovo/Downloads/Roboto/Roboto-Bold.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
LINE_SPACING = 6
FEEDBACK_DURATION = 2.0  # זמן תצוגת הפידבק

ANGLE_DOWN_THRESH = 95
ANGLE_UP_THRESH = 160
GOOD_REP_MIN_SCORE = 8.0
TORSO_LEAN_MIN = 135
VALGUS_X_TOL = 0.03

mp_pose = mp.solutions.pose


# ========== Overlay Drawing ==========
def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        w, _ = draw.textsize(test_line, font=font)
        if w <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines

def draw_overlay(frame, reps, feedback, last_feedback_time, current_time):
    h, w, _ = frame.shape
    bar_height = int(h * 0.07)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    reps_font = ImageFont.truetype(FONT_PATH, REPS_FONT_SIZE)
    draw.text((20, int(bar_height * 0.2)), f"Reps: {reps}", font=reps_font, fill=(255, 255, 255))

    if feedback and (current_time - last_feedback_time < FEEDBACK_DURATION):
        feedback_font = ImageFont.truetype(FONT_PATH, FEEDBACK_FONT_SIZE)
        max_text_width = int(w * 0.9)
        feedback_lines = wrap_text(feedback, feedback_font, max_text_width, draw)
        total_text_height = sum([feedback_font.getsize(line)[1] for line in feedback_lines]) + (len(feedback_lines)-1) * LINE_SPACING
        bottom_box_height = total_text_height + 20
        box_top_y = h - bottom_box_height

        overlay = np.array(pil_img)
        cv2.rectangle(overlay, (0, box_top_y), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, np.array(pil_img), 0.4, 0)
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        y = box_top_y + 10
        for line in feedback_lines:
            line_width, _ = draw.textsize(line, font=feedback_font)
            x = (w - line_width) // 2
            draw.text((x, y), line, font=feedback_font, fill=(255, 255, 255))
            y += feedback_font.getsize(line)[1] + LINE_SPACING

        frame = np.array(pil_img)

    return frame


# ========== העוזר שלך לספירת חזרות ==========
# (החלק הזה נשאר כמו שהיה – ללא שינוי)
# ...
# (כל המחלקות BulgarianRepCounter וכו' נשארות זהות)


# ========== Main analysis function ==========
def run_bulgarian_analysis(video_path, frame_skip=1, scale=1.0, output_path="analyzed_output.mp4", feedback_path="feedback_summary.txt"):
    cap = cv2.VideoCapture(video_path)
    counter = BulgarianRepCounter()
    frame_no = 0
    active_leg = None
    out = None
    last_feedback = ""
    last_feedback_time = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

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
            out = cv2.VideoWriter(output_path, fourcc, 8, (w, h))

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

        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        feedbacks = []
        if counter.stage == "down":
            if torso_angle < TORSO_LEAN_MIN:
                feedbacks.append("Keep your back straight")
            if not v_ok:
                feedbacks.append("Avoid knee collapse")

        if feedbacks:
            last_feedback = " ".join(feedbacks)
            last_feedback_time = time.time()

        frame = draw_overlay(
            frame,
            reps=counter.count,
            feedback=last_feedback,
            last_feedback_time=last_feedback_time,
            current_time=time.time()
        )

        out.write(frame)

    pose.close()
    cap.release()
    out.release()
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
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        encoded_path
    ])
    os.remove(output_path)

    return {
        **result,
        "video_path": encoded_path,
        "feedback_path": feedback_path
    }




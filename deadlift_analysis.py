import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle, calculate_body_angle

def run_deadlift_analysis(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []

    stage = None
    min_knee_angle = 180

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_index % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                continue

            try:
                lm = results.pose_landmarks.landmark
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                knee_angle = calculate_angle(hip, knee, ankle)
                body_angle = calculate_body_angle(shoulder, hip)

                if stage is None and knee_angle < 150:
                    stage = "down"
                    min_knee_angle = knee_angle

                elif stage == "down" and knee_angle > 165:
                    stage = "up"

                if stage == "down":
                    min_knee_angle = min(min_knee_angle, knee_angle)

                if stage == "up":
                    feedbacks = []
                    penalty = 0

                    # חזרה שטחית מידי לא נחשבת
                    if min_knee_angle > 110:
                        feedbacks.append("Didn't go deep enough — not counted")
                        stage = None
                        continue

                    # ניקוד לפי זווית גב ונעילה
                    if min_knee_angle < 120:
                        feedbacks.append("Try to keep your back straighter")
                        penalty += 2

                    if knee_angle < 170:
                        feedbacks.append("Finish your rep with a full lockout")
                        penalty += 1.5

                    penalty = min(penalty, 6)
                    score = round(max(4, 10 - penalty) * 2) / 2

                    for f in feedbacks:
                        if f not in overall_feedback:
                            overall_feedback.append(f)

                    counter += 1
                    if score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                        problem_reps.append(counter)
                    all_scores.append(score)

                    stage = None
                    min_knee_angle = 180

            except Exception:
                continue

    cap.release()
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0

    if not overall_feedback:
        overall_feedback.append("Great form! Keep it up 💪")

    return {
        "technique_score": technique_score,
        "squat_count": counter,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": overall_feedback,
        "problem_reps": problem_reps,
    }


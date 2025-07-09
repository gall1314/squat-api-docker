
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
    min_body_angle = 180
    top_body_angle = 0
    frame_index = 0

    with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose:
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

                body_angle = calculate_body_angle(shoulder, hip)
                knee_angle = calculate_angle(hip, knee, ankle)

                if stage is None and knee_angle < 130:
                    stage = "down"
                    min_body_angle = body_angle
                    top_body_angle = 0

                elif stage == "down" and knee_angle > 158:
                    stage = "up"

                    feedbacks = []
                    penalty = 0

                    if min_body_angle < 125:
                        feedbacks.append("Try to keep your back straighter")
                        penalty += 1.5

                    if body_angle < 160:
                        feedbacks.append("Finish with a full lockout")
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


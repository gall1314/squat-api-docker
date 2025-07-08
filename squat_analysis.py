import cv2
import mediapipe as mp

from utils import calculate_angle, calculate_body_angle
import numpy as np
from .helpers import calculate_angle, calculate_body_angle  # אם תוציא אותם החוצה בהמשך

def run_analysis(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []
    depth_distances = []
    depth_feedback_given = False

    stage = None
    rep_min_knee_angle = 180
    max_lean_down = 0
    top_back_angle = 0

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
                heel_y = lm[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
                foot_y = lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)
                body_angle = calculate_body_angle(shoulder, hip)
                heel_lifted = foot_y - heel_y > 0.03

                if knee_angle < 90:
                    stage = "down"
                if stage == "down":
                    rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                    max_lean_down = max(max_lean_down, body_angle)
                if stage == "up":
                    top_back_angle = max(top_back_angle, body_angle)

                if knee_angle > 160 and stage == "down":
                    stage = "up"
                    feedbacks = []
                    penalty = 0

                    hip_to_heel_dist = abs(hip[1] - heel_y)
                    depth_distances.append(hip_to_heel_dist)

                    if hip_to_heel_dist > 0.48:
                        if not depth_feedback_given:
                            feedbacks.append("Too shallow — squat lower")
                            depth_feedback_given = True
                        depth_penalty = 3
                    elif hip_to_heel_dist > 0.45:
                        if not depth_feedback_given:
                            feedbacks.append("Almost there — go a bit lower")
                            depth_feedback_given = True
                        depth_penalty = 1.5
                    elif hip_to_heel_dist > 0.43:
                        if not depth_feedback_given:
                            feedbacks.append("Looking good — just a bit more depth")
                            depth_feedback_given = True
                        depth_penalty = 0.5
                    else:
                        depth_penalty = 0

                    penalty += depth_penalty

                    if stage == "up" and back_angle < 140:
                        feedbacks.append("Try to straighten your back more at the top")
                        penalty += 1

                    if heel_lifted:
                        feedbacks.append("Keep your heels down")
                        penalty += 1

                    if knee_angle < 160:
                        feedbacks.append("Incomplete lockout")
                        penalty += 1

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

                    rep_min_knee_angle = 180
                    max_lean_down = 0

            except Exception:
                continue

    cap.release()
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0

    if depth_distances and not depth_feedback_given:
        avg_depth = np.mean(depth_distances)
        if avg_depth > 0.48:
            overall_feedback.append("Try squatting lower")
        elif avg_depth > 0.45:
            overall_feedback.append("Almost there — go a bit lower")
        elif avg_depth > 0.43:
            overall_feedback.append("Looking good — just a bit more depth")

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

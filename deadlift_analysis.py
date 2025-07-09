import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle

def run_deadlift_analysis(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []

    rep_in_progress = False
    frame_index = 0
    last_rep_frame = -999
    MIN_FRAMES_BETWEEN_REPS = 10
    max_delta_x = 0
    max_curvature = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

                shoulder = np.array([
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                ])
                hip = np.array([
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                ])
                knee = np.array([
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
                ])
                ankle = np.array([
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                ])

                # אוזן ימין או שמאל – מה שזמין
                ear_r = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
                ear_l = lm[mp_pose.PoseLandmark.LEFT_EAR.value]

                if ear_r.visibility > 0.5:
                    head_point = np.array([ear_r.x, ear_r.y])
                elif ear_l.visibility > 0.5:
                    head_point = np.array([ear_l.x, ear_l.y])
                else:
                    continue  # אין ראש ברור

                delta_x = abs(hip[0] - shoulder[0])
                knee_angle = calculate_angle(hip, knee, ankle)

                # curvature = סטיית הראש מקו כתף–ירך
                v = shoulder - hip
                v_norm = v / np.linalg.norm(v)
                v_head = head_point - hip
                proj = np.dot(v_head, v_norm) * v_norm
                perp = v_head - proj
                curvature = np.linalg.norm(perp)

                if rep_in_progress:
                    max_delta_x = max(max_delta_x, delta_x)
                    min_knee_angle = min(min_knee_angle, knee_angle)
                    max_curvature = max(max_curvature, curvature)

                if not rep_in_progress:
                    if delta_x > 0.08:
                        rep_in_progress = True
                        max_delta_x = delta_x
                        min_knee_angle = knee_angle
                        max_curvature = curvature

                elif rep_in_progress and delta_x < 0.035:
                    if frame_index - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        delta_range = max_delta_x - delta_x

                        if delta_range > 0.05 and min_knee_angle < 160:
                            feedbacks = []
                            penalty = 0

                            if delta_x > 0.05:
                                feedbacks.append("Try to finish more upright")
                                penalty += 1

                            if max_delta_x > 0.18 and min_knee_angle > 170:
                                feedbacks.append("Try to bend your knees as you lean forward")
                                penalty += 1

                            if min_knee_angle > 165 and max_delta_x > 0.2:
                                feedbacks.append("Try to lift your chest and hips together")
                                penalty += 1

                            # 🌡 מדרוג גב עגול לפי curvature
                            if 0.105 < max_curvature <= 0.11:
                                feedbacks.append("Try to keep your back a bit straighter")
                                penalty += 1.5
                            elif 0.11 < max_curvature <= 0.13:
                                feedbacks.append("Your back is rounding too much – focus on spinal alignment")
                                penalty += 2.5
                            elif max_curvature > 0.13:
                                feedbacks.append("Dangerous back rounding – stop and fix your form!")
                                penalty += 3.5

                            score = round(max(4, 10 - penalty) * 2) / 2
                            for f in feedbacks:
                                if f not in overall_feedback:
                                    overall_feedback.append(f)

                            counter += 1
                            last_rep_frame = frame_index
                            if score >= 9.5:
                                good_reps += 1
                            else:
                                bad_reps += 1
                                problem_reps.append(counter)
                            all_scores.append(score)

                    rep_in_progress = False
                    max_delta_x = 0
                    max_curvature = 0

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


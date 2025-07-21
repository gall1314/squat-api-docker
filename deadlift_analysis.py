import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def analyze_back_curvature(shoulder, hip, reference_point, threshold=0.04):
    line_vec = hip - shoulder
    line_unit = line_vec / np.linalg.norm(line_vec)
    proj_len = np.dot(reference_point - shoulder, line_unit)
    proj_point = shoulder + proj_len * line_unit
    offset_vec = reference_point - proj_point
    direction_sign = np.sign(offset_vec[1]) * -1  # inward = negative
    curvature_magnitude = np.linalg.norm(offset_vec)
    signed_curvature = direction_sign * curvature_magnitude
    is_dangerous = signed_curvature < -threshold
    return signed_curvature, is_dangerous

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
    min_knee_angle = 999
    spine_curvature = 0
    spine_flagged = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
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

                shoulder = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                     lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
                hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
                knee = np.array([lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                 lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
                ankle = np.array([lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                  lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

                head_candidates = [
                    lm[mp_pose.PoseLandmark.RIGHT_EAR.value],
                    lm[mp_pose.PoseLandmark.LEFT_EAR.value],
                    lm[mp_pose.PoseLandmark.NOSE.value]
                ]
                head_point = None
                for candidate in head_candidates:
                    if candidate.visibility > 0.5:
                        head_point = np.array([candidate.x, candidate.y])
                        break
                if head_point is None:
                    continue

                delta_x = abs(hip[0] - shoulder[0])
                knee_angle = calculate_angle(hip, knee, ankle)

                # אמצע גב משוקלל – נוטה לכיוון הראש
                mid_spine = (shoulder + hip) * 0.5 * 0.4 + head_point * 0.6
                curvature, is_rounded = analyze_back_curvature(shoulder, hip, mid_spine)

                if rep_in_progress:
                    max_delta_x = max(max_delta_x, delta_x)
                    min_knee_angle = min(min_knee_angle, knee_angle)
                    spine_curvature = min(spine_curvature, curvature)
                    spine_flagged = spine_flagged or is_rounded

                if not rep_in_progress:
                    if delta_x > 0.08:
                        rep_in_progress = True
                        max_delta_x = delta_x
                        min_knee_angle = knee_angle
                        spine_curvature = curvature
                        spine_flagged = is_rounded

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

                            if spine_flagged:
                                feedbacks.append("Your spine is rounding inward too much")
                                penalty += 2.5

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
                    min_knee_angle = 999
                    spine_curvature = 0
                    spine_flagged = False

            except Exception:
                continue

    cap.release()
    technique_score = round(np.mean(all_scores) * 2) / 2 if all_scores else 0
    if not overall_feedback:
        overall_feedback.append("Great form! Your back stayed straight throughout the movement. Keep it up! 💪")

    return {
        "technique_score": technique_score,
        "squat_count": counter,
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": overall_feedback,
        "problem_reps": problem_reps,
    }

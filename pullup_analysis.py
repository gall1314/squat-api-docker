import cv2
import numpy as np
import mediapipe as mp
import time

class PullUpAnalyzer:
    def __init__(self, angle_drop_threshold=15.0, min_separation=5, nose_rise_thresh=0.0018):
        self.angle_drop_threshold = angle_drop_threshold
        self.min_separation = min_separation
        self.nose_rise_thresh = nose_rise_thresh

    def analyze_all_reps(self, frames):
        rep_ranges = self.segment_reps(frames)
        rep_reports = []
        all_errors = []
        all_tips = []

        for start, end in rep_ranges:
            rep_frames = frames[start:end]
            result = self.analyze_rep(rep_frames)
            rep_reports.append(result)
            all_errors.extend(result["errors"])
            all_tips.extend(result["tips"])

        if rep_reports:
            raw_scores = [r["technique_score"] for r in rep_reports]
            avg_score = np.mean(raw_scores)
            rounded_score = np.clip(round(avg_score * 2) / 2, 4.0, 10.0)
        else:
            rounded_score = 0.0

        feedback = list(set(all_errors)) if all_errors else (["Great form! Keep it up"] if rep_reports else ["No pull-ups detected"])
        tips = list(set(all_tips))

        return {
            "rep_count": len(rep_reports),
            "squat_count": len(rep_reports),
            "technique_score": rounded_score,
            "good_reps": sum(1 for r in rep_reports if r["technique_score"] >= 8),
            "bad_reps": sum(1 for r in rep_reports if r["technique_score"] < 8),
            "feedback": feedback,
            "tips": tips,
            "reps": rep_reports
        }

    def analyze_rep(self, frames):
        errors = []
        tips = []

        elbow_angles = []
        knee_angles = []
        nose_ys = []
        wrist_ys = []

        for f in frames:
            l = self.elbow_angle(f, "LEFT")
            r = self.elbow_angle(f, "RIGHT")
            valid_elbows = [a for a in [l, r] if a is not None]
            if valid_elbows:
                elbow_angles.append(np.mean(valid_elbows))

            kl = self.knee_angle(f, "LEFT")
            kr = self.knee_angle(f, "RIGHT")
            valid_knees = [a for a in [kl, kr] if a is not None]
            if valid_knees:
                knee_angles.append(np.mean(valid_knees))

            if "NOSE" in f:
                nose_ys.append(f["NOSE"][1])
            if all(k in f for k in ["LEFT_WRIST", "RIGHT_WRIST"]):
                wrist_y = (f["LEFT_WRIST"][1] + f["RIGHT_WRIST"][1]) / 2
                wrist_ys.append(wrist_y)

        if not any(n < w - 0.0018 for n, w in zip(nose_ys, wrist_ys)):
            errors.append("Try to pull a bit higher – chin past the bar")

        if max(elbow_angles, default=0) < 155:
            errors.append("Start each rep from straight arms for full range")

        if knee_angles and (max(knee_angles) - min(knee_angles)) > 40:
            errors.append("Keep legs steadier for more control")

        if len(elbow_angles) >= 6:
            descent_len = self.detect_descent_duration(elbow_angles)
            if descent_len < 2:
                tips.append("Control the lowering phase for better muscle growth")

        score_map = {0: 10, 1: 8, 2: 6, 3: 5}
        technique_score = score_map.get(len(errors), 4)

        return {
            "technique_score": technique_score,
            "errors": errors,
            "tips": tips
        }

    def detect_descent_duration(self, angles):
        min_idx = np.argmin(angles)
        for i in range(min_idx + 1, len(angles)):
            if angles[i] > angles[min_idx] + 5:
                return i - min_idx
        return len(angles) - min_idx

    def elbow_angle(self, f, side):
        keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
        if all(k in f for k in keys):
            a, b, c = f[keys[0]], f[keys[1]], f[keys[2]]
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        return None

    def knee_angle(self, f, side):
        keys = [f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE"]
        if all(k in f for k in keys):
            a, b, c = f[keys[0]], f[keys[1]], f[keys[2]]
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        return None

    def segment_reps(self, frames):
        elbow_angles = []
        noses_y = []

        for f in frames:
            l = self.elbow_angle(f, "LEFT")
            r = self.elbow_angle(f, "RIGHT")
            valid = [a for a in [l, r] if a is not None]
            avg = np.mean(valid) if valid else None
            elbow_angles.append(avg)
            noses_y.append(f["NOSE"][1] if "NOSE" in f else None)

        reps = []
        last_rep_index = -self.min_separation
        state = "waiting"
        start_idx = None
        start_angle = None

        for i in range(1, len(elbow_angles)):
            curr_angle = elbow_angles[i]
            prev_angle = elbow_angles[i - 1]
            curr_nose = noses_y[i]
            prev_nose = noses_y[i - 1]

            if None in (curr_angle, prev_angle, curr_nose, prev_nose):
                continue

            angle_drop = prev_angle - curr_angle
            nose_rising = curr_nose < prev_nose - self.nose_rise_thresh

            if (
                state == "waiting"
                and angle_drop >= self.angle_drop_threshold
                and nose_rising
                and (i - last_rep_index) >= self.min_separation
            ):
                state = "in_rep"
                start_idx = i - 1
                start_angle = curr_angle

            elif state == "in_rep" and curr_angle > start_angle:
                reps.append((start_idx, i))
                last_rep_index = i
                state = "waiting"

        return reps


def run_pullup_analysis(video_path, frame_skip=4, scale=0.25, verbose=True):
    cv2.setNumThreads(1)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    frame_count = 0

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        print(f"Reading frame {frame_count}...")
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            frame_landmarks = {}
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[mp_pose.PoseLandmark(i).name] = (lm.x, lm.y, lm.z)
            landmarks_list.append(frame_landmarks)

        frame_count += 1
        if verbose and frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    pose.close()
    print(f"Total frames processed: {frame_count} in {round(time.time() - start_time, 2)} seconds")

    analyzer = PullUpAnalyzer()
    return analyzer.analyze_all_reps(landmarks_list)

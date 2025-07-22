import cv2
import numpy as np
import mediapipe as mp

class PullUpAnalyzer:
    def __init__(self):
        pass

    def analyze_all_reps(self, frames):
        rep_ranges = self.segment_reps(frames)
        rep_reports = []
        good_reps, bad_reps = 0, 0
        all_feedback = []

        for start, end in rep_ranges:
            rep_frames = frames[start:end]
            result = self.analyze_rep(rep_frames)
            rep_reports.append(result)

            if result["technique_score"] >= 0.8:
                good_reps += 1
            else:
                bad_reps += 1
                all_feedback.extend(result["errors"])

        if rep_reports:
            technique_score = round(np.mean([r["technique_score"] for r in rep_reports]), 2)
        else:
            technique_score = 0.0

        return {
            "squat_count": len(rep_reports),
            "technique_score": technique_score,
            "good_reps": good_reps,
            "bad_reps": bad_reps,
            "feedback": list(set(all_feedback)),
            "reps": rep_reports
        }

    def analyze_rep(self, rep_frames):
        errors = []

        if not self.full_range(rep_frames):
            errors.append("טווח תנועה חלקי")
        if not self.full_extension(rep_frames):
            errors.append("לא יישר ידיים בסוף")
        if self.has_excessive_momentum(rep_frames):
            errors.append("שימוש מופרז במומנטום")

        technique_score = 1 - 0.2 * len(errors)
        technique_score = max(0.0, technique_score)

        return {
            "technique_score": round(technique_score, 2),
            "errors": errors
        }

    def full_range(self, frames):
        nose_ys = [f["NOSE"][1] for f in frames if "NOSE" in f]
        shoulder_ys = [(f["LEFT_SHOULDER"][1] + f["RIGHT_SHOULDER"][1]) / 2 for f in frames if "LEFT_SHOULDER" in f and "RIGHT_SHOULDER" in f]
        if not nose_ys or not shoulder_ys:
            return False

        min_nose_y = min(nose_ys)
        max_nose_y = max(nose_ys)
        delta = max_nose_y - min_nose_y
        return delta > 0.15

    def full_extension(self, frames):
        def angle(a, b, c):
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        for f in frames[-5:]:
            if all(k in f for k in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]):
                ang = angle(f["LEFT_SHOULDER"], f["LEFT_ELBOW"], f["LEFT_WRIST"])
                if ang > 165:
                    return True
            if all(k in f for k in ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]):
                ang = angle(f["RIGHT_SHOULDER"], f["RIGHT_ELBOW"], f["RIGHT_WRIST"])
                if ang > 165:
                    return True
        return False

    def has_excessive_momentum(self, frames):
        all_z = []
        for f in frames:
            if "LEFT_ANKLE" in f:
                all_z.append(f["LEFT_ANKLE"][2])
            if "RIGHT_ANKLE" in f:
                all_z.append(f["RIGHT_ANKLE"][2])
        if not all_z:
            return False

        z_range = max(all_z) - min(all_z)
        return z_range > 0.15

    def segment_reps(self, frames, min_frames_between=8):
        nose_ys = [f["NOSE"][1] if "NOSE" in f else None for f in frames]
        deltas = []
        for i in range(1, len(nose_ys)):
            if nose_ys[i] is not None and nose_ys[i-1] is not None:
                deltas.append(nose_ys[i] - nose_ys[i-1])
            else:
                deltas.append(0)

        state = "down"
        rep_start = None
        reps = []

        for i in range(1, len(deltas)):
            if state == "down" and deltas[i] < -0.01:
                rep_start = i
                state = "up"
            elif state == "up" and deltas[i] > 0.01:
                if rep_start and i - rep_start > min_frames_between:
                    reps.append((max(0, rep_start - 2), min(len(frames), i + 2)))
                state = "down"

        return reps

# ✅ זוהי הפונקציה ש־api.py קורא לה
def run_pullup_analysis(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        frame_landmarks = {}
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[mp_pose.PoseLandmark(i).name] = (lm.x, lm.y, lm.z)

        landmarks_list.append(frame_landmarks)
        frame_count += 1

    cap.release()
    pose.close()

    analyzer = PullUpAnalyzer()
    return analyzer.analyze_all_reps(landmarks_list)


import cv2
import numpy as np
import mediapipe as mp

# ------------------------------------------------------
# Keypoints extraction (NO dependency on video_utils)
# ------------------------------------------------------
def extract_keypoints(video_path, frame_skip=3, scale=0.4):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    cap = cv2.VideoCapture(video_path)

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip != 0:
            idx += 1
            continue

        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        frame_landmarks = {}
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[mp_pose.PoseLandmark(i).name] = (lm.x, lm.y, lm.z)
        frames.append(frame_landmarks)
        idx += 1

    cap.release()
    pose.close()
    return frames


# ------------------------------------------------------
# Analyzer
# ------------------------------------------------------
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

            if result["technique_score"] >= 8:
                good_reps += 1
            else:
                bad_reps += 1
                all_feedback.extend(result["errors"])

        if rep_reports:
            avg_score = np.mean([r["technique_score"] for r in rep_reports])
            # עיגול ל- .0 או .5
            technique_score = round(round(avg_score * 2) / 2, 2)
        else:
            technique_score = 0.0

        if technique_score == 10 and not all_feedback:
            all_feedback = ["Great form! Keep it up 💪"]

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
            errors.append("Try to pull yourself a bit higher – aim for chin-to-wrist height")
        if not self.full_extension(rep_frames):
            errors.append("Make sure to fully extend your arms at the bottom")
        if self.has_excessive_momentum(rep_frames):
            errors.append("Try to minimize leg movement – avoid kicking")

        deductions = len(errors)
        technique_score = 10 - (2 * deductions if deductions < 3 else 6)
        technique_score = max(4, technique_score)

        return {
            "technique_score": round(technique_score, 2),
            "errors": errors
        }

    # ---------- technique checks ----------

    def full_range(self, frames):
        # chin (nose proxy) above wrist height at peak
        for f in frames:
            if all(k in f for k in ["NOSE", "LEFT_WRIST", "RIGHT_WRIST"]):
                nose_y = f["NOSE"][1]
                avg_wrist_y = (f["LEFT_WRIST"][1] + f["RIGHT_WRIST"][1]) / 2
                if nose_y < avg_wrist_y:
                    return True
        return False

    def full_extension(self, frames):
        def angle(a, b, c):
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        angles = []
        # טיפה סלחני אבל לא מדי – 10 פריימים אחרונים, לפחות 3 זוויות > 170
        for f in frames[-10:]:
            for side in ["LEFT", "RIGHT"]:
                keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
                if all(k in f for k in keys):
                    ang = angle(f[keys[0]], f[keys[1]], f[keys[2]])
                    if not np.isnan(ang):
                        angles.append(ang)

        return sum(a > 170 for a in angles) >= 3

    def has_excessive_momentum(self, frames):
        # שינוי גדול בזווית הברך לאורך החזרה -> קיק
        def angle(a, b, c):
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        angles = []
        for f in frames:
            for side in ["LEFT", "RIGHT"]:
                keys = [f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE"]
                if all(k in f for k in keys):
                    ang = angle(f[keys[0]], f[keys[1]], f[keys[2]])
                    if not np.isnan(ang):
                        angles.append(ang)

        if len(angles) < 3:
            return False

        diff = max(angles) - min(angles)
        return diff > 35  # אפשר לכוונן אחרי בדיקות נוספות

    # ---------- rep segmentation ----------

    def segment_reps(self, frames, min_frames_between=6):
        reps = []
        start = None
        in_rep = False

        for i in range(1, len(frames)):
            if "NOSE" not in frames[i] or "NOSE" not in frames[i - 1]:
                continue

            # תנועה למעלה (y קטן) + מרפקים מכופפים => תחילת חזרה
            delta_y = frames[i - 1]["NOSE"][1] - frames[i]["NOSE"][1]

            elbow_angles = []
            for side in ["LEFT", "RIGHT"]:
                keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
                if all(k in frames[i] for k in keys):
                    a, b, c = frames[i][keys[0]], frames[i][keys[1]], frames[i][keys[2]]
                    ba = np.array(a) - np.array(b)
                    bc = np.array(c) - np.array(b)
                    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle_val = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                    elbow_angles.append(angle_val)

            if delta_y > 0.0025 and any(a < 110 for a in elbow_angles):
                if not in_rep:
                    start = i
                    in_rep = True
            elif in_rep and delta_y < 0:
                if i - start >= min_frames_between:
                    reps.append((start, i))
                in_rep = False

        return reps


# ------------------------------------------------------
# public entry point for app.py
# ------------------------------------------------------
def run_pullup_analysis(video_path, frame_skip=3, scale=0.4):
    frames = extract_keypoints(video_path, frame_skip=frame_skip, scale=scale)
    analyzer = PullUpAnalyzer()
    return analyzer.analyze_all_reps(frames)


import cv2
import numpy as np
import mediapipe as mp

class PullUpAnalyzer:
    def __init__(self, flex_threshold=100, min_separation=5):
        self.flex_threshold = flex_threshold  # זווית מתחתיה נחשב ככיפוף
        self.min_separation = min_separation  # רווח מינימלי בין חזרות

    def analyze_all_reps(self, frames):
        rep_ranges = self.segment_reps(frames)
        return {
            "rep_count": len(rep_ranges),
            "squat_count": len(rep_ranges),  # תאימות לשם ישן אם נדרש
            "technique_score": 10.0 if len(rep_ranges) > 0 else 0.0,
            "good_reps": len(rep_ranges),
            "bad_reps": 0,
            "feedback": [] if len(rep_ranges) > 0 else ["No pull-ups detected"],
            "reps": [{"technique_score": 10.0, "errors": []} for _ in rep_ranges]
        }

    def segment_reps(self, frames):
        def elbow_angle(f, side="LEFT"):
            keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
            if all(k in f for k in keys):
                a, b, c = f[keys[0]], f[keys[1]], f[keys[2]]
                ba = np.array(a) - np.array(b)
                bc = np.array(c) - np.array(b)
                cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            return None

        elbow_angles = []
        for f in frames:
            l = elbow_angle(f, "LEFT")
            r = elbow_angle(f, "RIGHT")
            valid = [a for a in [l, r] if a is not None]
            avg = np.mean(valid) if valid else None
            elbow_angles.append(avg)

        reps = []
        last_rep_frame = -self.min_separation
        in_rep = False

        for i in range(1, len(elbow_angles)):
            angle = elbow_angles[i]
            if angle is None:
                continue

            if not in_rep and angle < self.flex_threshold and (i - last_rep_frame) > self.min_separation:
                # התחלה של כיפוף
                start = max(0, i - 2)
                in_rep = True
            elif in_rep and angle > self.flex_threshold + 10:
                # סיום חזרה
                end = min(len(frames), i + 2)
                reps.append((start, end))
                last_rep_frame = i
                in_rep = False

        return reps

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


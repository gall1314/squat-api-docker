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

            if result["technique_score"] >= 8:
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
            errors.append("Did not pull high enough (chin should reach wrist height)")
        if not self.full_extension(rep_frames):
            errors.append("Did not fully extend arms at the bottom")
        if self.has_excessive_momentum(rep_frames):
            errors.append("Used excessive leg momentum (kipping)")

        score_map = {
            0: 10,
            1: 8,
            2: 6,
            3: 5
        }
        technique_score = score_map.get(len(errors), 4)

        return {
            "technique_score": technique_score,
            "errors": errors
        }

    def full_range(self, frames):
        """Check if nose reaches wrist height"""
        for f in frames:
            if all(k in f for k in ["NOSE", "LEFT_WRIST", "RIGHT_WRIST"]):
                nose_y = f["NOSE"][1]
                avg_wrist_y = (f["LEFT_WRIST"][1] + f["RIGHT_WRIST"][1]) / 2
                if nose_y < avg_wrist_y:  # chin above wrist level
                    return True
        return False

    def full_extension(self, frames):
        """Check if elbow gets near full extension (>170°) at bottom"""
        def angle(a, b, c):
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        max_angles = []
        for f in frames:
            if all(k in f for k in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]):
                ang = angle(f["LEFT_SHOULDER"], f["LEFT_ELBOW"], f["LEFT_WRIST"])
                max_angles.append(ang)
            if all(k in f for k in ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]):
                ang = angle(f["RIGHT_SHOULDER"], f["RIGHT_ELBOW"], f["RIGHT_WRIST"])
                max_angles.append(ang)

        return any(a > 170 for a in max_angles)

    def has_excessive_momentum(self, frames):
        """Detect fast knee bend & extend = kipping"""
        def knee_angle(f, side="LEFT"):
            keys = [f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE"]
            if all(k in f for k in keys):
                a = f[keys[0]]
                b = f[keys[1]]
                c = f[keys[2]]
                ba = np.array(a) - np.array(b)
                bc = np.array(c) - np.array(b)
                cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            return None

        knee_angles = []
        for f in frames:
            left = knee_angle(f, "LEFT")
            right = knee_angle(f, "RIGHT")
            if left:
                knee_angles.append(left)
            elif right:
                knee_angles.append(right)

        if len(knee_angles) < 3:
            return False

        diffs = [abs(knee_angles[i+1] - knee_angles[i]) for i in range(len(knee_angles)-1)]
        peak_movement = max(diffs) if diffs else 0
        return peak_movement > 30  # large swing in knee angle = kick

    def segment_reps(self, frames, min_gap=8):
        """Detect reps by combining elbow angle and head rise"""
        def elbow_angle(f, side="LEFT"):
            keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
            if all(k in f for k in keys):
                a, b, c = f[keys[0]], f[keys[1]], f[keys[2]]
                ba = np.array(a) - np.array(b)
                bc = np.array(c) - np.array(b)
                cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            return None

        angles = []
        noses = []
        for f in frames:
            ang_l = elbow_angle(f, "LEFT")
            ang_r = elbow_angle(f, "RIGHT")
            ang = np.mean([a for a in [ang_l, ang_r] if a is not None]) if ang_l or ang_r else None
            angles.append(ang)
            noses.append(f["NOSE"][1] if "NOSE" in f else None)

        reps = []
        in_rep = False
        start_idx = 0

        for i in range(1, len(frames)):
            if angles[i] is None or noses[i] is None:
                continue

            elbow_flexed = angles[i] < 100
            nose_moved_up = noses[i] < noses[i-1] - 0.01

            if not in_rep and elbow_flexed and nose_moved_up:
                in_rep = True
                start_idx = i
            elif in_rep and angles[i] > 165:
                if i - start_idx > min_gap:
                    reps.append((max(0, start_idx-2), min(len(frames), i+2)))
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


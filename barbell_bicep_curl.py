import cv2
import numpy as np
import mediapipe as mp

class BarbellBicepCurlAnalyzer:
    def __init__(
        self,
        angle_drop_threshold=20.0,
        min_separation=6,
        min_bottom_extension_angle=155.0,
        max_top_flexion_angle=60.0,
        max_elbow_xy_drift=0.03,
        eccentric_slow_min_frames=3
    ):
        self.angle_drop_threshold = angle_drop_threshold
        self.min_separation = min_separation
        self.min_bottom_extension_angle = min_bottom_extension_angle
        self.max_top_flexion_angle = max_top_flexion_angle
        self.max_elbow_xy_drift = max_elbow_xy_drift
        self.eccentric_slow_min_frames = eccentric_slow_min_frames

    def analyze_all_reps(self, frames):
        rep_ranges = self.segment_reps(frames)
        rep_reports = []
        all_errors, all_tips = [], []

        for start, end in rep_ranges:
            rep_frames = frames[start:end]
            result = self.analyze_rep(rep_frames)
            rep_reports.append(result)
            all_errors.extend(result["errors"])
            all_tips.extend(result["tips"])

        if rep_reports:
            avg_score = float(np.mean([r["technique_score"] for r in rep_reports]))
            final_score = np.clip(round(avg_score * 2) / 2, 4.0, 10.0)
        else:
            final_score = 0.0

        feedback = list(set(all_errors)) if all_errors else (["Great form! Keep it up"] if rep_reports else ["No curls detected"])
        tips = list(set(all_tips))

        return {
            "rep_count": len(rep_reports),
            "squat_count": len(rep_reports),
            "technique_score": final_score,
            "good_reps": sum(1 for r in rep_reports if r["technique_score"] >= 8),
            "bad_reps": sum(1 for r in rep_reports if r["technique_score"] < 8),
            "feedback": feedback,
            "tips": tips,
            "reps": rep_reports
        }

    def analyze_rep(self, frames):
        errors, tips = [], []
        elbow_angles = []
        elbows_x, elbows_y = [], []

        for f in frames:
            l_ang = self.elbow_angle(f, "LEFT")
            r_ang = self.elbow_angle(f, "RIGHT")
            valid = [a for a in (l_ang, r_ang) if a is not None]
            if valid:
                elbow_angles.append(np.mean(valid))

            lx, ly = self.get_point_xy(f, "LEFT_ELBOW")
            rx, ry = self.get_point_xy(f, "RIGHT_ELBOW")
            if lx is not None and rx is not None:
                elbows_x.append((lx + rx) / 2)
            if ly is not None and ry is not None:
                elbows_y.append((ly + ry) / 2)

        if max(elbow_angles, default=0) < self.min_bottom_extension_angle:
            errors.append("Straighten your arms fully at the bottom for a full range of motion")

        if min(elbow_angles, default=180) > self.max_top_flexion_angle:
            errors.append("Try to curl higher – aim to squeeze at the top")

        if elbows_x and elbows_y:
            if (max(elbows_x) - min(elbows_x)) > self.max_elbow_xy_drift or (max(elbows_y) - min(elbows_y)) > self.max_elbow_xy_drift:
                errors.append("Keep your elbows fixed – avoid drifting forward/upward")

        if len(elbow_angles) >= 4 and self.detect_eccentric_duration(elbow_angles) < self.eccentric_slow_min_frames:
            tips.append("Slow down the lowering phase to maximize hypertrophy")

        score_map = {0: 10, 1: 8, 2: 6, 3: 5}
        technique_score = score_map.get(len(errors), 4)

        return {
            "technique_score": technique_score,
            "errors": errors,
            "tips": tips
        }

    def segment_reps(self, frames):
        elbow_angles = []
        for f in frames:
            l = self.elbow_angle(f, "LEFT")
            r = self.elbow_angle(f, "RIGHT")
            valid = [a for a in (l, r) if a is not None]
            elbow_angles.append(np.mean(valid) if valid else None)

        reps = []
        last_rep_index = -self.min_separation
        state = "waiting"
        start_idx = None
        start_angle = None

        for i in range(1, len(elbow_angles)):
            curr = elbow_angles[i]
            prev = elbow_angles[i - 1]
            if curr is None or prev is None:
                continue

            if state == "waiting" and (prev - curr) >= self.angle_drop_threshold and (i - last_rep_index) >= self.min_separation:
                state = "in_rep"
                start_idx = i - 1
                start_angle = prev

            elif state == "in_rep" and curr >= start_angle - 5:
                reps.append((start_idx, i))
                last_rep_index = i
                state = "waiting"

        return reps

    def detect_eccentric_duration(self, angles):
        if not angles:
            return 0
        min_idx = int(np.argmin(angles))
        length = 0
        last = angles[min_idx]
        for i in range(min_idx + 1, len(angles)):
            if angles[i] >= last + 2:
                length += 1
                last = angles[i]
            else:
                last = max(last, angles[i])
        return length

    def elbow_angle(self, f, side):
        keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
        if all(k in f for k in keys):
            a, b, c = np.array(f[keys[0]]), np.array(f[keys[1]]), np.array(f[keys[2]])
            ba, bc = a - b, c - b
            denom = np.linalg.norm(ba) * np.linalg.norm(bc)
            if denom == 0:
                return None
            cos = np.dot(ba, bc) / denom
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return None

    def get_point_xy(self, f, key):
        if key in f:
            return f[key][0], f[key][1]
        return None, None

def run_barbell_bicep_curl_analysis(video_path, frame_skip=3, scale=0.4, verbose=True):
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

    analyzer = BarbellBicepCurlAnalyzer()
    return analyzer.analyze_all_reps(landmarks_list)


    cap.release()
    pose.close()

    analyzer = BicepCurlAnalyzer()
    return analyzer.analyze_all_reps(landmarks_list)

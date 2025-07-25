import cv2
import numpy as np
import mediapipe as mp


class PullUpAnalyzer:
    def __init__(
        self,
        angle_drop_threshold: float = 18.0,   # Δ elbow angle to start a rep
        min_separation: int = 5,              # minimal frames between reps
        nose_rise_thresh: float = 0.001,      # how much NOSE.y must drop to confirm "up"
        chin_over_wrist_margin: float = 0.003,# how far nose must go above wrists (y lower)
        full_ext_deg: float = 165.0,          # elbow angle considered "straight enough"
        kipping_delta_deg: float = 40.0       # knee-angle delta to flag kipping
    ):
        self.angle_drop_threshold = angle_drop_threshold
        self.min_separation = min_separation
        self.nose_rise_thresh = nose_rise_thresh
        self.chin_over_wrist_margin = chin_over_wrist_margin
        self.full_ext_deg = full_ext_deg
        self.kipping_delta_deg = kipping_delta_deg

    # ------------- PUBLIC API -------------
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
            raw_scores = [r["technique_score"] for r in rep_reports]
            avg_score = float(np.mean(raw_scores))
            technique_score = float(np.clip(np.round(avg_score * 2) / 2.0, 4.0, 10.0))
        else:
            technique_score = 0.0

        feedback = (
            list(set(all_errors))
            if all_errors
            else (["Great form! Keep it up"] if rep_reports else ["No pull-ups detected"])
        )
        tips = list(set(all_tips))

        return {
            "rep_count": len(rep_reports),
            "squat_count": len(rep_reports),  # backward compat if your FE expects it
            "technique_score": technique_score,
            "good_reps": sum(r["technique_score"] >= 8 for r in rep_reports),
            "bad_reps": sum(r["technique_score"] < 8 for r in rep_reports),
            "feedback": feedback,
            "tips": tips,
            "reps": rep_reports
        }

    # ------------- PER-REP ANALYSIS -------------
    def analyze_rep(self, frames):
        errors, tips = [], []

        elbow_angles = []
        knee_angles = []
        nose_ys = []
        wrist_ys = []

        for f in frames:
            # elbows
            l = self.elbow_angle(f, "LEFT")
            r = self.elbow_angle(f, "RIGHT")
            valid_elbows = [a for a in (l, r) if a is not None]
            if valid_elbows:
                elbow_angles.append(np.mean(valid_elbows))

            # knees (kipping)
            kl = self.knee_angle(f, "LEFT")
            kr = self.knee_angle(f, "RIGHT")
            valid_knees = [a for a in (kl, kr) if a is not None]
            if valid_knees:
                knee_angles.append(np.mean(valid_knees))

            # range up
            if "NOSE" in f:
                nose_ys.append(f["NOSE"][1])
            if all(k in f for k in ("LEFT_WRIST", "RIGHT_WRIST")):
                wrist_y = (f["LEFT_WRIST"][1] + f["RIGHT_WRIST"][1]) / 2
                wrist_ys.append(wrist_y)

        # --- checks ---
        # 1) Full range up (chin past wrists)
        full_range_up = any(
            (nose < wrist - self.chin_over_wrist_margin)
            for nose, wrist in zip(nose_ys, wrist_ys)
        )
        if not full_range_up:
            errors.append("Try to pull a bit higher – chin past the bar")

        # 2) Full extension bottom (anywhere in the rep)
        full_extension = any(a > self.full_ext_deg for a in elbow_angles)
        if not full_extension:
            errors.append("Start each rep from straight arms for full range")

        # 3) Kipping (leg momentum)
        kipping = False
        if len(knee_angles) >= 2:
            if (max(knee_angles) - min(knee_angles)) > self.kipping_delta_deg:
                kipping = True
        if kipping:
            errors.append("Keep legs steadier for more control")

        # Tip: slow descent for hypertrophy
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

    # ------------- HELPERS -------------
    def detect_descent_duration(self, angles):
        """Very rough descent duration heuristic (frames)."""
        if not angles:
            return 0
        min_idx = int(np.argmin(angles))  # top (most flexed)
        for i in range(min_idx + 1, len(angles)):
            # enough opening to say "descent ended"
            if angles[i] > angles[min_idx] + 5:
                return i - min_idx
        return len(angles) - min_idx

    def elbow_angle(self, f, side: str):
        keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
        if all(k in f for k in keys):
            a, b, c = f[keys[0]], f[keys[1]], f[keys[2]]
            return self._three_point_angle(a, b, c)
        return None

    def knee_angle(self, f, side: str):
        keys = [f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE"]
        if all(k in f for k in keys):
            a, b, c = f[keys[0]], f[keys[1]], f[keys[2]]
            return self._three_point_angle(a, b, c)
        return None

    @staticmethod
    def _three_point_angle(a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
        if denom == 0:
            return None
        cos = np.dot(ba, bc) / denom
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    # ------------- REP SEGMENTATION -------------
    def segment_reps(self, frames):
        elbow_angles = []
        noses_y = []

        for f in frames:
            l = self.elbow_angle(f, "LEFT")
            r = self.elbow_angle(f, "RIGHT")
            valid = [a for a in (l, r) if a is not None]
            elbow_angles.append(np.mean(valid) if valid else None)
            noses_y.append(f["NOSE"][1] if "NOSE" in f else None)

        reps = []
        last_rep_index = -self.min_separation
        state = "waiting"
        start_idx, start_angle = None, None

        for i in range(1, len(elbow_angles)):
            curr_angle, prev_angle = elbow_angles[i], elbow_angles[i - 1]
            curr_nose, prev_nose = noses_y[i], noses_y[i - 1]
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


def run_pullup_analysis(
    video_path: str,
    frame_skip: int = 4,   # faster (tune to 2–3 if you miss reps)
    scale: float = 0.25,   # 0.2–0.3 is usually fine
    verbose: bool = False
):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,               # fastest
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    cap = cv2.VideoCapture(video_path)

    # collect minimal data (fewer dict ops -> a bit faster, but still readable)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                fl = {}
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    name = mp_pose.PoseLandmark(i).name
                    fl[name] = (lm.x, lm.y, lm.z)
                frames.append(fl)

            if verbose and frame_count % (30 * frame_skip) == 0:
                print(f"Processed ~{frame_count} raw frames")

        frame_count += 1

    cap.release()
    pose.close()

    analyzer = PullUpAnalyzer()
    return analyzer.analyze_all_reps(frames)

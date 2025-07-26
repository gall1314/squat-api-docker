import cv2
import numpy as np
import mediapipe as mp
import time

class PullUpAnalyzer:
    def __init__(
        self,
        angle_drop_threshold: float = 18.0,
        min_separation: int = 5,
        nose_rise_thresh: float = 0.0012,
        chin_over_wrist_margin: float = 0.0012,
        full_ext_deg: float = 145.0,
        kipping_delta_deg: float = 40.0,
    ):
        self.angle_drop_threshold = angle_drop_threshold
        self.min_separation = min_separation
        self.nose_rise_thresh = nose_rise_thresh
        self.chin_over_wrist_margin = chin_over_wrist_margin
        self.full_ext_deg = full_ext_deg
        self.kipping_delta_deg = kipping_delta_deg

    # ---------- PUBLIC ----------
    def analyze_all_reps(self, frames):
        rep_ranges = self.segment_reps(frames)
        rep_reports, all_errors, all_tips = [], [], []

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
            "squat_count": len(rep_reports),  # תאימות לאחור
            "technique_score": technique_score,
            "good_reps": sum(r["technique_score"] >= 8 for r in rep_reports),
            "bad_reps": sum(r["technique_score"] < 8 for r in rep_reports),
            "feedback": feedback,
            "tips": tips,
            "reps": rep_reports,
        }

    # ---------- PER-REP ----------
    def analyze_rep(self, frames):
        errors, tips = [], []

        elbow_angles, knee_angles = [], []
        nose_ys, wrist_ys = [], []

        for f in frames:
            # Elbows
            l = self.elbow_angle(f, "LEFT")
            r = self.elbow_angle(f, "RIGHT")
            valid_elbows = [a for a in (l, r) if a is not None]
            if valid_elbows:
                elbow_angles.append(float(np.mean(valid_elbows)))

            # Knees (kipping)
            kl = self.knee_angle(f, "LEFT")
            kr = self.knee_angle(f, "RIGHT")
            valid_knees = [a for a in (kl, kr) if a is not None]
            if valid_knees:
                knee_angles.append(float(np.mean(valid_knees)))

            # Range up (chin vs wrists)
            if "NOSE" in f:
                nose_ys.append(f["NOSE"][1])
            if all(k in f for k in ("LEFT_WRIST", "RIGHT_WRIST")):
                wrist_y = (f["LEFT_WRIST"][1] + f["RIGHT_WRIST"][1]) / 2
                wrist_ys.append(wrist_y)

        # 1) Full range up – chin past wrists (relaxed)
        if wrist_ys and nose_ys:
            chin_over_bar = any(n < w - self.chin_over_wrist_margin for n, w in zip(nose_ys, wrist_ys))
        else:
            chin_over_bar = True  # אל תעניש אם אין דאטה ברור
        if not chin_over_bar:
            errors.append("Try to pull a bit higher – chin past the bar")

        # 2) Full extension bottom – max elbow angle across the whole rep
        full_extension = bool(elbow_angles) and max(elbow_angles) >= self.full_ext_deg
        if not full_extension:
            errors.append("Start each rep from straight arms for full range")

        # 3) Kipping
        kipping = False
        if len(knee_angles) >= 2:
            if (max(knee_angles) - min(knee_angles)) > self.kipping_delta_deg:
                kipping = True
        if kipping:
            errors.append("Keep legs steadier for more control")

        # Tip: eccentric control
        if len(elbow_angles) >= 6:
            descent_len = self.detect_descent_duration(elbow_angles)
            if descent_len < 2:
                tips.append("Control the lowering phase for better muscle growth")

        score_map = {0: 10, 1: 8, 2: 6, 3: 5}
        technique_score = score_map.get(len(errors), 4)

        return {
            "technique_score": technique_score,
            "errors": errors,
            "tips": tips,
        }

    # ---------- HELPERS ----------
    def detect_descent_duration(self, angles):
        """Heuristic descent duration measured in frames."""
        if not angles:
            return 0
        min_idx = int(np.argmin(angles))  # top (most flexed)
        for i in range(min_idx + 1, len(angles)):
            if angles[i] > angles[min_idx] + 5:
                return i - min_idx
        return len(angles) - min_idx

    def elbow_angle(self, f, side):
        keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
        if all(k in f for k in keys):
            a, b, c = f[keys[0]], f[keys[1]], f[keys[2]]
            return self._three_point_angle(a, b, c)
        return None

    def knee_angle(self, f, side):
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

    # ---------- REP SEGMENTATION ----------
    def segment_reps(self, frames):
        elbow_angles, noses_y = [], []

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
    frame_skip: int = 4,
    scale: float = 0.25,
    verbose: bool = True
):
    # speed ups
    cv2.setNumThreads(1)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,              # FAST
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    frame_count = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            if verbose and frame_count % (30 * frame_skip) == 0:
                print(f"Reading raw frame #{frame_count}...")
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                fl = {}
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    fl[mp_pose.PoseLandmark(i).name] = (lm.x, lm.y, lm.z)
                frames.append(fl)

        frame_count += 1

    cap.release()
    pose.close()

    if verbose:
        print(f"Processed {frame_count} raw frames in {time.time() - t0:.2f}s")
        print(f"Frames with landmarks kept: {len(frames)}")

    analyzer = PullUpAnalyzer()
    return analyzer.analyze_all_reps(frames)


# Example usage:
# result = run_pullup_analysis("pullups.mp4", frame_skip=4, scale=0.25, verbose=True)
# print(result)


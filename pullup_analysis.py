import numpy as np

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
            errors.append("Try to fully extend your arms at the bottom for better form")
        if self.has_excessive_momentum(rep_frames):
            errors.append("Try to avoid using leg momentum – keep your form strict")

        deductions = min(len(errors), 3)
        technique_score = 10 - (2 * deductions if deductions < 3 else 6)
        technique_score = max(4, technique_score)

        return {
            "technique_score": round(technique_score, 2),
            "errors": errors
        }

    def full_range(self, frames):
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
        for f in frames[-12:]:  # טווח רחב יותר של מסגרות לסוף החזרה
            for side in ["LEFT", "RIGHT"]:
                keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
                if all(k in f for k in keys):
                    angles.append(angle(f[keys[0]], f[keys[1]], f[keys[2]]))

        angles = [a for a in angles if a is not None]
        return sum(a > 160 for a in angles) >= 3  # הורדנו מ-165 ל-160 כדי לא לפסול חינם

    def has_excessive_momentum(self, frames):
        y_values = [f["LEFT_ANKLE"][1] for f in frames if "LEFT_ANKLE" in f]
        if len(y_values) < 5:
            return False
        diffs = [abs(y_values[i+1] - y_values[i]) for i in range(len(y_values)-1)]
        avg_change = np.mean(diffs)
        return avg_change > 0.02  # אפשר לכוונן עוד

    def segment_reps(self, frames, min_frames_between=5):
        reps = []
        start = None
        in_rep = False

        for i in range(1, len(frames)):
            if "NOSE" not in frames[i] or "NOSE" not in frames[i - 1]:
                continue
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

            # שיפור רגישות התחלת חזרה
            if delta_y > 0.002 and any(a < 115 for a in elbow_angles):
                if not in_rep:
                    start = i
                    in_rep = True
            elif in_rep and delta_y < 0:
                if i - start >= min_frames_between:
                    reps.append((start, i))
                in_rep = False

        return reps


def run_pullup_analysis(video_path, frame_skip=3, scale=0.4):
    from video_utils import extract_keypoints  # שמירה על המבנה הקיים
    frames = extract_keypoints(video_path, frame_skip=frame_skip, scale=scale)
    analyzer = PullUpAnalyzer()
    return analyzer.analyze_all_reps(frames)


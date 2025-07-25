import cv2
import numpy as np
import mediapipe as mp

def round_to_half(x, min_score=4.0, max_score=10.0):
    """Round to nearest 0.5 and clamp to [min_score, max_score]."""
    return float(np.clip(np.round(x * 2) / 2.0, min_score, max_score))

class PullUpAnalyzer:
    def __init__(self,
                 elbow_flex_threshold=110,     # מרפק כפוף אם הזווית מתחת לערך הזה
                 angle_drop_deg=2.0,           # ירידה מינימלית בזווית בין פריימים כדי לזהות תחילת עלייה
                 nose_up_delta=0.002,          # כמה הראש צריך "לעלות" (y לרדת) כדי להיחשב עלייה
                 min_separation=4,             # מרחק מינימלי בין חזרות
                 min_up_frames=3):             # כמה פריימים רצופים של עלייה כדי לאשר חזרה
        self.elbow_flex_threshold = elbow_flex_threshold
        self.angle_drop_deg = angle_drop_deg
        self.nose_up_delta = nose_up_delta
        self.min_separation = min_separation
        self.min_up_frames = min_up_frames

    def analyze_all_reps(self, frames):
        rep_ranges = self.segment_reps(frames)
        rep_reports = []
        good_reps, bad_reps = 0, 0
        all_feedback = []

        for start, end in rep_ranges:
            rep_frames = frames[start:end]
            result = self.analyze_rep(rep_frames)
            # מעגל גם את ציון החזרה (לא חובה, אבל בד"כ נוח שיהיה עקבי)
            result["technique_score"] = round_to_half(result["technique_score"])
            rep_reports.append(result)

            if result["technique_score"] >= 8:
                good_reps += 1
            else:
                bad_reps += 1
                all_feedback.extend(result["errors"])

        if rep_reports:
            raw_mean = np.mean([r["technique_score"] for r in rep_reports])
            technique_score = round_to_half(raw_mean)
        else:
            technique_score = 0.0

        return {
            "rep_count": len(rep_reports),
            "squat_count": len(rep_reports),   # תאימות אחורה אם אתה כבר משתמש בזה
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
        for f in frames:
            if all(k in f for k in ["NOSE", "LEFT_WRIST", "RIGHT_WRIST"]):
                nose_y = f["NOSE"][1]
                avg_wrist_y = (f["LEFT_WRIST"][1] + f["RIGHT_WRIST"][1]) / 2
                # y קטן יותר = למעלה
                if nose_y < avg_wrist_y:
                    return True
        return False

    def full_extension(self, frames):
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
            if left is not None:
                knee_angles.append(left)
            elif right is not None:
                knee_angles.append(right)

        if len(knee_angles) < 3:
            return False

        diffs = [abs(knee_angles[i+1] - knee_angles[i]) for i in range(len(knee_angles)-1)]
        peak_movement = max(diffs) if diffs else 0
        return peak_movement > 30

    # --------- לוגיקה חדשה: סופרים רק את העלייה ---------
    def segment_reps(self, frames):
        """
        מזהה חזרות ע"י תחילת עלייה:
        - זווית מרפק יורדת (מתחת ל-elbow_flex_threshold וירידה של angle_drop_deg לפחות)
        - הראש (NOSE.y) עולה = y קטן יותר לפחות nose_up_delta
        סופרים רק את העלייה (כלומר טווח הפריימים שלב העלייה).
        """
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
        noses = []

        for f in frames:
            l = elbow_angle(f, "LEFT")
            r = elbow_angle(f, "RIGHT")
            valid = [a for a in [l, r] if a is not None]
            avg = np.mean(valid) if valid else None
            elbow_angles.append(avg)
            noses.append(f["NOSE"][1] if "NOSE" in f else None)

        reps = []
        last_rep_end = -self.min_separation
        state = "idle"
        start_idx = None
        up_frames = 0

        for i in range(1, len(frames)):
            if i - last_rep_end < self.min_separation:
                continue

            angle_prev = elbow_angles[i - 1]
            angle_curr = elbow_angles[i]
            nose_prev = noses[i - 1]
            nose_curr = noses[i]

            if angle_prev is None or angle_curr is None or nose_prev is None or nose_curr is None:
                # אם חסרים נתונים, סגור חזרה אם היינו במצב עלייה
                if state == "up":
                    reps.append((start_idx, i))
                    last_rep_end = i
                    state = "idle"
                    up_frames = 0
                continue

            angle_drop = (angle_prev - angle_curr) >= self.angle_drop_deg
            elbow_flexed = angle_curr < self.elbow_flex_threshold
            nose_up = (nose_curr < nose_prev - self.nose_up_delta)

            if state == "idle":
                # מזהה התחלה של עלייה
                if angle_drop and elbow_flexed and nose_up:
                    state = "up"
                    start_idx = i - 1
                    up_frames = 1
            else:  # state == "up"
                if nose_up and angle_curr <= angle_prev + 1.0:  # מאפשר קצת רעש
                    up_frames += 1
                else:
                    # העלייה נעצרה – אם היו מספיק פריימים של עלייה, נחשב כחזרה
                    if up_frames >= self.min_up_frames:
                        reps.append((start_idx, i))
                        last_rep_end = i
                    state = "idle"
                    up_frames = 0

        # אם נגמר הווידאו בזמן עלייה
        if state == "up" and up_frames >= self.min_up_frames:
            reps.append((start_idx, len(frames) - 1))

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


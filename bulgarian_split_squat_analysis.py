import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict

# =========================
# Config (שנה אם צריך)
# =========================
ANGLE_DOWN_THRESH = 95      # מתחת לזה אנחנו ב"ירידה"
ANGLE_UP_THRESH   = 160     # מעל זה "עלייה" וסיום חזרה
GOOD_REP_MIN_SCORE = 8.0

DEPTH_MAX_KNEE_ANGLE = 110  # אם מעל – "Go deeper"
TORSO_LEAN_MIN       = 145  # אם מתחת – "Stand taller"
VALGUS_X_TOL         = 0.02 # knee.x לא צריך להיות הרבה יותר פנימה מהקרסול

# =========================
# MediaPipe
# =========================
mp_pose = mp.solutions.pose

# =========================
# Helpers
# =========================
def calculate_angle(a, b, c):
    """חישוב זווית בין שלוש נקודות (2D)"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def lm_xy(landmarks, idx, w, h):
    """החזרה כנקודת (x,y) בפיקסלים"""
    return (landmarks[idx].x * w, landmarks[idx].y * h)

def detect_active_leg_simple(landmarks):
    """הגדרה נאיבית – איזו ברך זזה יותר עמוק (בפועל אפשר לשפר)"""
    left  = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    return 'right' if left < right else 'left'

def valgus_ok_norm(landmarks, side):
    """בדיקת Valgus גסה בציר X הנורמליזי (0-1) של Mediapipe"""
    knee   = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x
    ankle  = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x
    # נניח מצלמה מקדימה: אם הברך נכנסת יותר מדי פנימה (לכיוון קו האמצע)
    return not (knee < ankle - VALGUS_X_TOL)

# =========================
# Rep Counter
# =========================
class BulgarianRepCounter:
    def __init__(self):
        self.count = 0
        self.stage = None

        self.rep_reports = []
        self.rep_index = 1
        self.rep_start_frame = None

        self.good_reps = 0
        self.bad_reps = 0
        self.all_feedback = []

        # מעקב מינ/מקס פר-חזרה
        self._curr_min_knee = None
        self._curr_max_knee = None
        self._curr_min_torso = None
        self._curr_valgus_bad = 0

    def _start_rep(self, frame_no):
        self.rep_start_frame = frame_no
        self._curr_min_knee = 999
        self._curr_max_knee = -999
        self._curr_min_torso = 999
        self._curr_valgus_bad = 0

    def _finish_rep(self, frame_no, score, feedback):
        if score >= GOOD_REP_MIN_SCORE:
            self.good_reps += 1
        else:
            self.bad_reps += 1
            self.all_feedback.extend(feedback)

        rep_result = {
            "rep_index": self.rep_index,
            "score": round(score, 1),
            "feedback": feedback,
            "start_frame": self.rep_start_frame or 0,
            "end_frame": frame_no,
            "min_knee_angle": round(self._curr_min_knee, 2) if self._curr_min_knee is not None else None,
            "max_knee_angle": round(self._curr_max_knee, 2) if self._curr_max_knee is not None else None,
            "torso_min_angle": round(self._curr_min_torso, 2) if self._curr_min_torso is not None else None
        }
        self.rep_reports.append(rep_result)
        self.rep_index += 1
        self.rep_start_frame = None

    def evaluate_form(self, min_knee_angle, min_torso_angle, valgus_bad_frames):
        feedback = []
        score = 10.0

        # עומק
        if min_knee_angle is None or min_knee_angle > DEPTH_MAX_KNEE_ANGLE:
            feedback.append("Go deeper")
            score -= 2

        # גב
        if min_torso_angle is None or min_torso_angle < TORSO_LEAN_MIN:
            feedback.append("Stand taller")
            score -= 2

        # Valgus
        if valgus_bad_frames > 0:
            feedback.append("Avoid knee collapse")
            score -= 2

        if score < 4:
            score = 4.0

        return score, feedback

    def update(self, knee_angle, torso_angle, valgus_ok, frame_no):
        # קביעת סטייג'
        if knee_angle < ANGLE_DOWN_THRESH:
            if self.stage != 'down':
                self.stage = 'down'
                self._start_rep(frame_no)

        elif knee_angle > ANGLE_UP_THRESH:
            if self.stage == 'down':
                # סיום חזרה
                self.count += 1
                score, fb = self.evaluate_form(self._curr_min_knee, self._curr_min_torso, self._curr_valgus_bad)
                self._finish_rep(frame_no, score, fb)
            self.stage = 'up'

        # תוך כדי חזרה – עדכן מינ/מקס
        if self.stage == 'down' and self.rep_start_frame is not None:
            self._curr_min_knee = min(self._curr_min_knee, knee_angle)
            self._curr_max_knee = max(self._curr_max_knee, knee_angle)
            self._curr_min_torso = min(self._curr_min_torso, torso_angle)
            if not valgus_ok:
                self._curr_valgus_bad += 1

    def result(self):
        if self.count == 0:
            return {
                "squat_count": 0,
                "technique_score": 0.0,
                "good_reps": 0,
                "bad_reps": 0,
                "feedback": [],
                "reps": []
            }

        avg_score = np.mean([r["score"] for r in self.rep_reports])
        technique_score = round(round(avg_score * 2) / 2, 2)

        return {
            "squat_count": self.count,
            "technique_score": technique_score,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "feedback": self.all_feedback,
            "reps": self.rep_reports
        }

# =========================
# Main analysis function
# =========================
def run_bulgarian_analysis(video_path, frame_skip=1, scale=1.0):
    cap = cv2.VideoCapture(video_path)
    counter = BulgarianRepCounter()

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        frame_no = 0
        active_leg = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1
            if frame_skip > 1 and (frame_no % frame_skip) != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                continue

            landmarks = results.pose_landmarks.landmark

            if active_leg is None:
                active_leg = detect_active_leg_simple(landmarks)

            side = "RIGHT" if active_leg == "right" else "LEFT"

            HIP      = getattr(mp_pose.PoseLandmark, f"{side}_HIP").value
            KNEE     = getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value
            ANKLE    = getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value
            SHOULDER = getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value

            hip      = lm_xy(landmarks, HIP, w, h)
            knee     = lm_xy(landmarks, KNEE, w, h)
            ankle    = lm_xy(landmarks, ANKLE, w, h)
            shoulder = lm_xy(landmarks, SHOULDER, w, h)

            knee_angle  = calculate_angle(hip, knee, ankle)
            torso_angle = calculate_angle(shoulder, hip, knee)
            v_ok        = valgus_ok_norm(landmarks, side)

            counter.update(knee_angle, torso_angle, v_ok, frame_no)

    cap.release()
    cv2.destroyAllWindows()
    return counter.result()

# =========================
# CLI usage example
# =========================
if __name__ == "__main__":
    path = "video1.mp4"
    res = run_bulgarian_analysis(path, frame_skip=3, scale=0.4)
    print(res)

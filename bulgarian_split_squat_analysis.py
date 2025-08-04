import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os

# ==== Constants ====
ANGLE_DOWN_THRESH = 95
ANGLE_UP_THRESH = 160
GOOD_REP_MIN_SCORE = 8.0
TORSO_LEAN_MIN = 135
VALGUS_X_TOL = 0.03

mp_pose = mp.solutions.pose

# ==== Overlay Function ====
def draw_overlay(frame, reps, feedback):
    h, w, _ = frame.shape
    bar_height = int(h * 0.07)

    # Top bar – reps
    top_overlay = frame.copy()
    cv2.rectangle(top_overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(top_overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, f"Reps: {reps}", (20, int(bar_height * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Bottom bar – feedback
    if feedback:
        bottom_overlay = frame.copy()
        cv2.rectangle(bottom_overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(bottom_overlay, 0.6, frame, 0.4, 0)
        text_size = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = int((w - text_size[0]) / 2)
        cv2.putText(frame, feedback, (text_x, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame

# ==== Helper Functions ====
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def lm_xy(landmarks, idx, w, h):
    return (landmarks[idx].x * w, landmarks[idx].y * h)

def detect_active_leg(landmarks):
    left_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    return 'right' if left_y < right_y else 'left'

def valgus_ok(landmarks, side):
    knee_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x
    ankle_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x
    return not (knee_x < ankle_x - VALGUS_X_TOL)

# ==== Rep Counter ====
class BulgarianRepCounter:
    def __init__(self):
        self.count = 0
        self.stage = None
        self.rep_reports = []
        self.rep_index = 1
        self.rep_start_frame = None
        self.good_reps = 0
        self.bad_reps = 0
        self.all_feedback = set()
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
            self.all_feedback.update(feedback)

        rep_result = {
            "rep_index": self.rep_index,
            "score": round(score, 1),
            "feedback": feedback,
            "start_frame": self.rep_start_frame or 0,
            "end_frame": frame_no,
            "min_knee_angle": round(self._curr_min_knee, 2),
            "max_knee_angle": round(self._curr_max_knee, 2),
            "torso_min_angle": round(self._curr_min_torso, 2)
        }

        self.rep_reports.append(rep_result)
        self.rep_index += 1
        self.rep_start_frame = None

    def evaluate_form(self, min_knee_angle, min_torso_angle, valgus_bad_frames):
        feedback = []
        score = 10.0
        if min_torso_angle < TORSO_LEAN_MIN:
            feedback.append("Keep your back straight")
            score -= 2
        if valgus_bad_frames > 0:
            feedback.append("Avoid knee collapse")
            score -= 2
        return score, feedback

    def update(self, knee_angle, torso_angle, valgus_ok, frame_no):
        if knee_angle < ANGLE_DOWN_THRESH:
            if self.stage != 'down':
                self.stage = 'down'
                self._start_rep(frame_no)
        elif knee_angle > ANGLE_UP_THRESH and self.stage == 'down':
            self.count += 1
            score, feedback = self.evaluate_form(self._curr_min_knee, self._curr_min_torso, self._curr_valgus_bad)
            self._finish_rep(frame_no, score, feedback)
            self.stage = 'up'

        if self.stage == 'down' and self.rep_start_frame:
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

        if self.bad_reps == 0 and len(self.all_feedback) == 0:
            return {
                "squat_count": self.count,
                "technique_score": 10.0,
                "good_reps": self.good_reps,
                "bad_reps": 0,
                "feedback": ["Great form! Keep it up 💪"],
                "reps": self.rep_reports
            }

        return {
            "squat_count": self.count,
            "technique_score": technique_score,
            "good_reps": self.good_reps,
            "bad_reps": self.bad_reps,
            "feedback": list(self.all_feedback),
            "reps": self.rep_reports
        }

# ==== Main Function ====
def run_bulgarian_analysis(video_path, frame_skip=1, scale=1.0, output_path="analyzed_output.mp4", feedback_path="feedback_summary.txt"):
    cap = cv2.VideoCapture(video_path)
    counter = BulgarianRepCounter()
    frame_no = 0
    active_leg = None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_skip > 1 and (frame_no % frame_skip) != 0:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        h, w = frame.shape[:2]
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 8, (w, h))

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            out.write(frame)
            continue

        landmarks = results.pose_landmarks.landmark
        if active_leg is None:
            active_leg = detect_active_leg(landmarks)

        side = "RIGHT" if active_leg == "right" else "LEFT"
        hip = lm_xy(landmarks, getattr(mp_pose.PoseLandmark, f"{side}_HIP").value, w, h)
        knee = lm_xy(landmarks, getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value, w, h)
        ankle = lm_xy(landmarks, getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value, w, h)
        shoulder = lm_xy(landmarks, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value, w, h)

        knee_angle = calculate_angle(hip, knee, ankle)
        torso_angle = calculate_angle(shoulder, hip, knee)
        v_ok = valgus_ok(landmarks, side)

        counter.update(knee_angle, torso_angle, v_ok, frame_no)

        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Overlay
        feedbacks = []
        if counter.stage == "down":
            if torso_angle < TORSO_LEAN_MIN:
                feedbacks.append("Keep your back straight")
            if not v_ok:
                feedbacks.append("Avoid knee collapse")

        feedback = " | ".join(feedbacks) if feedbacks else ""
        frame = draw_overlay(frame, reps=counter.count, feedback=feedback)

        out.write(frame)

    pose.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    result = counter.result()

    with open(feedback_path, "w", encoding="utf-8") as f:
        f.write(f"Total Reps: {result['squat_count']}\n")
        f.write(f"Technique Score: {result['technique_score']}/10\n")
        if result["feedback"]:
            f.write("Feedback:\n")
            for fb in result["feedback"]:
                f.write(f"- {fb}\n")

    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        encoded_path
    ])
    os.remove(output_path)
    output_path = encoded_path

    return result, output_path, feedback_path


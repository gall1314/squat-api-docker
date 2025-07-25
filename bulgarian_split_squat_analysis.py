import cv2
import mediapipe as mp
import numpy as np

# === Setup ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# === Helper Functions ===
def get_landmark_coords(landmarks, index, shape):
    h, w = shape[:2]
    return int(landmarks[index].x * w), int(landmarks[index].y * h)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def detect_active_leg(landmarks):
    left = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    return 'right' if left < right else 'left'

# === Rep Counter & Feedback ===
class BulgarianRepCounter:
    def __init__(self):
        self.count = 0
        self.stage = None
        self.feedback = ""
        self.score = 10
        self.rep_reports = []
        self.good_reps = 0
        self.bad_reps = 0
        self.all_feedback = []
        self.rep_index = 1
        self.rep_start_frame = None

    def evaluate_form(self, hip, knee, ankle, shoulder):
        knee_angle = calculate_angle(hip, knee, ankle)
        torso_angle = calculate_angle(shoulder, hip, knee)
        knee_valgus_angle = calculate_angle(hip, knee, ankle)  # TODO: fix for true valgus

        feedback = []
        score = 10

        if knee_angle > 110:
            feedback.append("Go deeper")
            score -= 2

        if torso_angle < 145:
            feedback.append("Stand taller")
            score -= 2

        if knee_valgus_angle < 150:
            feedback.append("Avoid knee collapse")
            score -= 2

        return score, feedback

    def update(self, hip, knee, ankle, shoulder, frame_number):
        angle = calculate_angle(hip, knee, ankle)

        if angle < 90:
            if self.stage != 'down':
                self.stage = 'down'
                self.rep_start_frame = frame_number

        elif angle > 160 and self.stage == 'down':
            self.stage = 'up'
            self.count += 1
            score, feedback = self.evaluate_form(hip, knee, ankle, shoulder)

            if score >= 8:
                self.good_reps += 1
            else:
                self.bad_reps += 1
                self.all_feedback.extend(feedback)

            rep_result = {
                "rep_index": self.rep_index,
                "score": round(score, 1),
                "feedback": feedback,
                "start_frame": self.rep_start_frame or 0,
                "end_frame": frame_number,
                "min_knee_angle": angle,
                "max_knee_angle": angle  # אפשר לשפר אם שומרים זוויות לאורך החזרה
            }
            self.rep_reports.append(rep_result)
            self.rep_index += 1

    def get_result(self, video_path):
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

# === Main Code ===
def run_bulgarian_analysis(video_path):
    cap = cv2.VideoCapture(video_path)
    counter = BulgarianRepCounter()

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            frame = cv2.resize(frame, (960, 540))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                shape = frame.shape
                active = detect_active_leg(landmarks)

                if active == 'right':
                    hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, shape)
                    knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value, shape)
                    ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value, shape)
                    shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, shape)
                else:
                    hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, shape)
                    knee = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value, shape)
                    ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value, shape)
                    shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, shape)

                counter.update(hip, knee, ankle, shoulder, frame_number)

        cap.release()
        cv2.destroyAllWindows()

    return counter.get_result(video_path)

# === Run Example ===
if __name__ == "__main__":
    result = run_bulgarian_analysis("video1.mp4")
    print(result)


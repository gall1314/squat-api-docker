import cv2
import mediapipe as mp
import numpy as np
import time

# ====== Helper Functions ======

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def get_coords(landmarks, index, shape):
    h, w = shape[:2]
    return [
        landmarks[index].x * w,
        landmarks[index].y * h
    ]

def get_feedback_and_score(knee_angle, back_angle, heel_y, foot_index_y, knee_x, foot_x,
                           shoulder_x, hip_x, front_hip_y, back_knee_y, front_knee_y):
    feedback = []
    score = 10

    if knee_angle > 100:
        feedback.append("⚠️ Go deeper; front knee isn't bent enough.")
        score -= 2

    if back_angle < 150:
        feedback.append("⚠️ Keep your torso more upright.")
        score -= 1

    if heel_y > foot_index_y + 5:
        feedback.append("⚠️ Front heel is lifting. Keep it grounded.")
        score -= 1

    if knee_x > foot_x + 20:
        feedback.append("⚠️ Front knee is moving too far past the foot.")
        score -= 1

    if abs(shoulder_x - hip_x) > 30:
        feedback.append("⚠️ Avoid leaning sideways during the rep.")
        score -= 1

    if front_hip_y < front_knee_y - 20:
        feedback.append("⚠️ Hip too high — lower for proper depth.")
        score -= 1

    if back_knee_y > front_knee_y + 30:
        feedback.append("⚠️ Back knee might be touching the ground.")
        score -= 1

    return feedback, max(1, score)

# ====== Setup ======

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

counter = 0
stage = None
side = "RIGHT"  # Use "LEFT" for left leg
start_time = time.time()

rep_reports = []

rep_started = False
down_frame_count = 0
DOWN_FRAMES_REQUIRED = 3

cap = cv2.VideoCapture("video2.mp4")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        shape = image.shape

        try:
            lm = results.pose_landmarks.landmark

            # Front leg & body
            hip = get_coords(lm, getattr(mp_pose.PoseLandmark, f"{side}_HIP"), shape)
            knee = get_coords(lm, getattr(mp_pose.PoseLandmark, f"{side}_KNEE"), shape)
            ankle = get_coords(lm, getattr(mp_pose.PoseLandmark, f"{side}_ANKLE"), shape)
            shoulder = get_coords(lm, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER"), shape)
            heel = get_coords(lm, getattr(mp_pose.PoseLandmark, f"{side}_HEEL"), shape)
            foot_index = get_coords(lm, getattr(mp_pose.PoseLandmark, f"{side}_FOOT_INDEX"), shape)

            # Back leg
            other_side = "LEFT" if side == "RIGHT" else "RIGHT"
            back_knee = get_coords(lm, getattr(mp_pose.PoseLandmark, f"{other_side}_KNEE"), shape)

            # Angles
            knee_angle = calculate_angle(hip, knee, ankle)
            back_angle = calculate_angle(shoulder, hip, knee)

            # === Improved Rep Detection ===
            if knee_angle < 100:
                down_frame_count += 1
                if down_frame_count >= DOWN_FRAMES_REQUIRED:
                    rep_started = True
            else:
                down_frame_count = 0

            if knee_angle > 160 and rep_started:
                counter += 1
                rep_started = False

                # Extra biomechanical values
                shoulder_x = shoulder[0]
                hip_x = hip[0]
                front_hip_y = hip[1]
                front_knee_y = knee[1]
                back_knee_y = back_knee[1]

                feedback, score = get_feedback_and_score(
                    knee_angle, back_angle, heel[1], foot_index[1],
                    knee[0], foot_index[0],
                    shoulder_x, hip_x, front_hip_y, back_knee_y, front_knee_y
                )

                rep_reports.append({
                    "rep": counter,
                    "score": score,
                    "feedback": feedback
                })

            # === Display ===
            cv2.putText(image, f"Reps: {counter}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(image, f"Knee Angle: {int(knee_angle)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(image, f"State: {'Down' if rep_started else 'Up'}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if rep_reports:
                last = rep_reports[-1]
                y0 = 150
                cv2.putText(image, f"Score: {last['score']}/10", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for i, msg in enumerate(last['feedback']):
                    cv2.putText(image, msg, (20, y0 + (i+1)*30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        except Exception as e:
            print("Tracking failed:", e)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        image = cv2.resize(image, (640, 720))
        cv2.imshow("Bulgarian Split Squat Tracker", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# === Report Summary ===
print("\n=== Bulgarian Split Squat Report ===")
for rep in rep_reports:
    print(f"Rep {rep['rep']}: Score {rep['score']}/10")
    for fb in rep['feedback']:
        print("  -", fb)

# === Final Summary ===
technique_score = round(np.mean([r['score'] for r in rep_reports]), 2) if rep_reports else 0
good_reps = sum(1 for r in rep_reports if r['score'] == 10)
bad_reps = sum(1 for r in rep_reports if r['score'] != 10)

summary = {
    "squat_count": len(rep_reports),
    "technique_score": technique_score,
    "good_reps": good_reps,
    "bad_reps": bad_reps
}

print("\n=== Bulgarian Split Squat Report ===")
print("Total Reps:", summary["squat_count"])
print("Technique Score:", summary["technique_score"])
print("Good Reps:", summary["good_reps"])
print("Bad Reps:", summary["bad_reps"])

for rep in rep_reports:
    print(f"\nRep {rep['rep']}: Score {rep['score']}/10")
    for fb in rep['feedback']:
        print("  -", fb)

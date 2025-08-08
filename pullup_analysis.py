# --- הוספה מתחת ל-imports הקיימים ---
import cv2
import numpy as np
import mediapipe as mp

_DRAW_LMS = [
    "LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST","LEFT_HIP","RIGHT_HIP",
    "LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE",
]
_SKELETON_EDGES = [
    ("LEFT_SHOULDER","LEFT_ELBOW"), ("LEFT_ELBOW","LEFT_WRIST"),
    ("RIGHT_SHOULDER","RIGHT_ELBOW"), ("RIGHT_ELBOW","RIGHT_WRIST"),
    ("LEFT_SHOULDER","RIGHT_SHOULDER"),
    ("LEFT_HIP","RIGHT_HIP"),
    ("LEFT_SHOULDER","LEFT_HIP"), ("RIGHT_SHOULDER","RIGHT_HIP"),
    ("LEFT_HIP","LEFT_KNEE"), ("LEFT_KNEE","LEFT_ANKLE"),
    ("RIGHT_HIP","RIGHT_KNEE"), ("RIGHT_KNEE","RIGHT_ANKLE"),
]

def _draw_skeleton(frame, lms):
    h, w = frame.shape[:2]
    def pt(name):
        if name not in lms: return None
        x,y,_ = lms[name]; return int(x*w), int(y*h)
    for a,b in _SKELETON_EDGES:
        pa, pb = pt(a), pt(b)
        if pa and pb: cv2.line(frame, pa, pb, (255,255,255), 2)
    for name in _DRAW_LMS:
        p = pt(name)
        if p: cv2.circle(frame, p, 3, (255,255,255), -1)

def render_pullup_video(input_path, output_path, analysis_result,
                        frame_skip=3, scale=0.3):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) / max(frame_skip,1)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w0, h0))
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % max(frame_skip,1) != 0:
            i += 1; continue
        frame = cv2.resize(frame, (w0, h0))
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lms = { mp.solutions.pose.PoseLandmark(j).name:(lm.x,lm.y,lm.z)
                    for j,lm in enumerate(res.pose_landmarks.landmark) }
            _draw_skeleton(frame, lms)
        out.write(frame)
        i += 1

    cap.release(); out.release(); pose.close()

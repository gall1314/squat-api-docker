# pullup_analysis.py
import os
import cv2
import numpy as np
import mediapipe as mp

# ============================================================
# Pull-up analysis: always renders video and returns video_path
# Compatible with app.py that expects {"video_path": <path>, ...}
# ============================================================

mp_pose = mp.solutions.pose

# -------------------------
# Drawing helpers (skeleton)
# -------------------------
_DRAW_LMS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
]
_SKELETON_EDGES = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
    (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
    (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
    (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
    (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
    (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value),
]

def _draw_skeleton(frame, norm_landmarks):
    """norm_landmarks: list of (x,y,_) normalized to [0,1]"""
    h, w = frame.shape[:2]
    def pt(i):
        if i < 0 or i >= len(norm_landmarks): return None
        x, y = norm_landmarks[i][0], norm_landmarks[i][1]
        if x is None or y is None: return None
        return int(x * w), int(y * h)

    # edges
    for a, b in _SKELETON_EDGES:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, (255, 255, 255), 2)

    # joints
    for i in _DRAW_LMS:
        p = pt(i)
        if p:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)

# -------------------------
# ROM donut (ROM-Up%)
# -------------------------
def _draw_rom_donut(frame, rom_percent, center=(110, 110), radius=60):
    """rom_percent in [0..100]"""
    pct = int(max(0, min(100, rom_percent)))
    # background ring
    cv2.circle(frame, center, radius, (60,60,60), 10)
    # arc
    angle = int(360 * (pct / 100.0))
    for a in range(0, angle):
        theta = np.deg2rad(a - 90)
        x = int(center[0] + radius * np.cos(theta))
        y = int(center[1] + radius * np.sin(theta))
        cv2.circle(frame, (x, y), 5, (80, 220, 120), -1)
    # labels
    cv2.putText(frame, f"{pct}%", (center[0] - 28, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "ROM-Up", (center[0]-40, center[1]-radius-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)

# -------------------------
# Core pull-up logic
# -------------------------
def _rom_up_percent_from_landmarks(lms):
    """
    ROM-Up% פשוט: כמה ה-LEFT_WRIST גבוה יחסית ל-LEFT_SHOULDER.
    0% ~ שורש כף-יד בגובה שווה/מתחת, 100% ~ גבוה משמעותית.
    הסקיילינג אמפירי כדי להיראות דומה לגרפים שלך.
    """
    try:
        wrist_y = lms[mp_pose.PoseLandmark.LEFT_WRIST.value][1]
        shoulder_y = lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1]
    except Exception:
        return 0
    # קטן יותר ב-y = גבוה יותר בפריים; הופכים לסולם % פשוט
    raw = (shoulder_y - wrist_y) * 300.0  # סקלת רגישות
    return int(max(0, min(100, raw)))

def _count_reps_from_rom_series(rom_series, up_thresh=80, down_thresh=20):
    """
    ספירת ריפים בלוגיקה המקורית שתיארת:
    כשה-ROM עובר את 80% מלמטה → עולה rep (direction 0→1)
    כשיורד מתחת 20% → מוכנים לריפ הבא (direction 1→0)
    """
    rep_count = 0
    direction = 0  # 0=down/ready, 1=up/in rep
    for rom in rom_series:
        if rom > up_thresh and direction == 0:
            rep_count += 1
            direction = 1
        elif rom < down_thresh and direction == 1:
            direction = 0
    return rep_count

# -------------------------
# Main entry (always render)
# -------------------------
def run_pullup_analysis(input_path, output_path, frame_skip=3, scale=0.4):
    """
    - קורא וידאו, מריץ MediaPipe Pose אחת (מדוגם ב-frame_skip).
    - מחשב ROM-Up% לכל פריים דגום + ספירת ריפים בלוגיקה המקורית.
    - מרנדר וידאו עם שלד + דונאט ROM, כותב ל-output_path (mp4v, FPS אחרי דילול).
    - מחזיר dict עם video_path (כמו בשאר התרגילים), rep_count וכו'.
    """
    cap = cv2.VideoCapture(input_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(src_fps) / max(frame_skip, 1)  # FPS בפלט – כמו סקוואט/בולגרי

    out_w = int((cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) * scale)
    out_h = int((cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) * scale)
    out_w = max(out_w, 64)
    out_h = max(out_h, 64)

    # Video writer (mp4v) – עקבי עם התרגילים האחרים
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    # נשמור landmarks לכל פריים דגום כדי לחסוך ריצה נוספת ברינדור
    sampled_landmarks = []
    rom_series = []

    frame_id = 0
    wrote_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % max(frame_skip, 1) != 0:
            frame_id += 1
            continue

        # resize for compute + render
        frame_small = cv2.resize(frame, (out_w, out_h))
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lms = [(lm.x, lm.y, lm.z) for lm in res.pose_landmarks.landmark]
        else:
            lms = []

        sampled_landmarks.append(lms)

        # ROM-Up% מהעלייה
        rom = _rom_up_percent_from_landmarks(lms) if lms else 0
        rom_series.append(rom)

        # רינדור על אותו פריים (אין ריצת Pose שנייה)
        if lms:
            _draw_skeleton(frame_small, lms)
        _draw_rom_donut(frame_small, rom)

        out.write(frame_small)
        wrote_frames += 1
        frame_id += 1

    cap.release()
    pose.close()
    out.release()

    # ספירת ריפים – בדיוק בלוגיקה המקורית שתיארת
    rep_count = _count_reps_from_rom_series(rom_series, up_thresh=80, down_thresh=20)

    # ציון בסיסי – אם צריך אפשר להקשיח/לשנות כמו בשאר
    technique_score = 10.0
    feedback = []

    # לוג קטן לעזרה בדיבוג
    try:
        size = os.path.getsize(output_path)
        print(f"[pullup] wrote: {output_path} | frames={wrote_frames} | fps={fps:.2f} | size={size} bytes")
    except Exception as e:
        print("[pullup] output check failed:", e)

    return {
        "video_path": output_path,
        "rep_count": int(rep_count),
        "technique_score": float(round(technique_score, 1)),
        "feedback": feedback,
    }


# -------------
# CLI (optional)
# -------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        src, dst = sys.argv[1], sys.argv[2]
        res = run_pullup_analysis(src, dst, frame_skip=3, scale=0.4)
        print(res)
    else:
        print("Usage: python pullup_analysis.py <input.mp4> <output.mp4>")


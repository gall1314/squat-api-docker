# squat_analysis.py
import os
import cv2
import math
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / CONSTANTS =====================
TOP_BAR_FRAC        = 0.065
BOTTOM_BAR_FRAC     = 0.07
BAR_BG_ALPHA        = 0.55

# Donut style
DEPTH_RADIUS_SCALE   = 0.70   # relative to top bar height
DEPTH_THICKNESS_FRAC = 0.28
DEPTH_RING_BG        = (70, 70, 70)    # gray ring background
DEPTH_COLOR          = (40, 200, 80)   # green (BGR)

# Fonts
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE   = 18
def _load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()
REPS_FONT        = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

# Depth smoother (קליל, EMA עם peak-hold קצר לדינמיקה חלקה)
DEPTH_ALPHA = 0.35
DEPTH_PEAK_HOLD_FRAMES = 6

# Squat heuristics
PERFECT_MIN_KNEE = 60.0    # קירוב לשפיץ עומק "טוב"
STAND_KNEE_ANGLE = 160.0

# ===================== UTIL: robust JSON normalization =====================
def _as_py_int(x):
    try: return int(x)
    except: return 0

def _as_py_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except:
        return 0.0

def _as_str_list(xs):
    if xs is None: return []
    if isinstance(xs, str): return [xs]
    try: return [str(s) for s in xs]
    except: return []

def _as_int_list(xs):
    if xs is None: return []
    out = []
    try:
        for v in xs:
            try: out.append(int(v))
            except: pass
    except TypeError:
        pass
    return out

def _public_url(local_path: str, public_base_url: str) -> str:
    if not local_path: return ""
    base = (public_base_url or "").rstrip("/")
    if not base: return ""
    name = os.path.basename(local_path)
    return f"{base}/media/{name}"

def finalize_popup_payload(raw: dict, public_base_url: str = "") -> dict:
    counter       = _as_py_int(raw.get("squat_count") or raw.get("rep_count") or 0)
    good_reps     = _as_py_int(raw.get("good_reps"))
    bad_reps      = _as_py_int(raw.get("bad_reps"))
    technique     = _as_py_float(raw.get("technique_score"))
    feedback      = _as_str_list(raw.get("feedback"))
    problem_reps  = _as_int_list(raw.get("problem_reps"))
    video_path    = raw.get("video_path") or ""
    analyzed_url  = raw.get("analyzed_video_url") or _public_url(video_path, public_base_url)

    return {
        "technique_score": technique,     # float תמיד
        "rep_count": counter,
        "squat_count": counter,           # לשמירה על תאימות
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": feedback,             # list[str]
        "problem_reps": problem_reps,     # list[int]
        "video_path": video_path,         # תאימות לשרת
        "analyzed_video_url": analyzed_url or None,  # ל‑UI; None עדיף על מחרוזת ריקה
        "schema_version": 2
    }

# ===================== GEOMETRY HELPERS =====================
def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb)) + 1e-9
    cosine_angle = np.dot(ab, cb) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

# ===================== DRAW HELPERS =====================
def draw_plain_text(pil_img, xy, text, font, color=(255,255,255)):
    ImageDraw.Draw(pil_img).text((int(xy[0]), int(xy[1])), text, font=font, fill=color)
    return np.array(pil_img)

def draw_depth_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius   = int(radius)
    thickness = int(thickness)
    # ring background
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    # green arc
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    h, w, _ = frame.shape
    bar_h = int(h * TOP_BAR_FRAC)

    # Top bar
    over = frame.copy()
    cv2.rectangle(over, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)

    # Reps
    pil = Image.fromarray(frame)
    frame = draw_plain_text(pil, (16, int(bar_h * 0.22)), f"Reps: {reps}", REPS_FONT)

    # Depth donut
    margin = 12
    radius = int(bar_h * DEPTH_RADIUS_SCALE)
    thick  = max(3, int(radius * DEPTH_THICKNESS_FRAC))
    cx = w - margin - radius
    cy = int(bar_h * 0.52)
    frame = draw_depth_donut(frame, (cx, cy), radius, thick, depth_pct)

    # Depth labels in donut
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"
    pct_txt   = f"{int(depth_pct*100)}%"
    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt, font=DEPTH_PCT_FONT)
    gap = max(2, int(radius * 0.10))
    block_h = DEPTH_LABEL_FONT_SIZE + gap + DEPTH_PCT_FONT_SIZE
    base_y  = cy - block_h // 2
    draw.text((cx - int(label_w // 2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pct_w // 2), base_y + DEPTH_LABEL_FONT_SIZE + gap),
              pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Bottom feedback bar
    if feedback:
        bottom_h = int(h * BOTTOM_BAR_FRAC)
        over2 = frame.copy()
        cv2.rectangle(over2, (0, h - bottom_h), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(over2, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)

        pil2 = Image.fromarray(frame)
        draw2 = ImageDraw.Draw(pil2)
        tw = draw2.textlength(feedback, font=FEEDBACK_FONT)
        tx = (w - int(tw)) // 2
        ty = h - bottom_h + 6
        draw2.text((tx, ty), feedback, font=FEEDBACK_FONT, fill=(255,255,255))
        frame = np.array(pil2)

    return frame

# ===================== MAIN ANALYSIS =====================
def run_squat_analysis(
    video_path,
    frame_skip=3,
    scale=0.4,
    output_path="squat_analyzed.mp4",
    feedback_path="squat_feedback.txt",
    public_base_url: str = ""
):
    """
    Squat analysis aligned with Bulgarian overlay + robust JSON normalization.
    Returns a dict with BOTH 'rep_count' and 'squat_count', safe lists, and
    a public 'analyzed_video_url' when PUBLIC_BASE_URL is set.
    """
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return finalize_popup_payload({"technique_score": 0, "squat_count": 0, "feedback": ["Could not open video"], "video_path": ""}, public_base_url)

    # -------- Metrics --------
    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []

    # -------- Rep state --------
    stage = None
    frame_idx = 0
    last_rep_frame = -999
    MIN_FRAMES_BETWEEN_REPS = 10

    # -------- In-rep trackers --------
    start_knee_angle = None
    rep_min_knee_angle = 180.0
    last_rep_feedback = []

    # -------- Depth smoothing --------
    depth_smooth = 0.0
    peak_hold = 0
    def update_depth(dt, target):
        nonlocal depth_smooth, peak_hold
        # EMA
        depth_smooth = DEPTH_ALPHA * target + (1 - DEPTH_ALPHA) * depth_smooth
        # peak-hold decay
        if peak_hold > 0:
            peak_hold -= 1
        else:
            depth_smooth *= 0.985
        depth_smooth = float(np.clip(depth_smooth, 0.0, 1.0))
        return depth_smooth

    # -------- Video writer --------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            rt_feedback = ""

            if not results.pose_landmarks:
                # overlay even without landmarks
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=update_depth(dt, 0.0))
                out.write(frame)
                continue

            try:
                lm = results.pose_landmarks.landmark
                R = mp_pose.PoseLandmark

                # Points (right side)
                hip      = np.array([lm[R.RIGHT_HIP.value].x,      lm[R.RIGHT_HIP.value].y])
                knee     = np.array([lm[R.RIGHT_KNEE.value].x,     lm[R.RIGHT_KNEE.value].y])
                ankle    = np.array([lm[R.RIGHT_ANKLE.value].x,    lm[R.RIGHT_ANKLE.value].y])
                shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])
                heel_y   = lm[R.RIGHT_HEEL.value].y

                knee_angle = calculate_angle(hip, knee, ankle)

                # Stage transitions
                if knee_angle < 100:
                    if stage != "down":
                        last_rep_feedback = []
                        start_knee_angle = float(knee_angle)
                        rep_min_knee_angle = 180.0
                    stage = "down"

                if stage == "down":
                    rep_min_knee_angle = min(rep_min_knee_angle, knee_angle)
                    if start_knee_angle is not None:
                        denom = max(10.0, (start_knee_angle - PERFECT_MIN_KNEE))
                        depth_target = float(np.clip((start_knee_angle - rep_min_knee_angle) / denom, 0, 1))
                        update_depth(dt, depth_target)

                # End of rep: back to standing-ish
                if knee_angle > STAND_KNEE_ANGLE and stage == "down":
                    # score & feedback
                    feedbacks = []
                    penalty = 0.0

                    # Depth heuristic using hip-to-heel distance proxy
                    hip_to_heel_dist = abs(hip[1] - heel_y)
                    if hip_to_heel_dist > 0.48:
                        feedbacks.append("Try to squat deeper")
                        penalty += 3
                    elif hip_to_heel_dist > 0.45:
                        feedbacks.append("Almost there — go a bit lower")
                        penalty += 1.5
                    elif hip_to_heel_dist > 0.43:
                        feedbacks.append("Looking good — just a bit more depth")
                        penalty += 0.5

                    # Back posture (simple proxy with knee/shoulder/hip)
                    back_angle = calculate_angle(shoulder, hip, knee)
                    if back_angle < 140:
                        feedbacks.append("Try to keep your back a bit straighter")
                        penalty += 1.0

                    # Lockout
                    if knee_angle < 160:
                        feedbacks.append("Finish with knees fully extended")
                        penalty += 1.0

                    if not feedbacks:
                        score = 10.0
                    else:
                        penalty = min(penalty, 6)
                        score = round(max(4, 10 - penalty) * 2) / 2

                    for f in feedbacks:
                        if f not in overall_feedback:
                            overall_feedback.append(f)
                    last_rep_feedback = feedbacks[:]

                    # peak-hold then fade for donut
                    peak_hold = DEPTH_PEAK_HOLD_FRAMES
                    start_knee_angle = None
                    stage = "up"

                    # Count rep
                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        counter += 1
                        last_rep_frame = frame_idx
                        if score >= 9.5:
                            good_reps += 1
                        else:
                            bad_reps += 1
                            problem_reps.append(counter)
                        all_scores.append(score)

                # Draw skeleton (fast)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # Overlay
                frame = draw_overlay(
                    frame,
                    reps=counter,
                    feedback=" | ".join(last_rep_feedback) if last_rep_feedback else None,
                    depth_pct=depth_smooth
                )
                out.write(frame)

            except Exception:
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=update_depth(dt, 0.0))
                if out is not None:
                    out.write(frame)
                continue

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Technique score
    technique_score = round((np.mean(all_scores) if all_scores else 0.0) * 2) / 2
    if not overall_feedback:
        overall_feedback.append("Great form! Keep it up 💪")

    # Write feedback file (best-effort)
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if overall_feedback:
                f.write("Feedback:\n")
                for fb in overall_feedback:
                    f.write(f"- {fb}\n")
    except Exception:
        pass

    # Encode faststart
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            encoded_path
        ], check=False)
        if os.path.exists(output_path) and os.path.exists(encoded_path):
            os.remove(output_path)
    except Exception:
        pass

    final_video_path = encoded_path if os.path.exists(encoded_path) else (
        output_path if os.path.exists(output_path) else ""
    )

    # --- RAW payload (לפני נירמול) ---
    raw = {
        "technique_score": float(technique_score),
        "squat_count": int(counter),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": overall_feedback,     # list[str]
        "problem_reps": problem_reps,     # list[int]
        "video_path": final_video_path,   # יכול להיות "", לא יקלקל את החלק שלפני הווידאו
    }

    # החזרה מנורמלת ובטוחה לפופ-אפ (מונע "מסך אפור")
    return finalize_popup_payload(raw, public_base_url=os.getenv("PUBLIC_BASE_URL", public_base_url))


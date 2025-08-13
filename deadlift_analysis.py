# deadlift_analysis.py
import os
import cv2
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / CONSTANTS =====================
# Top/Bottom bars
TOP_BAR_FRAC        = 0.065
BOTTOM_BAR_FRAC     = 0.07
BAR_BG_ALPHA        = 0.55

# Donut style
DEPTH_RADIUS_SCALE   = 0.70   # relative to top bar height
DEPTH_THICKNESS_FRAC = 0.28
DEPTH_RING_BG        = (70, 70, 70)    # gray ring background
DEPTH_COLOR          = (40, 200, 80)   # green (BGR)

# Fonts (fallback to default if font file is missing)
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

# Live depth smoothing (simple EMA)
DEPTH_ALPHA = 0.35

# Deadlift hinge normalization range
HINGE_START_THRESH = 0.08   # when delta_x crosses this, we consider a rep started
HINGE_GOOD_TARGET  = 0.22   # generous target for a deep hinge

# Debounce / rep state
MIN_FRAMES_BETWEEN_REPS = 10

# ===================== GEOMETRY HELPERS =====================
def calculate_angle(a, b, c):
    """Returns angle ABC in degrees for points a,b,c (2D)."""
    a, b, c = map(np.array, [a, b, c])
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb)) + 1e-9
    cosine_angle = np.dot(ab, cb) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

def analyze_back_curvature(shoulder, hip, reference_point, threshold=0.04):
    """
    Signed curvature magnitude of spine vs. reference line (shoulder->hip).
    Negative => inward rounding (flexion). 'is_dangerous' if beyond threshold.
    """
    line_vec = hip - shoulder
    nrm = np.linalg.norm(line_vec) + 1e-9
    line_unit = line_vec / nrm
    proj_len = np.dot(reference_point - shoulder, line_unit)
    proj_point = shoulder + proj_len * line_unit
    offset_vec = reference_point - proj_point
    direction_sign = np.sign(offset_vec[1]) * -1  # inward = negative
    curvature_magnitude = np.linalg.norm(offset_vec)
    signed_curvature = float(direction_sign * curvature_magnitude)
    is_dangerous = signed_curvature < -threshold
    return signed_curvature, is_dangerous

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
    """
    Top bar: Reps + DEPTH donut; Bottom bar: feedback when present.
    """
    h, w, _ = frame.shape
    bar_h = int(h * TOP_BAR_FRAC)

    # Top bar
    over = frame.copy()
    cv2.rectangle(over, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)

    # Reps (left)
    pil = Image.fromarray(frame)
    frame = draw_plain_text(pil, (16, int(bar_h * 0.22)), f"Reps: {reps}", REPS_FONT)

    # Depth donut (right)
    margin = 12
    radius = int(bar_h * DEPTH_RADIUS_SCALE)
    thick  = max(3, int(radius * DEPTH_THICKNESS_FRAC))
    cx = w - margin - radius
    cy = int(bar_h * 0.52)
    frame = draw_depth_donut(frame, (cx, cy), radius, thick, depth_pct)

    # Depth labels inside donut
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
def run_deadlift_analysis(
    video_path,
    frame_skip=3,
    scale=0.4,
    output_path="deadlift_analyzed.mp4",
    feedback_path="deadlift_feedback.txt"
):
    """
    Deadlift analysis aligned with your Bulgarian overlay:
    - Skeleton via drawing_utils (מהיר)
    - Stable rep logic with debounce
    - Live depth donut (hinge progress) with EMA smoothing
    - Bottom feedback only when issues persist
    - H.264 faststart encode
    Returns schema that matches your UI.
    """
    mp_pose_mod = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    # -------- Metrics --------
    counter = good_reps = bad_reps = 0
    all_scores = []
    problem_reps = []
    overall_feedback = []

    # -------- Rep state --------
    rep_in_progress = False
    frame_index = 0
    last_rep_frame = -999

    # -------- In-rep trackers --------
    max_delta_x = 0.0           # hinge depth proxy (hip-shoulder X distance)
    min_knee_angle = 999.0
    spine_flagged = False

    # -------- Live depth (donut) --------
    live_depth_pct = 0.0
    depth_smooth = 0.0

    # -------- Video writer --------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))

    with mp_pose_mod.Pose(
        model_complexity=1,                 # איזון מהירות/דיוק (אפשר 0 למהירות מקס')
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_index % frame_skip != 0:
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # ברירת מחדל: אין פידבק שורה תחתונה
            rt_feedback = ""

            if not results.pose_landmarks:
                # שלד לא זמין — מציגים overlay עם עומק 0
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=0.0)
                out.write(frame)
                continue

            try:
                lm = results.pose_landmarks.landmark

                # שלד מהיר
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose_mod.POSE_CONNECTIONS
                )

                # נקודות עיקריות (ימין כברירת מחדל)
                R = mp_pose_mod.PoseLandmark
                shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])
                hip      = np.array([lm[R.RIGHT_HIP.value].x,      lm[R.RIGHT_HIP.value].y])
                knee     = np.array([lm[R.RIGHT_KNEE.value].x,     lm[R.RIGHT_KNEE.value].y])
                ankle    = np.array([lm[R.RIGHT_ANKLE.value].x,    lm[R.RIGHT_ANKLE.value].y])

                # ראש/אוזן/אף – לבחירת נק’ ייחוס לעקמומיות הגב
                head_candidates = [
                    lm[R.RIGHT_EAR.value],
                    lm[R.LEFT_EAR.value],
                    lm[R.NOSE.value]
                ]
                head_point = None
                for c in head_candidates:
                    if c.visibility > 0.5:
                        head_point = np.array([c.x, c.y])
                        break
                if head_point is None:
                    frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=live_depth_pct)
                    out.write(frame)
                    continue

                # מדדים לפריים
                delta_x = abs(hip[0] - shoulder[0])                 # hinge proxy
                knee_angle = calculate_angle(hip, knee, ankle)      # לניטור “ברכיים נעולות"
                mid_spine = (shoulder + hip) * 0.5 * 0.4 + head_point * 0.6
                curvature, is_rounded = analyze_back_curvature(shoulder, hip, mid_spine)

                # ---------- In-rep ----------
                if rep_in_progress:
                    max_delta_x = max(max_delta_x, delta_x)
                    min_knee_angle = min(min_knee_angle, knee_angle)
                    spine_flagged = spine_flagged or is_rounded

                    # דונאט עומק – נרמול והחלקה
                    raw_depth = (max_delta_x - HINGE_START_THRESH) / max(1e-6, (HINGE_GOOD_TARGET - HINGE_START_THRESH))
                    live_depth_pct = float(np.clip(raw_depth, 0.0, 1.0))
                    depth_smooth = DEPTH_ALPHA * live_depth_pct + (1 - DEPTH_ALPHA) * depth_smooth

                    if is_rounded:
                        rt_feedback = "Keep your back straighter"

                # תחילת חזרה: חציית סף
                if not rep_in_progress and delta_x > HINGE_START_THRESH:
                    # debounce
                    if frame_index - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        rep_in_progress = True
                        max_delta_x = delta_x
                        min_knee_angle = knee_angle
                        spine_flagged = is_rounded
                        live_depth_pct = 0.0
                        depth_smooth = 0.0

                # סוף חזרה: חזרה לעמידה יחסית
                elif rep_in_progress and delta_x < 0.035:
                    if frame_index - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        # בדיקת תנועה מספקת + ברכיים לא “נעולות” כל הזמן
                        moved_enough = (max_delta_x - delta_x) > 0.05
                        knee_softened = min_knee_angle < 170

                        if moved_enough and knee_softened:
                            feedbacks = []
                            penalty = 0.0

                            # סיום יותר זקוף
                            if delta_x > 0.05:
                                feedbacks.append("Try to finish more upright")
                                penalty += 1.0
                            # טורסו עמוק עם ברכיים נעולות
                            if max_delta_x > 0.18 and min_knee_angle > 170:
                                feedbacks.append("Try to bend your knees as you lean forward")
                                penalty += 1.0
                            # חוסר סינכרון ירך/חזה
                            if min_knee_angle > 165 and max_delta_x > 0.2:
                                feedbacks.append("Try to lift your chest and hips together")
                                penalty += 1.0
                            # עיגול גב
                            if spine_flagged:
                                feedbacks.append("Your spine is rounding inward too much")
                                penalty += 2.5

                            score = round(max(4, 10 - penalty) * 2) / 2
                            for f in feedbacks:
                                if f not in overall_feedback:
                                    overall_feedback.append(f)

                            counter += 1
                            last_rep_frame = frame_index
                            if score >= 9.5:
                                good_reps += 1
                            else:
                                bad_reps += 1
                                problem_reps.append(counter)
                            all_scores.append(score)

                    # reset rep
                    rep_in_progress = False
                    max_delta_x = 0.0
                    min_knee_angle = 999.0
                    spine_flagged = False
                    live_depth_pct = 0.0
                    depth_smooth = 0.0
                    rt_feedback = ""

                # Overlay + write
                frame = draw_overlay(
                    frame,
                    reps=counter,
                    feedback=rt_feedback if rt_feedback else None,
                    depth_pct=depth_smooth
                )
                out.write(frame)

            except Exception:
                frame = draw_overlay(frame, reps=counter, feedback=None, depth_pct=0.0)
                if out is not None:
                    out.write(frame)
                continue

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    technique_score = round((np.mean(all_scores) if all_scores else 0) * 2) / 2
    if not overall_feedback:
        overall_feedback.append("Great form! Keep your spine neutral and hinge smoothly. 💪")

    # סיכום לטקסט
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

    # Encode to H.264 faststart
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

    return {
        "technique_score": technique_score,
        "squat_count": counter,          # נשאר בשם הזה בשביל התאמה ל-UI שלך
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": overall_feedback,
        "problem_reps": problem_reps,
        "video_path": final_video_path,
        "feedback_path": feedback_path,
    }



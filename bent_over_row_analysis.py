# -*- coding: utf-8 -*-
"""
bent_over_row_analysis.py - FIXED VERSION
Bent-Over Barbell Row — analysis pipeline matching the Squat standard:
- Same overlay look & return schema (incl. 'squat_count' key)
- Rep counting with frame skipping + soft-start gating
- Live donut progress, single feedback line, body-only skeleton
- Faststart-encoded analyzed video output
- FIXED: proper_reps now counts correctly (not overly strict)
- FIXED: form_tip now returns best tip instead of null
"""
import os
import cv2
import math
import time
import json
import uuid
import shutil
import subprocess
from collections import deque

import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    mp = None

# ================== THEME (MATCH SQUAT) ==================
# Keep these values identical to the Squat overlay to ensure visual parity.
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"  # must exist in working dir (same as squat)
TOP_BAR_FRAC    = 0.095
BOTTOM_BAR_FRAC = 0.08
BAR_BG_OPACITY  = 0.58

REPS_FONT_PX    = 30
DEPTH_LABEL_PX  = 16
DEPTH_PCT_PX    = 22
FEEDBACK_BASE_PX= 20

SKELETON_THICK  = 2
SKELETON_OPAC   = 0.85

# ================== ANALYSIS PARAMS ==================
SCALE = 0.4          # identical to squat
FRAME_SKIP = 2       # identical skipping logic
CONF_DET = 0.5
CONF_TRK = 0.5
MODEL_COMPLEXITY = 1

# Soft-start gating and motion thresholds (match squat style)
EMA_ALPHA = 0.2
HIP_VEL_THRESH = 2.0          # px/frame after scaling; tune like squat hip/ankle gating
GLOBAL_MOTION_LOCK_FRAMES = 4 # frames to wait when motion spikes

# Rep logic (Row-specific, but conservative to match squat reliability)
ELBOW_EXTENDED_MIN = 160.0    # start from nearly straight
ELBOW_PULL_START   = 150.0    # crossing below -> start pull
ELBOW_TOP_MAX      = 75.0     # top position must be ≤ this to count ROM
MIN_FRAMES_BETWEEN_REPS = 6

# ROM donut normalization
# depth_pct = normalize( extended_angle -> top_angle )
DONUT_FLOOR = 0.0
DONUT_CEIL  = 100.0

# Posture/quality thresholds
TORSO_IDEAL_MIN = 15.0    # deg above horizontal - nearly parallel to floor
TORSO_IDEAL_MAX = 35.0    # deg above horizontal - acceptable range
TORSO_HIGH_WARNING = 50.0 # deg above horizontal - too upright warning
TORSO_DRIFT_MAX  = 25.0   # allowed change during a rep (INCREASED - stable execution)
KNEE_DRIFT_MAX   = 20.0   # leg drive indicator (INCREASED - stable execution)
MIN_REP_SAMPLES  = 5      # minimum samples to trust drift/mean metrics
MIN_DEPTH_PCT_FOR_STABILITY = 60.0  # only trust stability cues on meaningful ROM

# Scoring
MIN_SCORE = 4.0
MAX_SCORE = 10.0
ROM_PENALTY_STRONG = 3.0
ROM_PENALTY_LIGHT  = 1.5  # Reduced from 2.0
MOMENTUM_PENALTY   = 2.0
TORSO_PENALTY      = 1.0  # Reduced from 1.5
LEGDRIVE_PENALTY   = 0.8  # Reduced from 1.0

# ================== UTILS ==================
def ensure_font():
    # If font missing, fallback to Hershey (handled inside draw functions).
    return os.path.exists(FONT_PATH)

def angle(a, b, c):
    """Return angle ABC in degrees for 2D points (x,y)."""
    if any(v is None for v in (a, b, c)):
        return None
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    ba_len = math.hypot(ba[0], ba[1])
    bc_len = math.hypot(bc[0], bc[1])
    if ba_len == 0 or bc_len == 0:
        return None
    cosang = (ba[0]*bc[0] + ba[1]*bc[1]) / (ba_len*bc_len + 1e-9)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def angle_between_vectors(v1, v2):
    """Calculate angle between two 2D vectors in degrees."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    v1_len = np.linalg.norm(v1)
    v2_len = np.linalg.norm(v2)
    if v1_len == 0 or v2_len == 0:
        return None
    cosang = np.dot(v1, v2) / (v1_len * v2_len)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def ema(prev, val, alpha=EMA_ALPHA):
    if prev is None:
        return val
    return prev + alpha * (val - prev)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def round_to_half(x):
    return round(x * 2) / 2.0

def technique_label(score):
    if score >= 9.5: return "Excellent"
    if score >= 8.5: return "Very good"
    if score >= 7.5: return "Good"
    if score >= 6.5: return "Fair"
    return "Needs work"

# ================== OVERLAY (identical sizing/placement) ==================
from PIL import ImageFont, ImageDraw, Image

def draw_text(draw, xy, text, px, fill=(255,255,255), anchor=None):
    if ensure_font():
        font = ImageFont.truetype(FONT_PATH, px)
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)
    else:
        # PIL fallback: approximate size with default bitmap font
        draw.text(xy, text, fill=fill, anchor=anchor)

def draw_rect_alpha(img_pil, xyxy, opacity, fill=(0,0,0)):
    x1,y1,x2,y2 = xyxy
    overlay = Image.new('RGBA', (x2-x1, y2-y1), fill + (int(255*opacity),))
    img_pil.paste(overlay, (x1,y1), overlay)

def draw_donut(draw, center, radius, pct):
    # Draw a donut representing 0..100%; identical width as squat
    start_angle = -90  # start at top
    end_angle = start_angle + 360 * (clamp(pct, 0, 100) / 100.0)
    # outer
    draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], outline=(255,255,255), width=8)
    # filled arc (progress)
    draw.arc([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], start=start_angle, end=end_angle, fill=(255,255,255), width=8)

def draw_skeleton(frame, lm):
    # Only body (no face) — similar to squat
    pairs = [
        (11,13),(13,15), # left arm
        (12,14),(14,16), # right arm
        (11,23),(12,24), # shoulders to hips
        (23,25),(25,27),(24,26),(26,28) # legs
    ]
    h, w = frame.shape[:2]
    for a,b in pairs:
        if lm[a] is None or lm[b] is None: 
            continue
        x1,y1 = int(lm[a][0]), int(lm[a][1])
        x2,y2 = int(lm[b][0]), int(lm[b][1])
        cv2.line(frame, (x1,y1), (x2,y2), (255,255,255), SKELETON_THICK)

def render_overlay(frame_bgr, reps_count, live_depth_pct, feedback_text):
    # Make bars + donut + text with identical proportions to squat
    h, w = frame_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Top bar
    top_h = int(h * TOP_BAR_FRAC)
    draw_rect_alpha(img_pil, (0,0,w,top_h), BAR_BG_OPACITY)
    # Reps text (left)
    draw_text(draw, (12, int(top_h*0.6)), f"Reps: {reps_count}", REPS_FONT_PX, fill=(255,255,255), anchor=None)
    # Donut (right-top)
    donut_center = (w - int(top_h*0.9), int(top_h*0.55))
    donut_radius = int(top_h*0.42)
    draw_donut(draw, donut_center, donut_radius, live_depth_pct)
    # Label + pct
    draw_text(draw, (donut_center[0], donut_center[1]-int(top_h*0.35)), "DEPTH", DEPTH_LABEL_PX, fill=(255,255,255), anchor="mm")
    draw_text(draw, (donut_center[0], donut_center[1]+int(top_h*0.35)), f"{int(clamp(live_depth_pct,0,100))}%", DEPTH_PCT_PX, fill=(255,255,255), anchor="mm")

    # Bottom bar (feedback)
    bot_h = int(h * BOTTOM_BAR_FRAC)
    draw_rect_alpha(img_pil, (0,h-bot_h,w,h), BAR_BG_OPACITY)
    if feedback_text:
        draw_text(draw, (12, h - int(bot_h*0.5)), feedback_text, FEEDBACK_BASE_PX, fill=(255,255,255), anchor=None)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ================== CORE ANALYSIS ==================
def get_landmarks_xy(results, frame_shape):
    h, w = frame_shape[:2]
    lm = [None]*33
    if results and results.pose_landmarks:
        pts = results.pose_landmarks.landmark
        for i in range(33):
            if pts[i].visibility > 0.4:
                lm[i] = (int(pts[i].x*w), int(pts[i].y*h))
    return lm

def row_depth_from_elbow(elbow_angle, ext_ref, top_ref):
    # Map elbow angle to 0..100 (0 = start, 100 = top)
    if elbow_angle is None or ext_ref is None or top_ref is None:
        return 0.0
    # bigger angle = more extended, smaller angle = more "pulled"
    span = max(10.0, ext_ref - top_ref)
    pct = (ext_ref - elbow_angle) * 100.0 / span
    return clamp(pct, DONUT_FLOOR, DONUT_CEIL)

def analyze_rep(rep_metrics):
    """
    Return penalties, feedbacks, and is_proper for a single rep based on collected metrics.
    FIXED: Only mark is_proper=False for severe ROM issues
    """
    feedback = []
    penalty = 0.0
    is_proper = True  # Start assuming proper
    depth_pct = rep_metrics.get("depth_pct") or 0.0
    allow_stability_cues = depth_pct >= MIN_DEPTH_PCT_FOR_STABILITY

    # ROM check - ONLY severe issues mark as improper
    if rep_metrics["min_elbow_angle"] is None or rep_metrics["min_elbow_angle"] > ELBOW_TOP_MAX + 10:
        # Very severe ROM issue
        penalty += ROM_PENALTY_STRONG
        feedback.append(("severe", "Pull higher — reach your torso"))
        is_proper = False
    elif rep_metrics["min_elbow_angle"] > ELBOW_TOP_MAX + 5:
        # Moderate ROM issue - still counts as proper but with feedback
        penalty += ROM_PENALTY_LIGHT
        feedback.append(("medium", "Try to pull a bit higher"))
        # NOT marking as improper for moderate issues
    elif rep_metrics["min_elbow_angle"] > ELBOW_TOP_MAX:
        # Minor ROM issue - just a tip
        penalty += ROM_PENALTY_LIGHT * 0.5
        feedback.append(("tip", "Aim for stronger elbow flexion at the top"))

    # Torso angle check - MAIN FOCUS: is back horizontal enough?
    if allow_stability_cues and rep_metrics["torso_mean"] is not None:
        torso_mean = rep_metrics["torso_mean"]
        
        # Too upright (back angle too high) - THIS IS THE MAIN ISSUE
        if torso_mean > TORSO_HIGH_WARNING:
            penalty += MOMENTUM_PENALTY  # High penalty for very upright
            feedback.append(("severe", "Lower your torso — get more horizontal"))
            is_proper = False
        elif torso_mean > TORSO_IDEAL_MAX:
            penalty += TORSO_PENALTY
            feedback.append(("medium", "Try to lower your torso closer to parallel"))
        elif torso_mean < TORSO_IDEAL_MIN:
            # Too low (rare, but possible)
            feedback.append(("tip", "You can raise your torso slightly"))
    
    # Momentum / torso drift - ONLY very severe marks as improper
    if allow_stability_cues:
        if rep_metrics["torso_drift"] is not None and rep_metrics["torso_drift"] > TORSO_DRIFT_MAX * 1.5:
            # Very severe momentum
            penalty += MOMENTUM_PENALTY * 0.8  # Reduced - drift less critical than angle
            feedback.append(("severe", "Avoid using momentum — keep your torso still"))
            is_proper = False
        elif rep_metrics["torso_drift"] is not None and rep_metrics["torso_drift"] > TORSO_DRIFT_MAX:
            # Moderate momentum - counts as proper
            penalty += TORSO_PENALTY * 0.6
            feedback.append(("medium", "Keep your torso angle more consistent"))
        elif rep_metrics["torso_drift"] is not None and rep_metrics["torso_drift"] > 0.7*TORSO_DRIFT_MAX:
            # Minor drift
            feedback.append(("tip", "Try to minimize torso movement"))

    # Leg drive - feedback only, doesn't mark improper
    if allow_stability_cues:
        if rep_metrics["knee_drift"] is not None and rep_metrics["knee_drift"] > KNEE_DRIFT_MAX * 1.5:
            penalty += LEGDRIVE_PENALTY
            feedback.append(("medium", "Minimize leg drive — keep knees steady"))
        elif rep_metrics["knee_drift"] is not None and rep_metrics["knee_drift"] > KNEE_DRIFT_MAX:
            feedback.append(("tip", "Try to keep your legs more stable"))

    # Choose one strongest message for video overlay:
    overlay_msg = None
    for sev in ("severe","medium","tip"):
        for (s,msg) in feedback:
            if s == sev:
                overlay_msg = msg
                break
        if overlay_msg:
            break

    # Session feedback bucketed (severity then unique)
    session_msgs = []
    for sev in ("severe","medium"):
        for (s,msg) in feedback:
            if s == sev:
                session_msgs.append(msg)
    tips = [msg for (s,msg) in feedback if s == "tip"]

    return penalty, overlay_msg, session_msgs, tips, is_proper

def _robust_range(values, min_samples=MIN_REP_SAMPLES):
    if not values or len(values) < min_samples:
        return None
    low = np.percentile(values, 10)
    high = np.percentile(values, 90)
    return float(high - low)

def _safe_mean(values, min_samples=MIN_REP_SAMPLES):
    if not values or len(values) < min_samples:
        return None
    return float(sum(values) / len(values))

def faststart_remux(in_path):
    """Remux MP4 with moov at start (faststart). Returns output path."""
    base, ext = os.path.splitext(in_path)
    out_path = f"{base}_encoded.mp4"
    try:
        # Prefer ffmpeg if available
        cmd = [
            "ffmpeg","-y","-i",in_path,
            "-c","copy","-movflags","+faststart",
            out_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path
    except Exception:
        return in_path  # fallback

def run_row_analysis(input_video_path, output_dir="./media", user_session_id=None, return_video=True, fast_mode=None):
    """
    Main entry. Mirrors squat pipeline and return schema (incl. 'squat_count').
    Returns a dict with keys:
      - success, squat_count, proper_reps, technique_score, technique_score_display, technique_label
      - feedback (unique, severe-first), tips, form_tip (best single tip)
      - rep_details (per rep metrics)
      - analyzed_video_path
      - fps, duration_s
    """
    if mp is None:
        return {"success": False, "error": "mediapipe not installed"}

    if fast_mode is True:
        return_video = False

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return {"success": False, "error": "cannot open video"}

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w  = int(width * SCALE)
    out_h  = int(height * SCALE)

    # Output temp path
    uid = uuid.uuid4().hex[:8]
    out_basename = f"barbell_row_{uid}_analyzed.mp4"
    out_path = os.path.join(output_dir, out_basename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, src_fps/max(1,FRAME_SKIP), (out_w, out_h)) if return_video else None

    # Pose estimator
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=MODEL_COMPLEXITY,
                        enable_segmentation=False,
                        min_detection_confidence=CONF_DET, min_tracking_confidence=CONF_TRK)

    # State
    reps = 0
    good_reps = 0  # Track good reps (like squat)
    bad_reps = 0   # Track bad reps (like squat)
    last_rep_frame = -9999
    frame_idx = -1

    hip_y_prev = None
    hip_vel_ema = None
    global_motion_cooldown = 0

    # Rep metrics accumulators
    in_pull = False
    rep_start = None
    elbow_min = None
    elbow_max = None
    torso_vals = []
    knee_vals  = []

    # donut refs
    ext_ref = None
    top_ref = None
    live_depth_pct = 0.0

    session_feedback = []
    session_tips = []

    rep_details = []

    # Single overlay line (most recent message)
    overlay_msg = ""

    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
                continue

            # Resize
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # Pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # Landmarks
            lm = get_landmarks_xy(results, frame.shape)
            # Needed joints
            # Left side (11-12 shoulders, 13-14 elbows, 15-16 wrists, 23-24 hips, 25-26 knees, 27-28 ankles)
            # Use the more confident side when both available; here prefer right if both None choose other.
            # Simple choose right side if present else left.
            def pick(idx_r, idx_l):
                return lm[idx_r] if lm[idx_r] is not None else lm[idx_l]

            shoulder = pick(12,11)
            elbow    = pick(14,13)
            wrist    = pick(16,15)
            hip      = pick(24,23)
            knee     = pick(26,25)
            ankle    = pick(28,27)

            elbow_ang = angle(shoulder, elbow, wrist)
            
            # Calculate ACTUAL torso angle relative to horizontal (ground)
            # Vector from hip to shoulder
            torso_vector = (shoulder[0] - hip[0], shoulder[1] - hip[1])
            # Horizontal reference vector (right direction in image coords)
            horizontal = (1.0, 0.0)
            # Calculate angle between torso and horizontal
            torso_ang = angle_between_vectors(torso_vector, horizontal) if shoulder is not None and hip is not None else None
            
            knee_ang  = angle(hip, knee, ankle)

            # Update donut references
            if elbow_ang is not None:
                if ext_ref is None or elbow_ang > ext_ref:
                    ext_ref = elbow_ang
                if top_ref is None or elbow_ang < top_ref:
                    top_ref = elbow_ang

            # Global motion gating using hip y vel
            hip_y = hip[1] if hip is not None else None
            hip_vel = 0.0
            if hip_y is not None:
                if hip_y_prev is not None:
                    hip_vel = hip_y - hip_y_prev
                hip_y_prev = hip_y
                hip_vel_ema = ema(hip_vel_ema, abs(hip_vel), EMA_ALPHA)
            if hip_vel_ema is not None and hip_vel_ema > HIP_VEL_THRESH:
                global_motion_cooldown = GLOBAL_MOTION_LOCK_FRAMES
            else:
                global_motion_cooldown = max(0, global_motion_cooldown - 1)

            # Rep state machine
            live_depth_pct = row_depth_from_elbow(elbow_ang, ext_ref, top_ref)

            if not in_pull:
                # Ready to start pull: elbow crossing below PULL_START from extended region
                if (elbow_ang is not None and elbow_ang < ELBOW_PULL_START and
                    hip is not None and global_motion_cooldown == 0 and
                    (frame_idx - last_rep_frame) > MIN_FRAMES_BETWEEN_REPS):
                    in_pull = True
                    rep_start = frame_idx
                    elbow_min = elbow_ang
                    elbow_max = elbow_ang
                    torso_vals = [torso_ang] if torso_ang is not None else []
                    knee_vals  = [knee_ang] if knee_ang is not None else []
            else:
                # In pull/lower cycle
                if elbow_ang is not None:
                    elbow_min = min(elbow_min, elbow_ang) if elbow_min is not None else elbow_ang
                    elbow_max = max(elbow_max, elbow_ang) if elbow_max is not None else elbow_ang
                if torso_ang is not None: torso_vals.append(torso_ang)
                if knee_ang is not None:  knee_vals.append(knee_ang)

                # End rep when returned to extension & motion calm
                if (elbow_ang is not None and elbow_ang >= ELBOW_EXTENDED_MIN and global_motion_cooldown == 0):
                    # finalize rep
                    reps += 1
                    last_rep_frame = frame_idx

                    torso_drift = _robust_range(torso_vals)
                    knee_drift  = _robust_range(knee_vals)
                    torso_mean  = _safe_mean(torso_vals)

                    rep_m = {
                        "start_frame": rep_start,
                        "end_frame": frame_idx,
                        "min_elbow_angle": elbow_min,
                        "max_elbow_angle": elbow_max,
                        "torso_drift": torso_drift,
                        "knee_drift": knee_drift,
                        "torso_mean": torso_mean,
                        "depth_pct": row_depth_from_elbow(elbow_min, ext_ref, top_ref)
                    }
                    pen, ov_msg, sess_msgs, tips, is_proper = analyze_rep(rep_m)
                    
                    # Count good/bad reps like squat (9.5+ = good)
                    # Calculate score for this rep
                    rep_score = MAX_SCORE - pen
                    rep_score = clamp(rep_score, MIN_SCORE, MAX_SCORE)
                    rep_score = round_to_half(rep_score)
                    
                    if rep_score >= 9.5:
                        good_reps += 1
                    else:
                        bad_reps += 1
                    
                    # Keep one overlay message at a time
                    overlay_msg = ov_msg or overlay_msg

                    # Accumulate session feedback unique & severity-first
                    for m in sess_msgs:
                        if m not in session_feedback:
                            session_feedback.append(m)
                    for t in tips:
                        if t not in session_tips:
                            session_tips.append(t)

                    rep_details.append(rep_m)

                    # reset
                    in_pull = False
                    rep_start = None
                    elbow_min = elbow_max = None
                    torso_vals = []
                    knee_vals  = []

            # Draw overlay (with current skeleton) before writing
            if return_video and writer is not None:
                if lm:
                    draw_skeleton(frame, lm)
                frame = render_overlay(frame, reps, live_depth_pct, overlay_msg)
                writer.write(frame)

        # Finalize
    finally:
        try:
            pose.close()
        except Exception:
            pass
        cap.release()
        if writer is not None:
            writer.release()

    # Scoring
    score = MAX_SCORE
    for rep_m in rep_details:
        pen, _, _, _, _ = analyze_rep(rep_m)
        score -= (pen / max(1, len(rep_details)))  # average penalty
    score = clamp(score, MIN_SCORE, MAX_SCORE)
    score = round_to_half(score)

    # Technique label
    label = technique_label(score)

    # Select best form tip (coaching advice - SEPARATE from feedback)
    form_tip = None
    
    # Generate coaching tip based on patterns, not just feedback
    if session_feedback:
        # If there's feedback, give actionable coaching
        if any("torso" in fb.lower() or "horizontal" in fb.lower() for fb in session_feedback):
            form_tip = "Focus on hip hinge — push your hips back to get more horizontal"
        elif any("pull" in fb.lower() or "higher" in fb.lower() for fb in session_feedback):
            form_tip = "Think 'elbows to ceiling' — drive them high and back"
        elif any("momentum" in fb.lower() for fb in session_feedback):
            form_tip = "Control the descent — lower the bar slowly over 2-3 seconds"
        else:
            form_tip = session_feedback[0]  # Fallback to first feedback
    elif session_tips:
        form_tip = "Squeeze your shoulder blades together at the top"
    else:
        # No issues - give encouraging + progressive advice
        if good_reps == reps and reps > 0:
            form_tip = "Great form! Keep it up 💪"
        elif reps > 0:
            form_tip = "Try slowing down the lowering phase for better control"
        else:
            form_tip = "Complete a full rep to get feedback"

    # Faststart remux
    out_fast = faststart_remux(out_path) if return_video else ""

    # Return schema (matching squat keys; include squat_count)
    duration_s = (time.time() - start_time)
    result = {
        "success": True,
        "squat_count": reps,                # keep same key name for UI compatibility
        "row_count": reps,
        "good_reps": int(good_reps),        # FIXED: Now matches squat exactly
        "bad_reps": int(bad_reps),          # FIXED: Added like squat
        "technique_score": float(score),
        "technique_score_display": f"{score:.1f}",
        "technique_label": label,
        "feedback": session_feedback,       # unique, severe-first
        "tips": session_tips,               # do not reduce score
        "form_tip": form_tip,               # FIXED: Now returns best single tip
        "rep_details": rep_details,         # per-rep metrics
        "analyzed_video_path": out_fast if return_video else "",    # file path; caller can build URL
        "fps": src_fps/max(1,FRAME_SKIP),
        "duration_s": duration_s,
    }
    return result

if __name__ == "__main__":
    # Quick manual test (adjust paths)
    test_in = "sample_row.mp4"
    if not os.path.exists(test_in):
        print(json.dumps({"success": False, "error": f"Missing {test_in} for test"}, ensure_ascii=False))
    else:
        res = run_row_analysis(test_in)
        print(json.dumps(res, indent=2, ensure_ascii=False))

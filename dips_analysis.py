# -*- coding: utf-8 -*-
# dips_analysis.py — complete dips counter with form feedback
# v2 — fixed video rotation + improved rep detection

import os, cv2, math, numpy as np, subprocess, json
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ============ Styles ============
BAR_BG_ALPHA=0.55; DONUT_RADIUS_SCALE=0.72; DONUT_THICKNESS_FRAC=0.28
DEPTH_COLOR=(40,200,80); DEPTH_RING_BG=(70,70,70)
FONT_PATH="Roboto-VariableFont_wdth,wght.ttf"
_REF_H = 480.0
_REF_REPS_FONT_SIZE = 28
_REF_FEEDBACK_FONT_SIZE = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE = 18

def _load_font(p,s):
    try: return ImageFont.truetype(p,s)
    except: return ImageFont.load_default()

def _scaled_font_size(ref_size, canvas_h):
    return max(10, int(round(ref_size * (canvas_h / _REF_H))))

# ============ MediaPipe ============
try:
    import mediapipe as mp
    mp_pose=mp.solutions.pose
except Exception:
    mp_pose=None

# ============ Helpers ============
def _ang(a,b,c):
    ba=np.array([a[0]-b[0],a[1]-b[1]]); bc=np.array([c[0]-b[0],c[1]-b[1]])
    den=(np.linalg.norm(ba)*np.linalg.norm(bc))+1e-9
    cos=float(np.clip(np.dot(ba,bc)/den,-1,1)); return float(np.degrees(np.arccos(cos)))

def _ema(prev,new,a): return float(new) if prev is None else (a*float(new)+(1-a)*float(prev))
def _half_floor10(x): return max(0.0,min(10.0, math.floor(x*2.0)/2.0))
def display_half_str(x): q=round(float(x)*2)/2.0; return str(int(round(q))) if abs(q-round(q))<1e-9 else f"{q:.1f}"
def score_label(s):
    s=float(s)
    if s>=9.0: return "Excellent"
    if s>=7.0: return "Good"
    if s>=5.5: return "Fair"
    return "Needs work"

def _wrap_two_lines(draw, text, font, max_width):
    words=text.split()
    if not words: return [""]
    lines=[]; cur=""
    for w in words:
        t=(cur+" "+w).strip()
        if draw.textlength(t, font=font)<=max_width: cur=t
        else:
            if cur: lines.append(cur)
            cur=w
        if len(lines)==2: break
    if cur and len(lines)<2: lines.append(cur)
    if len(lines)>=2 and draw.textlength(lines[-1], font=font)>max_width:
        last=lines[-1]+"…"
        while draw.textlength(last, font=font)>max_width and len(last)>1:
            last=last[:-2]+"…"
        lines[-1]=last
    return lines

def _dyn_thickness(h):
    return max(2,int(round(h*0.002))), max(3,int(round(h*0.004)))

def _dist_xy(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ============ Video Rotation Detection & Fix ============
def _get_video_rotation(video_path):
    """Detect rotation metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate:stream=width,height",
            "-show_entries", "format_tags=rotate",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)

        rotation = 0

        # Check stream tags
        streams = data.get("streams", [])
        if streams:
            tags = streams[0].get("tags", {})
            if "rotate" in tags:
                rotation = int(tags["rotate"])

        # Check format tags as fallback
        if rotation == 0:
            fmt_tags = data.get("format", {}).get("tags", {})
            if "rotate" in fmt_tags:
                rotation = int(fmt_tags["rotate"])

        # Also check side_data for displaymatrix rotation (newer ffmpeg)
        if rotation == 0:
            cmd2 = [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream_side_data_list",
                "-of", "json",
                video_path
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=10)
            data2 = json.loads(result2.stdout)
            streams2 = data2.get("streams", [])
            if streams2:
                side_data = streams2[0].get("side_data_list", [])
                for sd in side_data:
                    if "rotation" in sd:
                        rotation = int(sd["rotation"])
                        break

        return rotation
    except Exception as e:
        print(f"[ROTATION] ffprobe failed: {e}")
        return 0


def _pre_rotate_video(video_path):
    """
    Pre-process video with ffmpeg to apply rotation metadata,
    producing a correctly-oriented video that OpenCV can read directly.
    Returns path to the rotated video (or original if no rotation needed).
    """
    rotation = _get_video_rotation(video_path)
    print(f"[ROTATION] Detected rotation metadata: {rotation}°")

    # Only pre-rotate if there's actual rotation metadata
    if rotation == 0:
        return video_path

    # Write rotated file next to the original or in /tmp
    base, ext = os.path.splitext(os.path.basename(video_path))
    # Try same directory first, fall back to /tmp
    parent_dir = os.path.dirname(video_path)
    rotated_path = os.path.join(parent_dir, base + "_rotated" + ext)
    try:
        # Test if we can write to parent dir
        test_file = os.path.join(parent_dir, ".write_test")
        with open(test_file, "w") as f: f.write("t")
        os.remove(test_file)
    except (OSError, PermissionError):
        rotated_path = os.path.join("/tmp", base + "_rotated" + ext)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",  # drop audio for speed
            "-movflags", "+faststart",
            rotated_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)

        if os.path.exists(rotated_path) and os.path.getsize(rotated_path) > 0:
            print(f"[ROTATION] Pre-rotated video created: {rotated_path}")
            return rotated_path
        else:
            print("[ROTATION] Pre-rotation produced empty file, using original")
            return video_path
    except Exception as e:
        print(f"[ROTATION] ffmpeg pre-rotation failed: {e}, using original")
        return video_path


def _rotate_frame_cv2(frame, rotation):
    """Rotate a frame in OpenCV based on rotation angle."""
    if rotation == 90 or rotation == -270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270 or rotation == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ============ Body-only skeleton ============
_FACE_LMS=set(); _BODY_CONNECTIONS=tuple(); _BODY_POINTS=tuple()
if mp_pose:
    _FACE_LMS={
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
        mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
    }
    _BODY_CONNECTIONS=tuple((a,b) for (a,b) in mp_pose.POSE_CONNECTIONS if a not in _FACE_LMS and b not in _FACE_LMS)
    _BODY_POINTS=tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

def draw_body_only(frame, lms, color=(255,255,255)):
    h,w=frame.shape[:2]; line, dot=_dyn_thickness(h)
    for a,b in _BODY_CONNECTIONS:
        pa, pb=lms[a], lms[b]
        ax,ay=int(pa.x*w), int(pa.y*h); bx,by=int(pb.x*w), int(pb.y*h)
        cv2.line(frame,(ax,ay),(bx,by),color,line,cv2.LINE_AA)
    for i in _BODY_POINTS:
        p=lms[i]; x,y=int(p.x*w),int(p.y*h)
        cv2.circle(frame,(x,y),dot,color,-1,cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    h, w, _ = frame.shape
    HD_H = 1080
    hd_scale = HD_H / float(h)
    HD_W = max(1, int(round(w * hd_scale)))

    reps_font_size = _scaled_font_size(_REF_REPS_FONT_SIZE, HD_H)
    feedback_font_size = _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, HD_H)
    depth_label_font_size = _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, HD_H)
    depth_pct_font_size = _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, HD_H)

    _REPS_FONT = _load_font(FONT_PATH, reps_font_size)
    _FEEDBACK_FONT = _load_font(FONT_PATH, feedback_font_size)
    _DEPTH_LABEL_FONT = _load_font(FONT_PATH, depth_label_font_size)
    _DEPTH_PCT_FONT = _load_font(FONT_PATH, depth_pct_font_size)

    pct = float(np.clip(depth_pct, 0, 1))
    bg_alpha_val = int(round(255 * BAR_BG_ALPHA))

    ref_h = max(int(HD_H * 0.06), int(reps_font_size * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = int(12 * hd_scale)
    cx = HD_W - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)

    overlay_np = np.zeros((HD_H, HD_W, 4), dtype=np.uint8)

    pad_x, pad_y = int(10 * hd_scale), int(6 * hd_scale)
    tmp_pil = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_pil)
    txt = f"Reps: {int(reps)}"
    tw = tmp_draw.textlength(txt, font=_REPS_FONT)
    thh = _REPS_FONT.size
    box_w = int(tw + 2 * pad_x)
    box_h = int(thh + 2 * pad_y)
    cv2.rectangle(overlay_np, (0, 0), (box_w, box_h), (0, 0, 0, bg_alpha_val), -1)

    cv2.circle(overlay_np, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)
    start_ang = -90
    end_ang = start_ang + int(360 * pct)
    if end_ang != start_ang:
        cv2.ellipse(overlay_np, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                    (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)

    fb_y0 = 0
    fb_lines = []
    fb_pad_x = fb_pad_y = line_gap = line_h = 0
    if feedback:
        safe_margin = max(int(6 * hd_scale), int(HD_H * 0.02))
        fb_pad_x, fb_pad_y, line_gap = int(12 * hd_scale), int(8 * hd_scale), int(4 * hd_scale)
        max_text_w = int(HD_W - 2 * fb_pad_x - int(20 * hd_scale))
        fb_lines = _wrap_two_lines(tmp_draw, feedback, _FEEDBACK_FONT, max_text_w)
        line_h = _FEEDBACK_FONT.size + int(6 * hd_scale)
        block_h = 2 * fb_pad_y + len(fb_lines) * line_h + (len(fb_lines) - 1) * line_gap
        fb_y0 = max(0, HD_H - safe_margin - block_h)
        y1 = HD_H - safe_margin
        cv2.rectangle(overlay_np, (0, fb_y0), (HD_W, y1), (0, 0, 0, bg_alpha_val), -1)

    overlay_pil = Image.fromarray(overlay_np, mode="RGBA")
    draw = ImageDraw.Draw(overlay_pil)

    draw.text((pad_x, pad_y - 1), txt, font=_REPS_FONT, fill=(255, 255, 255, 255))

    gap = max(2, int(radius * 0.10))
    by = cy - (_DEPTH_LABEL_FONT.size + gap + _DEPTH_PCT_FONT.size) // 2
    label = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label, font=_DEPTH_LABEL_FONT)
    pw = draw.textlength(pct_txt, font=_DEPTH_PCT_FONT)
    draw.text((cx - int(lw // 2), by), label, font=_DEPTH_LABEL_FONT, fill=(255, 255, 255, 255))
    draw.text((cx - int(pw // 2), by + _DEPTH_LABEL_FONT.size + gap), pct_txt, font=_DEPTH_PCT_FONT, fill=(255, 255, 255, 255))

    if feedback and fb_lines:
        ty = fb_y0 + fb_pad_y
        for ln in fb_lines:
            tw2 = draw.textlength(ln, font=_FEEDBACK_FONT)
            tx = max(fb_pad_x, (HD_W - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=_FEEDBACK_FONT, fill=(255, 255, 255, 255))
            ty += line_h + line_gap

    overlay_rgba = np.array(overlay_pil)
    overlay_small = cv2.resize(overlay_rgba, (w, h), interpolation=cv2.INTER_AREA)
    alpha = overlay_small[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr_ch = overlay_small[:, :, [2, 1, 0]].astype(np.float32)
    frame_f = frame.astype(np.float32)
    result = frame_f * (1.0 - alpha) + overlay_bgr_ch * alpha
    return result.astype(np.uint8)

# ============ Dips Parameters ============
# Rep counting — RELAXED thresholds for better detection
ELBOW_BENT_ANGLE = 110.0          # was 90° — too strict; 110° catches more real dips
ELBOW_BENT_ANGLE_RAW_MARGIN = 15.0  # raw elbow can be up to this much above threshold
SHOULDER_MIN_DESCENT = 0.025      # was 0.045 — reduced for various camera angles
RESET_ASCENT = 0.015              # was 0.025 — reduced for smoother detection
RESET_ELBOW = 110.0               # was 150° → 140° → 110°; real data shows top elbow is 108-124°
REFRACTORY_FRAMES = 3             # was 4

# EMA smoothing — faster response
ELBOW_EMA_ALPHA = 0.55            # was 0.45 — faster tracking
SHOULDER_EMA_ALPHA = 0.50         # was 0.38 — faster tracking

# Minimum descent duration before allowing a count
MIN_DESC_FRAMES = 2

# Rearm requirements
REARM_ASCENT_RATIO = 0.30         # was 0.40 — more lenient
REARM_ELBOW_ABOVE = 105.0         # was 130° → 120° → 105°; real data shows top elbow is 108-124°

# Velocity smoothing window (frames)
VEL_WINDOW = 3

# Dips position detection — RELAXED
VIS_THR_STRICT = 0.20             # was 0.30 — accept lower visibility
VIS_THR_LOOSE = 0.10              # new: very loose threshold for fallback
WRIST_BELOW_SHOULDER_MARGIN = -0.01  # was 0.02 — allow wrists near shoulder level
TORSO_STABILITY_THR = 0.025       # was 0.012 — allow more torso movement
ONDIPS_MIN_FRAMES = 2             # was 3 — enter dips state faster
OFFDIPS_MIN_FRAMES = 8            # was 6 — stay in dips state longer
AUTO_STOP_AFTER_EXIT_SEC = 2.0    # was 1.2 — wait longer before stopping
TAIL_NOPOSE_STOP_SEC = 1.5        # was 1.0 — wait longer for no-pose

# Stabilization: prevent counting mount/dismount as reps
STABLE_BASELINE_FRAMES = 6        # must be stable for N frames before counting starts
STABLE_SHOULDER_VARIANCE = 0.015  # max shoulder Y variance during stabilization
MAX_SINGLE_DESCENT = 0.12         # reject descents larger than this (mount/dismount)
                                  # real dips: ~0.04-0.06, mounting: ~0.20

# Alternative position detection: elbow angle range indicating dips motion
# Even if wrists aren't perfectly below shoulders, if elbows are bending
# in the right range, we're probably doing dips
DIPS_ELBOW_RANGE_LOW = 60.0
DIPS_ELBOW_RANGE_HIGH = 170.0

# ============ Feedback Cues & Weights ============
FB_CUE_DEEPER = "Go deeper (elbows to 90°)"
FB_CUE_LEAN = "Reduce forward lean (keep torso upright)"
FB_CUE_LOCKOUT = "Fully lockout elbows at top"
FB_CUE_ELBOWS_IN = "Keep elbows closer to body"

FB_W_DEEPER = float(os.getenv("FB_W_DEEPER", "1.0"))
FB_W_LEAN = float(os.getenv("FB_W_LEAN", "0.7"))
FB_W_LOCKOUT = float(os.getenv("FB_W_LOCKOUT", "0.8"))
FB_W_ELBOWS_IN = float(os.getenv("FB_W_ELBOWS_IN", "0.6"))

FB_WEIGHTS = {
    FB_CUE_DEEPER: FB_W_DEEPER,
    FB_CUE_LEAN: FB_W_LEAN,
    FB_CUE_LOCKOUT: FB_W_LOCKOUT,
    FB_CUE_ELBOWS_IN: FB_W_ELBOWS_IN,
}
FB_DEFAULT_WEIGHT = 0.5
PENALTY_MIN_IF_ANY = 0.5
FORM_TIP_PRIORITY = [FB_CUE_DEEPER, FB_CUE_LOCKOUT, FB_CUE_ELBOWS_IN, FB_CUE_LEAN]

# ============ Form Detection Thresholds ============
DEPTH_MIN_ANGLE = 100.0           # was 95 — relaxed
DEPTH_NEAR_DEG = 12.0             # was 8 — wider near zone
TORSO_MAX_LEAN = 30.0             # was 25 — allow more lean
LEAN_NEAR_DEG = 8.0               # was 5
LOCKOUT_MIN_ANGLE = 140.0         # was 165° → 160° → 140°; camera angle makes full lockout look less
LOCKOUT_NEAR_DEG = 12.0           # was 8
ELBOW_FLARE_MAX = 50.0            # was 45 — more lenient
FLARE_NEAR_DEG = 10.0             # was 8

DEPTH_FAIL_MIN_REPS = 2
LEAN_FAIL_MIN_REPS = 2
LOCKOUT_FAIL_MIN_REPS = 2
FLARE_FAIL_MIN_REPS = 2

# ============ Micro-burst ============
BURST_FRAMES = int(os.getenv("BURST_FRAMES", "2"))
INFLECT_VEL_THR = float(os.getenv("INFLECT_VEL_THR", "0.0006"))

DEBUG_ONDIPS = bool(int(os.getenv("DEBUG_ONDIPS", "0")))
DEBUG_REPS = bool(int(os.getenv("DEBUG_REPS", "0")))

def run_dips_analysis(video_path,
                      frame_skip=3,
                      scale=0.4,
                      output_path="dips_analyzed.mp4",
                      feedback_path="dips_feedback.txt",
                      preserve_quality=False,
                      encode_crf=None,
                      return_video=True,
                      fast_mode=None):
    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    model_complexity = 0 if fast_mode else 1
    if fast_mode:
        return_video = False
        scale = min(scale, 0.35)

    if preserve_quality:
        scale=1.0; frame_skip=1; encode_crf=18 if encode_crf is None else encode_crf
    else:
        encode_crf=23 if encode_crf is None else encode_crf

    # ====== FIX #1: Handle video rotation ======
    # Pre-process with ffmpeg to apply rotation metadata
    rotated_video = _pre_rotate_video(video_path)
    use_path = rotated_video
    cleanup_rotated = (rotated_video != video_path)

    cap=cv2.VideoCapture(use_path)
    if not cap.isOpened():
        # Fallback: try original path
        if cleanup_rotated:
            cap = cv2.VideoCapture(video_path)
            cleanup_rotated = False
        if not cap.isOpened():
            return _ret_err("Could not open video", feedback_path)

    fps_in=cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    effective_fps=max(1.0, fps_in/max(1,frame_skip))
    sec_to_frames=lambda s: max(1,int(s*effective_fps))

    print(f"[DIPS] Video: {use_path}, FPS: {fps_in:.1f}, Total frames: {total_frames}, "
          f"Scale: {scale}, Frame skip: {frame_skip}")

    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=None; frame_idx=0

    # Counters
    rep_count=0; good_reps=0; bad_reps=0; rep_reports=[]; all_scores=[]

    # Landmarks
    LSH=mp_pose.PoseLandmark.LEFT_SHOULDER.value;  RSH=mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LE =mp_pose.PoseLandmark.LEFT_ELBOW.value;     RE =mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW =mp_pose.PoseLandmark.LEFT_WRIST.value;     RW =mp_pose.PoseLandmark.RIGHT_WRIST.value
    LH =mp_pose.PoseLandmark.LEFT_HIP.value;       RH =mp_pose.PoseLandmark.RIGHT_HIP.value

    def _pick_side_dyn(lms):
        vL=lms[LSH].visibility+lms[LE].visibility+lms[LW].visibility
        vR=lms[RSH].visibility+lms[RE].visibility+lms[RW].visibility
        return ("LEFT",LSH,LE,LW) if vL>=vR else ("RIGHT",RSH,RE,RW)

    # State
    elbow_ema=None; shoulder_ema=None; shoulder_prev=None; shoulder_vel_prev=None
    baseline_shoulder_y=None
    desc_base_shoulder=None; allow_new_bottom=True; last_bottom_frame=-10**9
    cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
    cycle_min_elbow_raw=999.0
    desc_frame_count=0
    cycle_bottom_shoulder=None
    reached_bottom=False
    vel_buf=deque(maxlen=VEL_WINDOW)

    ondips=False; ondips_streak=0; offdips_streak=0; prev_torso_cx=None
    offdips_frames_since_any_rep=0; nopose_frames_since_any_rep=0

    # Stabilization: prevent counting mount/dismount
    stabilized = False              # True once baseline is established
    stable_shoulder_buf = deque(maxlen=STABLE_BASELINE_FRAMES)
    ondips_frames_count = 0         # how many frames in current ondips session

    # Adaptive baseline: track shoulder Y over time to handle drift
    shoulder_y_history = deque(maxlen=30)
    baseline_update_interval = 15  # update baseline every N processed frames
    frames_since_baseline_update = 0

    # Feedback state
    session_feedback=set()
    rt_fb_msg=None; rt_fb_hold=0

    # Per-cycle trackers
    cycle_tip_deeper=False; cycle_tip_lean=False; cycle_tip_lockout=False; cycle_tip_elbows=False
    depth_fail_count=0; lean_fail_count=0; lockout_fail_count=0; flare_fail_count=0
    depth_already_reported=False; lean_already_reported=False
    lockout_already_reported=False; flare_already_reported=False

    # Phase trackers
    bottom_phase_min_elbow=None
    top_phase_max_elbow=None
    cycle_max_lean=None
    cycle_max_flare=None

    OFFDIPS_STOP_FRAMES=sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP_FRAMES=sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RT_FB_HOLD_FRAMES=sec_to_frames(0.8)

    REARM_ASCENT_EFF=max(RESET_ASCENT*0.60, 0.008)

    # Micro-burst
    burst_cntr=0

    # Debug counters
    pose_detected_count = 0
    ondips_frame_count = 0

    with mp_pose.Pose(model_complexity=model_complexity, min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            process_now = (burst_cntr > 0) or (frame_idx % max(1, frame_skip) == 0)
            if not process_now:
                continue
            if burst_cntr > 0:
                burst_cntr -= 1

            if scale != 1.0:
                frame=cv2.resize(frame,(0,0),fx=scale,fy=scale)
            h,w=frame.shape[:2]

            if return_video and out is None:
                out=cv2.VideoWriter(output_path,fourcc,effective_fps,(w,h))

            res=pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            depth_live=0.0

            if not res.pose_landmarks:
                nopose_frames_since_any_rep = (nopose_frames_since_any_rep+1) if rep_count>0 else 0
                if rep_count>0 and nopose_frames_since_any_rep>=NOPOSE_STOP_FRAMES: break
                if return_video and out is not None:
                    out.write(draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),depth_pct=0.0))
                if rt_fb_hold>0: rt_fb_hold-=1
                continue

            nopose_frames_since_any_rep=0
            pose_detected_count += 1
            lms=res.pose_landmarks.landmark
            side,S,E,W=_pick_side_dyn(lms)

            min_vis=min(lms[S].visibility,lms[E].visibility,lms[W].visibility)
            # Use loose threshold — strict was filtering too many frames
            vis_strict_ok=(min_vis>=VIS_THR_LOOSE)
            vis_good=(min_vis>=VIS_THR_STRICT)

            # Use average of both shoulders for more stable Y tracking
            shoulder_raw=float((lms[LSH].y + lms[RSH].y) / 2.0)
            raw_elbow_L=_ang((lms[LSH].x,lms[LSH].y),(lms[LE].x,lms[LE].y),(lms[LW].x,lms[LW].y))
            raw_elbow_R=_ang((lms[RSH].x,lms[RSH].y),(lms[RE].x,lms[RE].y),(lms[RW].x,lms[RW].y))
            raw_elbow=raw_elbow_L if side=="LEFT" else raw_elbow_R

            # Visibility-weighted elbow: avoid noise from low-visibility side
            vis_L_arm = min(lms[LSH].visibility, lms[LE].visibility, lms[LW].visibility)
            vis_R_arm = min(lms[RSH].visibility, lms[RE].visibility, lms[RW].visibility)

            # If one side has much better visibility, trust that side more
            VIS_RATIO_THR = 1.5  # if one side is 1.5x more visible, prefer it
            if vis_L_arm > vis_R_arm * VIS_RATIO_THR:
                # Trust left side
                raw_elbow_primary = raw_elbow_L
                raw_elbow_min = raw_elbow_L
                raw_elbow_max = raw_elbow_L
            elif vis_R_arm > vis_L_arm * VIS_RATIO_THR:
                # Trust right side
                raw_elbow_primary = raw_elbow_R
                raw_elbow_min = raw_elbow_R
                raw_elbow_max = raw_elbow_R
            else:
                # Both sides roughly equal visibility — use both
                raw_elbow_primary = raw_elbow
                raw_elbow_min = min(raw_elbow_L, raw_elbow_R)
                raw_elbow_max = max(raw_elbow_L, raw_elbow_R)

            # Sanity check: elbow angle below 30° is almost certainly noise
            if raw_elbow_min < 30.0:
                # Use the other side or primary
                raw_elbow_min = raw_elbow_primary
            if raw_elbow_max < 30.0:
                raw_elbow_max = raw_elbow_primary

            elbow_ema=_ema(elbow_ema,raw_elbow,ELBOW_EMA_ALPHA)
            shoulder_ema=_ema(shoulder_ema,shoulder_raw,SHOULDER_EMA_ALPHA)
            shoulder_y=shoulder_ema; elbow_angle=elbow_ema

            # Adaptive baseline
            shoulder_y_history.append(shoulder_y)
            frames_since_baseline_update += 1

            if baseline_shoulder_y is None:
                baseline_shoulder_y=shoulder_y
            elif not ondips and frames_since_baseline_update >= baseline_update_interval:
                # Update baseline to recent minimum (top position) when not in dips
                recent_min = min(shoulder_y_history)
                baseline_shoulder_y = min(baseline_shoulder_y, recent_min) * 0.3 + recent_min * 0.7
                frames_since_baseline_update = 0

            depth_live=float(np.clip((shoulder_y-baseline_shoulder_y)/max(0.06,SHOULDER_MIN_DESCENT*2.0),0.0,1.0))

            # Torso stability
            torso_cx=np.mean([lms[LSH].x,lms[RSH].x,lms[LH].x,lms[RH].x])*w
            torso_dx_norm=0.0 if prev_torso_cx is None else abs(torso_cx-prev_torso_cx)/max(1.0,w)
            prev_torso_cx=torso_cx

            # ====== FIX #2: Improved dips position detection ======
            # Method 1: Wrist below shoulder (original, relaxed)
            lw_below=(lms[LW].visibility>=VIS_THR_LOOSE) and (lms[LW].y>lms[LSH].y+WRIST_BELOW_SHOULDER_MARGIN)
            rw_below=(lms[RW].visibility>=VIS_THR_LOOSE) and (lms[RW].y>lms[RSH].y+WRIST_BELOW_SHOULDER_MARGIN)
            wrist_position_ok=(lw_below or rw_below)

            # Method 2: Elbow angle in dips range (catching movement even if wrist detection is off)
            elbow_in_range = (DIPS_ELBOW_RANGE_LOW <= raw_elbow_min <= DIPS_ELBOW_RANGE_HIGH)

            # Method 3: Hands near/below shoulder with some tolerance for camera angle
            # Check if ANY arm is in a dips-like configuration
            hip_vis_ok = (lms[LH].visibility >= VIS_THR_LOOSE or lms[RH].visibility >= VIS_THR_LOOSE)
            arms_engaged = elbow_in_range and hip_vis_ok

            # Combined: either traditional wrist check OR arms-engaged heuristic
            in_position = wrist_position_ok or arms_engaged

            # Relaxed torso stability check
            torso_ok = (torso_dx_norm <= TORSO_STABILITY_THR)

            if vis_strict_ok and in_position and torso_ok:
                ondips_streak+=1; offdips_streak=0
            elif vis_strict_ok and in_position:
                # Even without torso stability, count partial streak
                ondips_streak+=1; offdips_streak=0
            else:
                offdips_streak+=1; ondips_streak=0

            if DEBUG_ONDIPS and frame_idx % 5 == 0:
                print(f"[DBG] f={frame_idx} ondips={ondips} vis={min_vis:.2f} wrist_ok={wrist_position_ok} "
                      f"arms_eng={arms_engaged} elbow_raw={raw_elbow_min:.1f} torso_dx={torso_dx_norm:.4f} "
                      f"sh_y={shoulder_y:.3f} baseline={baseline_shoulder_y:.3f}")

            # Enter dips position
            if (not ondips) and ondips_streak>=ONDIPS_MIN_FRAMES:
                ondips=True
                desc_base_shoulder=None; allow_new_bottom=True
                cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_min_elbow_raw=999.0; desc_frame_count=0; cycle_bottom_shoulder=None
                reached_bottom=False
                vel_buf.clear()
                cycle_tip_deeper=False; cycle_tip_lean=False; cycle_tip_lockout=False; cycle_tip_elbows=False
                bottom_phase_min_elbow=None
                top_phase_max_elbow=None
                cycle_max_lean=None
                cycle_max_flare=None
                # Stabilization: don't set baseline yet — wait for stable frames
                stabilized = False
                stable_shoulder_buf.clear()
                ondips_frames_count = 0
                print(f"[DIPS] Entered dips position at frame {frame_idx}")

            # Exit dips position
            if ondips and offdips_streak>=OFFDIPS_MIN_FRAMES:
                # Don't count incomplete reps on exit — only fully ascended reps count

                ondips=False; offdips_frames_since_any_rep=0
                desc_base_shoulder=None; cycle_max_descent=0.0; cycle_min_elbow=999.0; counted_this_cycle=False
                cycle_min_elbow_raw=999.0; desc_frame_count=0; cycle_bottom_shoulder=None
                reached_bottom=False
                vel_buf.clear()
                cycle_tip_deeper=False; cycle_tip_lean=False; cycle_tip_lockout=False; cycle_tip_elbows=False
                bottom_phase_min_elbow=None
                top_phase_max_elbow=None
                cycle_max_lean=None
                cycle_max_flare=None
                print(f"[DIPS] Exited dips position at frame {frame_idx}")

            if ondips:
                ondips_frame_count += 1

            if (not ondips) and rep_count>0:
                offdips_frames_since_any_rep+=1
                if offdips_frames_since_any_rep>=OFFDIPS_STOP_FRAMES: break

            raw_vel=0.0 if shoulder_prev is None else (shoulder_y-shoulder_prev)
            vel_buf.append(raw_vel)
            shoulder_vel=sum(vel_buf)/len(vel_buf) if vel_buf else raw_vel
            cur_rt=None

            # Micro-burst near inflection
            if ondips and (desc_base_shoulder is not None):
                near_inflect = (abs(shoulder_vel) <= INFLECT_VEL_THR)
                sign_flip = (shoulder_vel_prev is not None) and ((shoulder_vel_prev < 0 and shoulder_vel >= 0) or (shoulder_vel_prev > 0 and shoulder_vel <= 0))
                if near_inflect or sign_flip:
                    burst_cntr = max(burst_cntr, BURST_FRAMES)
            shoulder_vel_prev = shoulder_vel

            if ondips and vis_strict_ok:
                ondips_frames_count += 1

                # STABILIZATION PHASE: wait for stable position before counting
                stable_shoulder_buf.append(shoulder_y)
                if not stabilized:
                    if len(stable_shoulder_buf) >= STABLE_BASELINE_FRAMES:
                        sh_var = max(stable_shoulder_buf) - min(stable_shoulder_buf)
                        if sh_var <= STABLE_SHOULDER_VARIANCE:
                            stabilized = True
                            baseline_shoulder_y = sum(stable_shoulder_buf) / len(stable_shoulder_buf)
                            print(f"[DIPS] Stabilized at frame {frame_idx}, baseline_sh={baseline_shoulder_y:.4f}")
                        else:
                            # Not stable yet — keep waiting, don't count anything
                            pass

                # REP COUNTING — count on ASCENT only (after descent+bottom+rise)
                at_bottom = False  # initialize before any branch

                if not stabilized:
                    # Not stabilized yet — skip counting entirely
                    pass
                elif desc_base_shoulder is None:
                    # Start descent detection: either velocity-based OR elbow starting to bend
                    descent_starting = (shoulder_vel > abs(INFLECT_VEL_THR)) or \
                                      (raw_elbow_min < 150 and shoulder_y > baseline_shoulder_y)
                    if descent_starting:
                        desc_base_shoulder=shoulder_y
                        cycle_max_descent=0.0; cycle_min_elbow=elbow_angle; counted_this_cycle=False
                        cycle_min_elbow_raw=raw_elbow_min; desc_frame_count=1; cycle_bottom_shoulder=None
                        bottom_phase_min_elbow=raw_elbow_min
                        top_phase_max_elbow=None
                        cycle_max_lean=None
                        cycle_max_flare=None
                        reached_bottom=False  # NEW: track if we've been at bottom
                else:
                    desc_frame_count+=1
                    cycle_max_descent=max(cycle_max_descent,(shoulder_y-desc_base_shoulder))
                    cycle_min_elbow=min(cycle_min_elbow,elbow_angle)
                    cycle_min_elbow_raw=min(cycle_min_elbow_raw,raw_elbow_min)

                    # Track bottom phase
                    if bottom_phase_min_elbow is None: bottom_phase_min_elbow = raw_elbow_min
                    else: bottom_phase_min_elbow = min(bottom_phase_min_elbow, raw_elbow_min)

                    # Track top phase
                    if top_phase_max_elbow is None: top_phase_max_elbow = raw_elbow_max
                    else: top_phase_max_elbow = max(top_phase_max_elbow, raw_elbow_max)

                    # Track lean
                    torso_angle = _calculate_torso_lean(lms, LSH, RSH, LH, RH)
                    if cycle_max_lean is None: cycle_max_lean = torso_angle
                    else: cycle_max_lean = max(cycle_max_lean, torso_angle)

                    # Track elbow flare
                    elbow_flare = _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW)
                    if cycle_max_flare is None: cycle_max_flare = elbow_flare
                    else: cycle_max_flare = max(cycle_max_flare, elbow_flare)

                    descent_amt=0.0 if desc_base_shoulder is None else (shoulder_y-desc_base_shoulder)

                    # Bottom detection — mark that we've been at bottom (but DON'T count yet!)
                    at_bottom_ema=(elbow_angle<=ELBOW_BENT_ANGLE) and (descent_amt>=SHOULDER_MIN_DESCENT)
                    at_bottom_raw=(raw_elbow_min<=(ELBOW_BENT_ANGLE+ELBOW_BENT_ANGLE_RAW_MARGIN)) and (descent_amt>=SHOULDER_MIN_DESCENT*0.7)
                    at_bottom_descent=(descent_amt >= SHOULDER_MIN_DESCENT * 1.5) and (raw_elbow_min <= ELBOW_BENT_ANGLE + 20)
                    at_bottom = at_bottom_ema or at_bottom_raw or at_bottom_descent

                    if at_bottom and not reached_bottom:
                        reached_bottom = True
                        cycle_bottom_shoulder = shoulder_y
                        if DEBUG_REPS:
                            print(f"[BOTTOM] frame={frame_idx} elbow={elbow_angle:.1f} raw={raw_elbow_min:.1f} "
                                  f"descent={descent_amt:.4f}")

                    # Track deepest point for bottom shoulder
                    if reached_bottom and shoulder_y > (cycle_bottom_shoulder or 0):
                        cycle_bottom_shoulder = shoulder_y

                    # Ascent detection — shoulder going back up + elbow extending
                    # This is where we COUNT the rep (on the way UP)
                    ascending = False
                    if reached_bottom and not counted_this_cycle:
                        ascent_from_bottom = 0.0
                        if cycle_bottom_shoulder is not None:
                            ascent_from_bottom = cycle_bottom_shoulder - shoulder_y
                        
                        elbow_extending = (elbow_angle >= RESET_ELBOW) or (raw_elbow_max >= RESET_ELBOW - 5)
                        shoulder_rising = (ascent_from_bottom >= RESET_ASCENT)
                        
                        # Count rep when ascending: shoulder is rising AND elbow is extending
                        ascending = (shoulder_rising and elbow_extending) or \
                                   (shoulder_rising and desc_frame_count >= MIN_DESC_FRAMES * 3) or \
                                   (elbow_extending and ascent_from_bottom >= RESET_ASCENT * 0.5 and desc_frame_count >= MIN_DESC_FRAMES * 2)

                    if ascending and not counted_this_cycle:
                        # Evaluate form
                        result_fb = _evaluate_cycle_form(bottom_phase_min_elbow, top_phase_max_elbow,
                                           cycle_max_lean, cycle_max_flare,
                                           session_feedback,
                                           depth_fail_count, lean_fail_count, lockout_fail_count, flare_fail_count,
                                           depth_already_reported, lean_already_reported,
                                           lockout_already_reported, flare_already_reported)
                        if result_fb:
                            depth_fail_count, lean_fail_count, lockout_fail_count, flare_fail_count = \
                                result_fb['depth_fail'], result_fb['lean_fail'], result_fb['lockout_fail'], result_fb['flare_fail']
                            depth_already_reported = result_fb['depth_reported']
                            lean_already_reported = result_fb['lean_reported']
                            lockout_already_reported = result_fb['lockout_reported']
                            flare_already_reported = result_fb['flare_reported']
                            cycle_tip_deeper = result_fb.get('tip_deeper', False)
                            cycle_tip_lean = result_fb.get('tip_lean', False)
                            cycle_tip_lockout = result_fb.get('tip_lockout', False)
                            cycle_tip_elbows = result_fb.get('tip_elbows', False)

                        # Verify rep quality + reject mount/dismount
                        _elbow_ok2 = (cycle_min_elbow <= ELBOW_BENT_ANGLE) or (cycle_min_elbow_raw <= ELBOW_BENT_ANGLE + ELBOW_BENT_ANGLE_RAW_MARGIN)
                        _descent_ok = (cycle_max_descent >= SHOULDER_MIN_DESCENT) and (cycle_max_descent <= MAX_SINGLE_DESCENT)
                        if _descent_ok and _elbow_ok2 and (desc_frame_count >= MIN_DESC_FRAMES):
                            rep_has_tip = cycle_tip_deeper or cycle_tip_lean or cycle_tip_lockout or cycle_tip_elbows
                            _count_rep(rep_reports,rep_count,cycle_min_elbow,
                                       desc_base_shoulder,
                                       baseline_shoulder_y+cycle_max_descent if baseline_shoulder_y is not None else shoulder_y,
                                       all_scores, rep_has_tip)
                            rep_count+=1
                            counted_this_cycle=True
                            if rep_has_tip: bad_reps+=1
                            else: good_reps+=1
                            if DEBUG_REPS:
                                print(f"[REP] #{rep_count} on ASCENT: elbow_ema={cycle_min_elbow:.1f} "
                                      f"elbow_raw={cycle_min_elbow_raw:.1f} descent={cycle_max_descent:.4f} "
                                      f"frames={desc_frame_count}")

                        # Reset for next cycle
                        desc_base_shoulder=shoulder_y
                        cycle_max_descent=0.0; cycle_min_elbow=elbow_angle
                        cycle_min_elbow_raw=raw_elbow_min; desc_frame_count=0; cycle_bottom_shoulder=None
                        bottom_phase_min_elbow=None
                        top_phase_max_elbow=None
                        cycle_max_lean=None
                        cycle_max_flare=None
                        cycle_tip_deeper=False; cycle_tip_lean=False; cycle_tip_lockout=False; cycle_tip_elbows=False
                        reached_bottom=False
                        counted_this_cycle=False

                # Real-time feedback at bottom (only if at_bottom was computed)
                if desc_base_shoulder is not None and at_bottom and not cycle_tip_deeper:
                    if bottom_phase_min_elbow and bottom_phase_min_elbow > DEPTH_MIN_ANGLE:
                        cycle_tip_deeper = True
                        depth_fail_count += 1
                        if depth_fail_count >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
                            session_feedback.add(FB_CUE_DEEPER)
                            depth_already_reported = True
                            cur_rt = FB_CUE_DEEPER

            else:
                # Not ondips or not visible — soft reset (don't lose all state)
                if not ondips:
                    desc_base_shoulder=None; allow_new_bottom=True
                    desc_frame_count=0; cycle_min_elbow_raw=999.0; cycle_bottom_shoulder=None

            # RT hold
            if cur_rt:
                if cur_rt!=rt_fb_msg: rt_fb_msg=cur_rt; rt_fb_hold=RT_FB_HOLD_FRAMES
                else: rt_fb_hold=max(rt_fb_hold,RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold>0: rt_fb_hold-=1

            # Draw
            if return_video and out is not None:
                frame=draw_body_only(frame,lms)
                frame=draw_overlay(frame,reps=rep_count,feedback=(rt_fb_msg if rt_fb_hold>0 else None),depth_pct=depth_live)
                out.write(frame)

            if shoulder_y is not None: shoulder_prev=shoulder_y

    # EOF post-hoc — do NOT count incomplete reps (only count if reached bottom AND was ascending)
    # If someone stops mid-rep at the bottom, that's an incomplete rep

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    print(f"[DIPS] Analysis complete: {rep_count} reps detected, "
          f"pose detected in {pose_detected_count} frames, ondips for {ondips_frame_count} frames")

    # Cleanup rotated temp file
    if cleanup_rotated and os.path.exists(rotated_video):
        try: os.remove(rotated_video)
        except: pass

    # Session score
    if rep_count==0: technique_score=0.0
    else:
        if session_feedback:
            penalty = sum(FB_WEIGHTS.get(m,FB_DEFAULT_WEIGHT) for m in set(session_feedback))
            penalty = max(PENALTY_MIN_IF_ANY, penalty)
        else:
            penalty = 0.0
        technique_score=_half_floor10(max(0.0,10.0-penalty))

    # Build feedback
    all_fb = set(session_feedback) if session_feedback else set()
    fb_list = [cue for cue in FORM_TIP_PRIORITY if cue in all_fb]
    if not fb_list and technique_score >= 10.0 - 1e-6:
        fb_list = ["Great form! Keep it up 💪"]

    form_tip = None
    if all_fb:
        form_tip = max(all_fb, key=lambda m: (FB_WEIGHTS.get(m, FB_DEFAULT_WEIGHT),
                                              -FORM_TIP_PRIORITY.index(m) if m in FORM_TIP_PRIORITY else -999))

    # Write feedback file
    try:
        with open(feedback_path,"w",encoding="utf-8") as f:
            f.write(f"Total Reps: {int(rep_count)}\n")
            f.write(f"Technique Score: {display_half_str(technique_score)} / 10  ({score_label(technique_score)})\n")
            if fb_list:
                f.write("Feedback:\n")
                for ln in fb_list: f.write(f"- {ln}\n")
    except Exception:
        pass

    # Encode video — preserve correct orientation
    final_path=""
    if return_video and os.path.exists(output_path):
        encoded_path=output_path.replace(".mp4","_encoded.mp4")
        try:
            subprocess.run(["ffmpeg","-y","-i",output_path,
                            "-c:v","libx264","-preset","medium",
                            "-crf",str(int(encode_crf if encode_crf is not None else 23)),
                            "-movflags","+faststart","-pix_fmt","yuv420p",
                            # No rotation metadata — video is already correctly oriented
                            "-metadata:s:v:0", "rotate=0",
                            encoded_path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            final_path=encoded_path if os.path.exists(encoded_path) else output_path
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                try: os.remove(output_path)
                except: pass
        except Exception:
            final_path=output_path if os.path.exists(output_path) else ""

    result = {
        "squat_count": int(rep_count),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": fb_list,
        "tips": [],
        "reps": rep_reports,
        "video_path": final_path if return_video else "",
        "feedback_path": feedback_path
    }
    if form_tip is not None:
        result["form_tip"] = form_tip

    return result

# ============ Helper Functions ============
def _calculate_torso_lean(lms, LSH, RSH, LH, RH):
    mid_sh = ((lms[LSH].x + lms[RSH].x)/2.0, (lms[LSH].y + lms[RSH].y)/2.0)
    mid_hp = ((lms[LH].x + lms[RH].x)/2.0, (lms[LH].y + lms[RH].y)/2.0)
    dx = mid_sh[0] - mid_hp[0]
    dy = mid_sh[1] - mid_hp[1]
    angle = abs(math.degrees(math.atan2(abs(dx), abs(dy))))
    return angle

def _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW):
    mid_sh_x = (lms[LSH].x + lms[RSH].x) / 2.0
    left_dx = abs(lms[LE].x - mid_sh_x)
    left_dy = abs(lms[LE].y - lms[LSH].y)
    left_angle = math.degrees(math.atan2(left_dx, left_dy + 1e-9))
    right_dx = abs(lms[RE].x - mid_sh_x)
    right_dy = abs(lms[RE].y - lms[RSH].y)
    right_angle = math.degrees(math.atan2(right_dx, right_dy + 1e-9))
    return max(left_angle, right_angle)

def _evaluate_cycle_form(bottom_phase_min_elbow, top_phase_max_elbow,
                        cycle_max_lean, cycle_max_flare,
                        session_feedback,
                        depth_fail_count, lean_fail_count, lockout_fail_count, flare_fail_count,
                        depth_already_reported, lean_already_reported,
                        lockout_already_reported, flare_already_reported):
    """Evaluate form at end of cycle. Returns updated counters."""
    
    tip_deeper = False
    tip_lean = False
    tip_lockout = False
    tip_elbows = False

    # Depth check
    depth_ok = (bottom_phase_min_elbow is not None) and (bottom_phase_min_elbow <= DEPTH_MIN_ANGLE)
    depth_near = (bottom_phase_min_elbow is not None) and (bottom_phase_min_elbow <= (DEPTH_MIN_ANGLE + DEPTH_NEAR_DEG))
    
    if not depth_ok and not depth_near:
        tip_deeper = True
        depth_fail_count += 1
        if depth_fail_count >= DEPTH_FAIL_MIN_REPS and not depth_already_reported:
            session_feedback.add(FB_CUE_DEEPER)
            depth_already_reported = True

    # Lockout check
    lockout_ok = (top_phase_max_elbow is not None) and (top_phase_max_elbow >= LOCKOUT_MIN_ANGLE)
    lockout_near = (top_phase_max_elbow is not None) and (top_phase_max_elbow >= (LOCKOUT_MIN_ANGLE - LOCKOUT_NEAR_DEG))
    
    if not lockout_ok and not lockout_near:
        tip_lockout = True
        lockout_fail_count += 1
        if lockout_fail_count >= LOCKOUT_FAIL_MIN_REPS and not lockout_already_reported:
            session_feedback.add(FB_CUE_LOCKOUT)
            lockout_already_reported = True

    # Lean check
    lean_ok = (cycle_max_lean is not None) and (cycle_max_lean <= TORSO_MAX_LEAN)
    lean_near = (cycle_max_lean is not None) and (cycle_max_lean <= (TORSO_MAX_LEAN + LEAN_NEAR_DEG))
    
    if not lean_ok and not lean_near:
        tip_lean = True
        lean_fail_count += 1
        if lean_fail_count >= LEAN_FAIL_MIN_REPS and not lean_already_reported:
            session_feedback.add(FB_CUE_LEAN)
            lean_already_reported = True

    # Elbow flare check
    flare_ok = (cycle_max_flare is not None) and (cycle_max_flare <= ELBOW_FLARE_MAX)
    flare_near = (cycle_max_flare is not None) and (cycle_max_flare <= (ELBOW_FLARE_MAX + FLARE_NEAR_DEG))
    
    if not flare_ok and not flare_near:
        tip_elbows = True
        flare_fail_count += 1
        if flare_fail_count >= FLARE_FAIL_MIN_REPS and not flare_already_reported:
            session_feedback.add(FB_CUE_ELBOWS_IN)
            flare_already_reported = True

    return {
        'depth_fail': depth_fail_count, 'lean_fail': lean_fail_count,
        'lockout_fail': lockout_fail_count, 'flare_fail': flare_fail_count,
        'depth_reported': depth_already_reported, 'lean_reported': lean_already_reported,
        'lockout_reported': lockout_already_reported, 'flare_reported': flare_already_reported,
        'tip_deeper': tip_deeper, 'tip_lean': tip_lean,
        'tip_lockout': tip_lockout, 'tip_elbows': tip_elbows,
    }

def _count_rep(rep_reports, rep_count, bottom_elbow, descent_from, bottom_shoulder_y, all_scores, rep_has_tip):
    rep_score = 10.0 if not rep_has_tip else 9.5
    all_scores.append(rep_score)
    rep_reports.append({
        "rep_index": int(rep_count+1),
        "score": float(rep_score),
        "good": bool(rep_score >= 10.0 - 1e-6),
        "bottom_elbow": float(bottom_elbow),
        "descent_from": float(descent_from),
        "bottom_shoulder_y": float(bottom_shoulder_y)
    })

def _ret_err(msg, feedback_path):
    try:
        with open(feedback_path,"w",encoding="utf-8") as f: f.write(msg+"\n")
    except Exception: pass
    return {
        "squat_count": 0, "technique_score": 0.0,
        "technique_score_display": display_half_str(0.0),
        "technique_label": score_label(0.0),
        "good_reps": 0, "bad_reps": 0,
        "feedback": [], "tips": [],
        "reps": [], "video_path": "", "feedback_path": feedback_path
    }

def run_analysis(*args, **kwargs):
    return run_dips_analysis(*args, **kwargs)

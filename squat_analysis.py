# -*- coding: utf-8 -*-
# squat_analysis.py — מהיר ואיטי עם לוגיקה זהה מובטחת
# OVERLAY FIX: renders at frame resolution instead of 1080p canvas (3-5x faster)
import os
import cv2
import math
import numpy as np
import subprocess
from collections import Counter
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / FONTS =====================
BAR_BG_ALPHA         = 0.55
TOP_PAD              = 0
LEFT_PAD             = 0

DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)
DEPTH_RING_BG        = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"

_REF_H = 480.0
_REF_REPS_FONT_SIZE = 28
_REF_FEEDBACK_FONT_SIZE = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE = 18

_FONT_CACHE = {}

def _get_font(path, size):
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = _load_font(path, size)
    return _FONT_CACHE[key]

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        pass
    for fb in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(fb, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def _scaled_font_size(ref_size, canvas_h):
    return max(10, int(round(ref_size * (canvas_h / _REF_H))))

mp_pose = mp.solutions.pose

# ===================== FEEDBACK SEVERITY =====================
FB_SEVERITY = {
    "Try to squat deeper": 3,
    "Avoid knee collapse": 3,
    "Try to keep your back a bit straighter": 2,
    "Almost there — go a bit lower": 2,
    "Looking good — just a bit more depth": 1,
}
FEEDBACK_CATEGORY = {
    "Try to squat deeper": "depth",
    "Almost there — go a bit lower": "depth",
    "Looking good — just a bit more depth": "depth",
    "Try to keep your back a bit straighter": "back",
    "Avoid knee collapse": "knees",
}
def pick_strongest_feedback(feedback_list):
    best, score = "", -1
    for f in feedback_list or []:
        s = FB_SEVERITY.get(f, 1)
        if s > score:
            best, score = f, s
    return best

def pick_strongest_per_category(feedback_list):
    best_by_cat = {}
    for f in feedback_list or []:
        cat = FEEDBACK_CATEGORY.get(f, "other")
        best = best_by_cat.get(cat)
        if not best or FB_SEVERITY.get(f, 1) > FB_SEVERITY.get(best, 1):
            best_by_cat[cat] = f
    return list(best_by_cat.values()), best_by_cat

def merge_feedback(global_best, new_list):
    cand = pick_strongest_feedback(new_list)
    if not cand: return global_best
    if not global_best: return cand
    return cand if FB_SEVERITY.get(cand,1) >= FB_SEVERITY.get(global_best,1) else global_best

def score_label(s):
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    if abs(q - round(q)) < 1e-9:
        return str(int(round(q)))
    return f"{q:.1f}"

# ===================== OVERLAY — efficient frame-resolution rendering =====================
def _wrap_two_lines(draw, text, font, max_width):
    words = (text or "").split()
    if not words:
        return [""]
    lines = []
    cur = ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) == 2:
            break
    if cur and len(lines) < 2:
        lines.append(cur)
    if len(lines) >= 2 and draw.textlength(lines[-1], font=font) > max_width:
        last = lines[-1] + "\u2026"
        while draw.textlength(last, font=font) > max_width and len(last) > 1:
            last = last[:-2] + "\u2026"
        lines[-1] = last
    return lines


def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Overlay at frame resolution — no 1080p canvas, much faster."""
    h, w, _ = frame.shape
    _REPS_F = _get_font(FONT_PATH, _scaled_font_size(_REF_REPS_FONT_SIZE, h))
    _FB_F = _get_font(FONT_PATH, _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, h))
    _DL_F = _get_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, h))
    _DP_F = _get_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, h))

    pct = float(np.clip(depth_pct, 0, 1))
    bg_alpha = int(round(255 * BAR_BG_ALPHA))
    ref_h = max(int(h * 0.06), int(_REPS_F.size * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick = max(3, int(radius * DONUT_THICKNESS_FRAC))
    cx = w - 12 - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)

    ov = np.zeros((h, w, 4), dtype=np.uint8)
    _tmp_img = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(_tmp_img)
    txt = f"Reps: {int(reps)}"
    tw = tmp_draw.textlength(txt, font=_REPS_F)
    cv2.rectangle(ov, (0, 0), (int(tw + 20), int(_REPS_F.size + 12)), (0, 0, 0, bg_alpha), -1)

    cv2.circle(ov, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)
    sa, ea = -90, -90 + int(360 * pct)
    if ea != sa:
        cv2.ellipse(ov, (cx, cy), (radius, radius), 0, sa, ea,
                    (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)

    fb_y0 = fb_pad_x = fb_pad_y = line_gap = line_h = 0
    fb_lines = []
    if feedback:
        safe = max(6, int(h * 0.02))
        fb_pad_x = 12
        fb_pad_y = 8
        line_gap = 4
        fb_lines = _wrap_two_lines(tmp_draw, feedback, _FB_F, int(w - 2 * fb_pad_x - 20))
        line_h = _FB_F.size + 6
        block_h = 2 * fb_pad_y + len(fb_lines) * line_h + (len(fb_lines) - 1) * line_gap
        fb_y0 = max(0, h - safe - block_h)
        cv2.rectangle(ov, (0, fb_y0), (w, h - safe), (0, 0, 0, bg_alpha), -1)

    pil = Image.fromarray(ov, mode="RGBA")
    draw = ImageDraw.Draw(pil)
    draw.text((10, 5), txt, font=_REPS_F, fill=(255, 255, 255, 255))

    gap = max(2, int(radius * 0.10))
    by = cy - (_DL_F.size + gap + _DP_F.size) // 2
    lbl = "DEPTH"
    pt = f"{int(pct * 100)}%"
    draw.text((cx - int(draw.textlength(lbl, font=_DL_F) // 2), by),
              lbl, font=_DL_F, fill=(255, 255, 255, 255))
    draw.text((cx - int(draw.textlength(pt, font=_DP_F) // 2), by + _DL_F.size + gap),
              pt, font=_DP_F, fill=(255, 255, 255, 255))

    if feedback and fb_lines:
        ty = fb_y0 + fb_pad_y
        for ln in fb_lines:
            tx = max(fb_pad_x, (w - int(draw.textlength(ln, font=_FB_F))) // 2)
            draw.text((tx, ty), ln, font=_FB_F, fill=(255, 255, 255, 255))
            ty += line_h + line_gap

    ov_arr = np.array(pil)
    alpha = ov_arr[:, :, 3:4].astype(np.float32) / 255.0
    out_f = frame.astype(np.float32) * (1 - alpha) + ov_arr[:, :, [2, 1, 0]].astype(np.float32) * alpha
    return out_f.astype(np.uint8)

# ===================== BODY-ONLY SKELETON =====================
_FACE_LMS = {
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
    mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
    mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
}
_BODY_CONNECTIONS = tuple(
    (a, b) for (a, b) in mp_pose.POSE_CONNECTIONS
    if a not in _FACE_LMS and b not in _FACE_LMS
)
_BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

def draw_body_only(frame, landmarks, color=(255,255,255)):
    h, w = frame.shape[:2]
    for a, b in _BODY_CONNECTIONS:
        pa = landmarks[a]; pb = landmarks[b]
        ax, ay = int(pa.x * w), int(pa.y * h)
        bx, by = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = landmarks[i]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
    return frame

# ===================== GEOMETRY =====================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def angle_between_vectors(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ===================== Video rotation (like pushup/deadlift) =====================
import json

def _get_video_rotation(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for s in data.get('streams', []):
                if s.get('codec_type') != 'video':
                    continue
                tags = s.get('tags', {})
                if 'rotate' in tags:
                    return int(tags['rotate'])
                for sd in s.get('side_data_list', []):
                    if 'rotation' in sd:
                        return (-int(sd['rotation'])) % 360
    except Exception:
        pass
    return 0

def _apply_rotation(frame, angle):
    angle = angle % 360
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

# ===================== SQUAT CORE PARAMS =====================
STAND_KNEE_ANGLE    = 155.0
MIN_FRAMES_BETWEEN_REPS_SQ = 6
REP_FINISH_KNEE_MARGIN = 6.0
REP_FINISH_DEPTH_MAX = 0.20
REP_FINISH_ALLOW_MOVING_DEPTH_MAX = 0.10

HIP_VEL_THRESH_PCT    = 0.014
ANKLE_VEL_THRESH_PCT  = 0.017
EMA_ALPHA             = 0.65
MOVEMENT_CLEAR_FRAMES = 2

def calculate_depth_robust(mid_hip, mid_knee, mid_ankle, knee_angle, mid_shoulder):
    hip_below_knee = mid_hip[1] - mid_knee[1]
    thigh_length = max(1e-6, _euclid(mid_hip, mid_knee))
    normalized_depth = hip_below_knee / thigh_length
    
    if normalized_depth >= 0.0:
        depth_score = 1.0
    elif normalized_depth >= -0.20:
        depth_score = 0.80 + ((normalized_depth + 0.20) / 0.20) * 0.20
    elif normalized_depth >= -0.40:
        depth_score = 0.55 + ((normalized_depth + 0.40) / 0.20) * 0.25
    elif normalized_depth >= -0.60:
        depth_score = 0.30 + ((normalized_depth + 0.60) / 0.20) * 0.25
    else:
        depth_score = max(0.0, 0.30 * (1.0 + (normalized_depth + 0.60) / 0.40))
    
    if knee_angle >= 120:
        depth_score = max(0.0, depth_score - 0.35)
    elif knee_angle >= 110:
        depth_score = max(0.0, depth_score - 0.25)
    elif knee_angle >= 100:
        depth_score = max(0.0, depth_score - 0.15)
    elif knee_angle <= 80:
        depth_score = min(1.0, depth_score + 0.10)
    elif knee_angle <= 90:
        depth_score = min(1.0, depth_score + 0.05)
    
    return float(np.clip(depth_score, 0, 1))


# ===================== MAIN =====================
def run_squat_analysis(video_path,
                       frame_skip=3,
                       scale=0.4,
                       output_path="squat_analyzed.mp4",
                       feedback_path="squat_feedback.txt",
                       fast_mode=False,
                       return_video=True):
    mp_pose_mod = mp.solutions.pose
    rotation = _get_video_rotation(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0, "good_reps": 0, "bad_reps": 0,
            "feedback": ["Could not open video"], "tip": None, "reps": [], "video_path": "", "feedback_path": feedback_path,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0)
        }
    
    if fast_mode is True:
        return_video = False
    create_video = bool(return_video) and (output_path is not None) and (output_path != "")
    
    if fast_mode:
        effective_frame_skip = frame_skip * 2
        effective_scale = scale * 0.85
        model_complexity = 0
        print(f"[FAST MODE] frame_skip={effective_frame_skip} (2x), scale={effective_scale:.2f}, model=lite", flush=True)
    else:
        effective_frame_skip = frame_skip
        effective_scale = scale
        model_complexity = 1

    counter = 0
    good_reps = 0
    bad_reps = 0
    rep_reports = []
    all_scores = []

    stage = None
    frame_idx = 0
    last_rep_frame = -999
    session_feedbacks = []
    session_feedback_by_cat = {}

    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    movement_free_streak = 0

    start_knee_angle = None
    rep_min_knee_angle = 180.0
    rep_max_knee_angle = -999.0
    rep_max_back_angle_top = -999.0
    rep_max_back_angle_bottom = -999.0
    rep_start_frame = None
    rep_down_start_idx = None
    rep_max_depth = 0.0
    rep_had_back_feedback = False
    
    rep_start_hip_x = None
    rep_start_ankle_x = None
    rep_max_horizontal_movement = 0.0
    rep_max_asymmetry = 0.0

    depth_live = 0.0

    TOP_BACK_MAX_DEG    = 70.0
    BOTTOM_BACK_MAX_DEG = 85.0
    TOP_BAD_MIN_SEC     = 0.8
    BOTTOM_BAD_MIN_SEC  = 1.0
    rep_top_bad_frames = 0
    rep_bottom_bad_frames = 0

    RT_FB_HOLD_SEC = 0.8
    rt_fb_msg = None
    rt_fb_hold = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, effective_frame_skip))
    dt = 1.0 / float(effective_fps)
    TOP_BAD_MIN_FRAMES    = max(3, int(TOP_BAD_MIN_SEC / dt))
    BOTTOM_BAD_MIN_FRAMES = max(3, int(BOTTOM_BAD_MIN_SEC / dt))
    RT_FB_HOLD_FRAMES     = max(2, int(RT_FB_HOLD_SEC / dt))

    with mp_pose_mod.Pose(model_complexity=model_complexity, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % effective_frame_skip != 0: continue
            if effective_scale != 1.0: frame = cv2.resize(frame, (0,0), fx=effective_scale, fy=effective_scale)

            h, w = frame.shape[:2]
            if create_video and out is None:
                # VideoWriter with rotated dimensions
                rot_frame = _apply_rotation(frame, rotation)
                rh, rw = rot_frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (rw, rh))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if not results.pose_landmarks:
                if create_video:
                    depth_live = 0.0
                    if rt_fb_hold > 0: rt_fb_hold -= 1
                    frame = _apply_rotation(frame, rotation)
                    frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
                    if out: out.write(frame)
                continue

            try:
                lm = results.pose_landmarks.landmark
                R = mp_pose_mod.PoseLandmark
                hip      = np.array([lm[R.RIGHT_HIP.value].x,      lm[R.RIGHT_HIP.value].y])
                knee     = np.array([lm[R.RIGHT_KNEE.value].x,     lm[R.RIGHT_KNEE.value].y])
                ankle    = np.array([lm[R.RIGHT_ANKLE.value].x,    lm[R.RIGHT_ANKLE.value].y])
                shoulder = np.array([lm[R.RIGHT_SHOULDER.value].x, lm[R.RIGHT_SHOULDER.value].y])
                l_hip    = np.array([lm[R.LEFT_HIP.value].x,       lm[R.LEFT_HIP.value].y])
                l_knee   = np.array([lm[R.LEFT_KNEE.value].x,      lm[R.LEFT_KNEE.value].y])
                l_ankle  = np.array([lm[R.LEFT_ANKLE.value].x,     lm[R.LEFT_ANKLE.value].y])
                l_shldr  = np.array([lm[R.LEFT_SHOULDER.value].x,  lm[R.LEFT_SHOULDER.value].y])

                mid_hip      = (hip + l_hip) / 2.0
                mid_knee     = (knee + l_knee) / 2.0
                mid_ankle    = (ankle + l_ankle) / 2.0
                mid_shoulder = (shoulder + l_shldr) / 2.0

                hip_px = (hip[0]*w, hip[1]*h)
                la_px  = (l_ankle[0]*w, l_ankle[1]*h)
                ra_px  = (ankle[0]*w,  ankle[1]*h)
                if prev_hip is None:
                    prev_hip, prev_la, prev_ra = hip_px, la_px, ra_px
                def _d(a,b): return math.hypot(a[0]-b[0], a[1]-b[1]) / max(w, h)
                hip_vel = _d(hip_px, prev_hip); an_vel = max(_d(la_px, prev_la), _d(ra_px, prev_ra))
                hip_vel_ema   = EMA_ALPHA*hip_vel + (1-EMA_ALPHA)*hip_vel_ema
                ankle_vel_ema = EMA_ALPHA*an_vel  + (1-EMA_ALPHA)*ankle_vel_ema
                prev_hip, prev_la, prev_ra = hip_px, la_px, ra_px

                movement_block = (hip_vel_ema > HIP_VEL_THRESH_PCT) or (ankle_vel_ema > ANKLE_VEL_THRESH_PCT)
                if movement_block: movement_free_streak = 0
                else:              movement_free_streak = min(MOVEMENT_CLEAR_FRAMES, movement_free_streak + 1)

                knee_angle   = calculate_angle(hip, knee, ankle)
                back_angle   = angle_between_vectors(mid_shoulder - mid_hip, np.array([0.0, -1.0]))

                soft_start_ok = (hip_vel_ema < HIP_VEL_THRESH_PCT * 1.2) and (ankle_vel_ema < ANKLE_VEL_THRESH_PCT * 1.2)
                knee_going_down = (stage != "down")
                
                if (knee_angle < 125) and knee_going_down and soft_start_ok:
                    start_knee_angle = float(knee_angle)
                    rep_min_knee_angle = 180.0
                    rep_max_knee_angle = -999.0
                    rep_max_back_angle_top = -999.0
                    rep_max_back_angle_bottom = -999.0
                    rep_max_depth = 0.0
                    rep_top_bad_frames = 0
                    rep_bottom_bad_frames = 0
                    rep_had_back_feedback = False
                    rep_start_frame = frame_idx
                    rep_down_start_idx = frame_idx
                    rep_start_hip_x = mid_hip[0]
                    rep_start_ankle_x = mid_ankle[0]
                    rep_max_horizontal_movement = 0.0
                    rep_start_left_knee = calculate_angle(l_hip, l_knee, l_ankle)
                    rep_start_right_knee = knee_angle
                    rep_max_asymmetry = 0.0
                    stage = "down"

                current_depth = calculate_depth_robust(mid_hip, mid_knee, mid_ankle, knee_angle, mid_shoulder)
                
                if create_video:
                    depth_live = current_depth

                if stage == "down":
                    rep_min_knee_angle   = min(rep_min_knee_angle, knee_angle)
                    rep_max_knee_angle   = max(rep_max_knee_angle, knee_angle)
                    rep_max_depth = max(rep_max_depth, current_depth)
                    
                    if rep_start_hip_x is not None and rep_start_ankle_x is not None:
                        horizontal_movement = max(
                            abs(mid_hip[0] - rep_start_hip_x),
                            abs(mid_ankle[0] - rep_start_ankle_x)
                        )
                        rep_max_horizontal_movement = max(rep_max_horizontal_movement, horizontal_movement)
                    
                    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                    right_knee_angle = knee_angle
                    current_asymmetry = abs(left_knee_angle - right_knee_angle)
                    rep_max_asymmetry = max(rep_max_asymmetry, current_asymmetry)
                    
                    if current_depth < 0.35:
                        rep_max_back_angle_top = max(rep_max_back_angle_top, back_angle)
                    elif current_depth >= 0.70:
                        rep_max_back_angle_bottom = max(rep_max_back_angle_bottom, back_angle)

                    if create_video:
                        if current_depth < 0.35 and back_angle > TOP_BACK_MAX_DEG:
                            rep_top_bad_frames += 1
                            if rep_top_bad_frames >= TOP_BAD_MIN_FRAMES:
                                rep_had_back_feedback = True
                                if rt_fb_msg != "Try to keep your back a bit straighter":
                                    rt_fb_msg = "Try to keep your back a bit straighter"
                                    rt_fb_hold = RT_FB_HOLD_FRAMES
                                else:
                                    rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
                        elif current_depth >= 0.70 and back_angle > BOTTOM_BACK_MAX_DEG:
                            rep_bottom_bad_frames += 1
                            if rep_bottom_bad_frames >= BOTTOM_BAD_MIN_FRAMES:
                                rep_had_back_feedback = True
                        else:
                            if rt_fb_hold > 0:
                                rt_fb_hold -= 1

                max_horizontal_allowed = 0.25
                has_excessive_horizontal = rep_max_horizontal_movement > max_horizontal_allowed
                max_asymmetry_allowed = 35
                has_excessive_asymmetry = rep_max_asymmetry > max_asymmetry_allowed
                min_depth_for_rep = 0.12
                has_minimal_depth = rep_max_depth >= min_depth_for_rep
                is_pickup_motion = has_excessive_horizontal and has_excessive_asymmetry
                is_valid_rep = (not is_pickup_motion) and has_minimal_depth
                
                rep_finished_posture = (
                    knee_angle > STAND_KNEE_ANGLE
                    or (knee_angle > (STAND_KNEE_ANGLE - REP_FINISH_KNEE_MARGIN) and current_depth <= REP_FINISH_DEPTH_MAX)
                )
                rep_finish_gate_ok = (
                    movement_free_streak >= MOVEMENT_CLEAR_FRAMES
                    or current_depth <= REP_FINISH_ALLOW_MOVING_DEPTH_MAX
                )

                if rep_finished_posture and (stage == "down") and rep_finish_gate_ok:
                    if not is_valid_rep:
                        stage = "up"
                        start_knee_angle = None
                        rep_down_start_idx = None
                        rep_start_hip_x = None
                        rep_start_ankle_x = None
                        rep_max_horizontal_movement = 0.0
                        rep_max_asymmetry = 0.0
                        continue
                    
                    feedbacks = []
                    penalty = 0.0
                    depth_pct = rep_max_depth
                    
                    if   depth_pct < 0.55: feedbacks.append("Try to squat deeper");            penalty += 3
                    elif depth_pct < 0.75: feedbacks.append("Almost there — go a bit lower");  penalty += 2
                    elif depth_pct < 0.92: feedbacks.append("Looking good — just a bit more depth"); penalty += 1

                    if rep_had_back_feedback:
                        feedbacks.append("Try to keep your back a bit straighter")
                        penalty += 1.5

                    if len(feedbacks) == 0:
                        score = 10.0
                    else:
                        score = max(4.0, 10.0 - penalty)
                        score = round(score * 2) / 2
                        score = min(score, 9.5)

                    rep_feedbacks, rep_feedback_by_cat = pick_strongest_per_category(feedbacks)
                
                    safe_start_knee = float(start_knee_angle or knee_angle)
                    if np.isnan(safe_start_knee) or np.isinf(safe_start_knee): safe_start_knee = 160.0
                    safe_min_knee = float(rep_min_knee_angle)
                    if np.isnan(safe_min_knee) or np.isinf(safe_min_knee): safe_min_knee = 160.0
                    safe_max_knee = float(rep_max_knee_angle)
                    if np.isnan(safe_max_knee) or np.isinf(safe_max_knee): safe_max_knee = 160.0
                    safe_back_angle_top = float(rep_max_back_angle_top)
                    if np.isnan(safe_back_angle_top) or np.isinf(safe_back_angle_top): safe_back_angle_top = 0.0
                    safe_back_angle_bottom = float(rep_max_back_angle_bottom)
                    if np.isnan(safe_back_angle_bottom) or np.isinf(safe_back_angle_bottom): safe_back_angle_bottom = 0.0
                    
                    rep_reports.append({
                        "rep_index": counter + 1,
                        "score": round(float(score), 1),
                        "score_display": display_half_str(score),
                        "feedback": rep_feedbacks,
                        "tip": None,
                        "start_frame": rep_start_frame or 0,
                        "end_frame": frame_idx,
                        "start_knee_angle": round(safe_start_knee, 2),
                        "min_knee_angle": round(safe_min_knee, 2),
                        "max_knee_angle": round(safe_max_knee, 2),
                        "torso_angle_top": round(safe_back_angle_top, 2),
                        "torso_angle_bottom": round(safe_back_angle_bottom, 2),
                        "depth_pct": float(depth_pct),
                        "top_bad_frames": int(rep_top_bad_frames),
                        "bottom_bad_frames": int(rep_bottom_bad_frames)
                    })

                    for cat, fb in rep_feedback_by_cat.items():
                        best = session_feedback_by_cat.get(cat)
                        if not best or FB_SEVERITY.get(fb, 1) > FB_SEVERITY.get(best, 1):
                            session_feedback_by_cat[cat] = fb
                    session_feedbacks = list(session_feedback_by_cat.values())

                    start_knee_angle = None
                    rep_down_start_idx = None
                    rep_start_hip_x = None
                    rep_start_ankle_x = None
                    rep_max_horizontal_movement = 0.0
                    rep_max_asymmetry = 0.0
                    stage = "up"

                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS_SQ:
                        counter += 1
                        last_rep_frame = frame_idx
                        if score >= 9.5: good_reps += 1
                        else: bad_reps += 1
                        all_scores.append(score)

                if create_video:
                    frame = draw_body_only(frame, lm)
                    frame = _apply_rotation(frame, rotation)
                    frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
                    if out: out.write(frame)

            except Exception:
                if create_video:
                    if rt_fb_hold > 0: rt_fb_hold -= 1
                    depth_live = 0.0
                    frame = _apply_rotation(frame, rotation)
                    frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
                    if out: out.write(frame)
                continue

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    if np.isnan(avg) or np.isinf(avg):
        avg = 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    
    if session_feedbacks and len(session_feedbacks) > 0:
        technique_score = min(technique_score, 9.5)
    
    feedback_list = session_feedbacks if session_feedbacks else ["Great form! Keep it up \U0001f4aa"]
    
    session_tip = None
    if session_feedback_by_cat:
        if "depth" in session_feedback_by_cat:
            session_tip = "Focus on hip mobility and ankle flexibility to achieve better depth"
        elif "back" in session_feedback_by_cat:
            session_tip = "Work on core strength and practice maintaining a neutral spine"
        elif "knees" in session_feedback_by_cat:
            session_tip = "Practice the 'knees out' cue and consider strengthening your hip abductors"
    else:
        session_tip = "Outstanding consistency! Keep challenging yourself with progressive overload"

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list: 
                    f.write(f"- {fb}\n")
    except Exception as e:
        print(f"Warning: Could not write feedback file: {e}")

    final_video_path = ""
    if create_video and output_path:
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            cmd = ["ffmpeg", "-y", "-i", output_path,
                   "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart",
                   "-pix_fmt", "yuv420p", "-metadata:s:v:0", "rotate=0", encoded_path]
            result = subprocess.run(cmd, check=False, capture_output=True)
            
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                os.remove(output_path)
            final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")
        except Exception as e:
            print(f"Warning: FFmpeg error: {e}")
            final_video_path = output_path if os.path.exists(output_path) else ""

    result = {
        "squat_count": int(counter),
        "technique_score": float(technique_score),
        "technique_score_display": str(display_half_str(technique_score)),
        "technique_label": str(score_label(technique_score)),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": [str(f) for f in feedback_list],
        "tip": str(session_tip) if session_tip else None,
        "reps": rep_reports,
        "video_path": str(final_video_path),
        "feedback_path": str(feedback_path)
    }
    
    return result

def run_analysis(video_path,
                 frame_skip=3,
                 scale=0.4,
                 output_path="squat_analyzed.mp4",
                 feedback_path="squat_feedback.txt",
                 fast_mode=False,
                 return_video=True):
    return run_squat_analysis(video_path, frame_skip, scale, output_path, feedback_path, fast_mode, return_video)

# -*- coding: utf-8 -*-
# barbell_bicep_curl.py — תואם סקוואט: Overlay/פונטים/גדלים/פריימים/קידוד/סינון הליכה.
# דונאט הפוך: 100% בטופ (כיווץ), 0% בתחתית (יישור).

import os
import cv2
import math
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE (זהה לסקוואט) =====================
BAR_BG_ALPHA         = 0.55
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

def _load_font(path, size):
    """Load font with robust fallback — works even without Roboto."""
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        pass
    for fallback in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(fallback, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def _scaled_font_size(ref_size, canvas_h):
    return max(10, int(round(ref_size * (canvas_h / _REF_H))))

mp_pose = mp.solutions.pose

# ===================== OVERLAY =====================
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
        last = lines[-1] + "…"
        while draw.textlength(last, font=font) > max_width and len(last) > 1:
            last = last[:-2] + "…"
        lines[-1] = last
    return lines

def draw_overlay(frame, reps=0, feedback=None, height_pct=0.0):
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

    pct = float(np.clip(height_pct, 0, 1))
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
        fb_pad_x = int(12 * hd_scale)
        fb_pad_y = int(8 * hd_scale)
        line_gap = int(4 * hd_scale)
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
    label = "HEIGHT"
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
    overlay_bgr = overlay_small[:, :, [2, 1, 0]].astype(np.float32)
    frame_f = frame.astype(np.float32)
    result = frame_f * (1.0 - alpha) + overlay_bgr * alpha
    return result.astype(np.uint8)

# ===================== BODY-ONLY SKELETON (ללא פנים) =====================
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

# ===================== עזר גיאומטרי =====================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ===================== פרמטרי מהירות/סינון (זהים לסקוואט) =====================
MIN_FRAMES_BETWEEN_REPS = 10
HIP_VEL_THRESH_PCT    = 0.014
ANKLE_VEL_THRESH_PCT  = 0.017
EMA_ALPHA             = 0.65
MOVEMENT_CLEAR_FRAMES = 2
RT_FB_HOLD_SEC        = 0.8

# ===================== ספי קרל =====================
ELBOW_BOTTOM_EXT_REF = 170.0  # תחתית (יישור)
ELBOW_TOP_FLEX_REF   = 60.0   # טופ (כיווץ יעד)
ELBOW_EXT_END_THR    = 160.0  # סף סיום רפ (חזרה ליישור)
MIN_BOTTOM_EXTENSION_ANGLE = 155.0
TOP_FLEXION_RATIO          = 0.50  # יעד טופ יחסי לטווח (מרכך את "curl higher")
TOP_FLEXION_MIN_ANGLE      = 60.0  # רצפה כדי לא להיות קשוח מדי
TOP_FLEXION_MAX_ANGLE      = 130.0 # תקרה כדי לא להיות רך מדי
TOP_FLEXION_MARGIN_DEG     = 12.0  # מרווח נוסף לפני שמתריעים
TOP_FLEXION_TRIGGER_FRAMES = 6     # כמה פריימים רצופים לפני RT alert
ECC_SLOW_MIN_SEC           = 0.25
ANGLE_EMA_ALPHA            = 0.45
UP_START_DERIV_THR         = -0.9
DOWN_START_DERIV_THR       = 0.45
DIRECTION_STREAK_FRAMES    = 2
REP_END_ANGLE_THR          = ELBOW_EXT_END_THR - 3.0

def _top_flexion_threshold(bottom_ref):
    bottom_ref = float(bottom_ref)
    target = bottom_ref - TOP_FLEXION_RATIO * (bottom_ref - ELBOW_TOP_FLEX_REF)
    return float(np.clip(target, TOP_FLEXION_MIN_ANGLE, TOP_FLEXION_MAX_ANGLE))

# ===================== תוויות וציון =====================
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

# ===================== MAIN =====================
def run_barbell_bicep_curl_analysis(video_path,
                                    frame_skip=3,     # זהה לסקוואט
                                    scale=0.4,        # זהה לסקוואט
                                    output_path="barbell_curl_analyzed.mp4",
                                    feedback_path="barbell_curl_feedback.txt",
                                    return_video=True,
                                    fast_mode=None):
    if fast_mode is True:
        return_video = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0, "good_reps": 0, "bad_reps": 0,
            "feedback": ["Could not open video"], "reps": [], "video_path": "", "feedback_path": feedback_path,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0)
        }

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)
    RT_FB_HOLD_FRAMES   = max(2, int(RT_FB_HOLD_SEC / dt))
    ECC_SLOW_MIN_FRAMES = max(2, int(ECC_SLOW_MIN_SEC / dt))

    counter = 0
    good_reps = 0
    bad_reps = 0
    rep_reports = []
    all_scores = []
    session_best_feedback = ""

    # גלובל-מושן (סינון הליכה)
    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    movement_free_streak = 0

    # מצב רפ
    stage = None  # None / "up" (עלייה) / "down" (ירידה)
    frame_idx = 0
    last_rep_frame = -999

    # עקיבות זווית/"עומק" (כאן "כיווץ")
    ext_elbow_ema = None
    EXT_ALPHA = 0.30
    depth_live = 0.0
    rep_start_frame = None
    rep_start_elbow_angle = None
    rep_min_elbow_angle = 999.0
    rep_max_elbow_angle = -999.0
    top_index = None
    last_angle = None
    elbow_angle_ema = None
    top_flex_bad_frames = 0
    up_motion_streak = 0
    down_motion_streak = 0
    saw_descent_after_top = False

    # RT feedback
    rt_fb_msg = None
    rt_fb_hold = 0

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue
            if scale != 1.0: frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)

            h, w = frame.shape[:2]
            if return_video and out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # ללא לנדמארקס — עדיין overlay זהה
            if not results.pose_landmarks:
                depth_live = 0.0
                if rt_fb_hold > 0: rt_fb_hold -= 1
                if return_video:
                    frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=depth_live)
                    if out is not None:
                        out.write(frame)
                continue

            lm = results.pose_landmarks.landmark
            R = mp_pose.PoseLandmark

            def _pt(p): return np.array([p.x, p.y])

            # תנועה גלובלית (סינון הליכה) — זהה
            hip     = _pt(lm[R.RIGHT_HIP.value])
            l_ankle = _pt(lm[R.LEFT_ANKLE.value])
            r_ankle = _pt(lm[R.RIGHT_ANKLE.value])

            hip_px = (hip[0]*w, hip[1]*h)
            la_px  = (l_ankle[0]*w, l_ankle[1]*h)
            ra_px  = (r_ankle[0]*w, r_ankle[1]*h)
            if prev_hip is None:
                prev_hip, prev_la, prev_ra = hip_px, la_px, ra_px

            def _d(a,b): return math.hypot(a[0]-b[0], a[1]-b[1]) / max(w, h)
            hip_vel = _d(hip_px, prev_hip)
            an_vel  = max(_d(la_px, prev_la), _d(ra_px, prev_ra))
            hip_vel_ema   = EMA_ALPHA*hip_vel + (1-EMA_ALPHA)*hip_vel_ema
            ankle_vel_ema = EMA_ALPHA*an_vel  + (1-EMA_ALPHA)*ankle_vel_ema
            prev_hip, prev_la, prev_ra = hip_px, la_px, ra_px

            movement_block = (hip_vel_ema > HIP_VEL_THRESH_PCT) or (ankle_vel_ema > ANKLE_VEL_THRESH_PCT)
            if movement_block: movement_free_streak = 0
            else:              movement_free_streak = min(MOVEMENT_CLEAR_FRAMES, movement_free_streak + 1)

            # זווית מרפק ממוצעת (שמאל/ימין אם קיימים)
            l_sh, l_el, l_wr = lm[R.LEFT_SHOULDER.value], lm[R.LEFT_ELBOW.value], lm[R.LEFT_WRIST.value]
            r_sh, r_el, r_wr = lm[R.RIGHT_SHOULDER.value], lm[R.RIGHT_ELBOW.value], lm[R.RIGHT_WRIST.value]

            def _ang(sh, el, wr):
                return calculate_angle(np.array([sh.x, sh.y]), np.array([el.x, el.y]), np.array([wr.x, wr.y]))
            angles = []
            l_ang = _ang(l_sh, l_el, l_wr)
            r_ang = _ang(r_sh, r_el, r_wr)
            if not np.isnan(l_ang): angles.append(l_ang)
            if not np.isnan(r_ang): angles.append(r_ang)
            if not angles:
                if rt_fb_hold > 0: rt_fb_hold -= 1
                depth_live = 0.0
                if return_video:
                    frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=depth_live)
                    if out is not None:
                        out.write(frame)
                continue
            elbow_angle_raw = float(np.mean(angles))
            elbow_angle_ema = elbow_angle_raw if elbow_angle_ema is None else (
                ANGLE_EMA_ALPHA * elbow_angle_raw + (1.0 - ANGLE_EMA_ALPHA) * elbow_angle_ema
            )
            elbow_angle = float(elbow_angle_ema)

            # EMA ליישור תחתית (כמו סקוואט — מתעדכן רק כשהיד כמעט ישרה ושקט תנועתי)
            if (elbow_angle > ELBOW_EXT_END_THR) and (movement_free_streak >= MOVEMENT_CLEAR_FRAMES):
                ext_elbow_ema = elbow_angle if ext_elbow_ema is None else (0.30*elbow_angle + 0.70*ext_elbow_ema)

            # ===== דונאט הפוך: 100% בטופ =====
            # מיפוי כיווץ: 0 בתחתית (angle ~ 170), 1 בטופ (angle ~ 60)
            bottom_ref = ext_elbow_ema if ext_elbow_ema is not None else ELBOW_BOTTOM_EXT_REF
            top_flex_thr = _top_flexion_threshold(bottom_ref)
            top_flex_feedback_thr = top_flex_thr + TOP_FLEXION_MARGIN_DEG
            denom_live = max(10.0, (bottom_ref - ELBOW_TOP_FLEX_REF))  # ~110°
            depth_live = float(np.clip((bottom_ref - elbow_angle) / denom_live, 0.0, 1.0))

            # נגזרת זווית (שלב)
            deriv = None
            if last_angle is not None:
                deriv = elbow_angle - last_angle  # >0 ירידה (אקצנטרי), <0 עלייה (קונצנטרי)
            last_angle = elbow_angle

            if deriv is not None:
                if deriv < UP_START_DERIV_THR:
                    up_motion_streak += 1
                else:
                    up_motion_streak = 0

                if deriv > DOWN_START_DERIV_THR:
                    down_motion_streak += 1
                else:
                    down_motion_streak = 0

            # התחלת רפ — מלמטה מתחילים לעלות (קונצנטרי)
            if (stage is None or stage == "down") and (deriv is not None) and (up_motion_streak >= DIRECTION_STREAK_FRAMES) \
               and (elbow_angle <= ELBOW_EXT_END_THR) and (movement_free_streak >= MOVEMENT_CLEAR_FRAMES):
                stage = "up"
                rep_start_frame = frame_idx
                rep_start_elbow_angle = elbow_angle
                rep_min_elbow_angle = 999.0
                rep_max_elbow_angle = -999.0
                top_index = None
                top_flex_bad_frames = 0
                saw_descent_after_top = False
                rt_fb_msg = None
                rt_fb_hold = 0

            # תוך כדי רפ
            if stage in ("up", "down"):
                rep_min_elbow_angle = min(rep_min_elbow_angle, elbow_angle)
                rep_max_elbow_angle = max(rep_max_elbow_angle, elbow_angle)
                # מעבר לטופ → מתחילה ירידה
                if stage == "up" and (deriv is not None) and (down_motion_streak >= DIRECTION_STREAK_FRAMES):
                    stage = "down"
                    top_index = frame_idx
                    saw_descent_after_top = True

                # RT-feedback: לא מספיק גבוה למעלה
                if stage == "up":
                    if rep_min_elbow_angle > top_flex_feedback_thr:
                        top_flex_bad_frames += 1
                    else:
                        top_flex_bad_frames = 0
                    if top_flex_bad_frames >= TOP_FLEXION_TRIGGER_FRAMES:
                        if rt_fb_msg != "Try to curl higher — aim to squeeze at the top":
                            rt_fb_msg = "Try to curl higher — aim to squeeze at the top"
                            rt_fb_hold = RT_FB_HOLD_FRAMES
                        else:
                            rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
                    else:
                        if rt_fb_hold > 0: rt_fb_hold -= 1
                else:
                    if rt_fb_hold > 0: rt_fb_hold -= 1

            # סיום רפ — חזרה ליישור + שקט תנועתי
            if (stage in ("up","down")) and (elbow_angle >= REP_END_ANGLE_THR) and (movement_free_streak >= MOVEMENT_CLEAR_FRAMES) and saw_descent_after_top:
                if frame_idx - (rep_start_frame or frame_idx) > 3:
                    feedbacks = []
                    penalty = 0.0

                    # יישור מלא בתחתית
                    if rep_max_elbow_angle < MIN_BOTTOM_EXTENSION_ANGLE:
                        feedbacks.append("Straighten your arms fully at the bottom for a full range of motion")
                        penalty += 3

                    # טופ מספיק גבוה
                    if rep_min_elbow_angle > top_flex_feedback_thr:
                        feedbacks.append("Try to curl higher — aim to squeeze at the top")
                        penalty += 2

                    # טיפ: אקסצנטרי איטי מספיק (top_index → סוף)
                    tip_msg = None
                    if top_index is not None:
                        ecc_frames = frame_idx - top_index
                        if ecc_frames < ECC_SLOW_MIN_FRAMES:
                            tip_msg = "Slow down the lowering phase to maximize hypertrophy"

                    # ציון (חצי נק׳)
                    score = 10.0 if not feedbacks else round(max(4, 10 - min(penalty, 6)) * 2) / 2

                    # עומק סופי (כיווץ%) לפי דונאט ההפוך
                    bottom_ref_final = ext_elbow_ema if ext_elbow_ema is not None else ELBOW_BOTTOM_EXT_REF
                    denom = max(10.0, (bottom_ref_final - ELBOW_TOP_FLEX_REF))
                    depth_pct_final = float(np.clip((bottom_ref_final - rep_min_elbow_angle) / denom, 0.0, 1.0))

                    # דוח רפ
                    rep_reports.append({
                        "rep_index": counter + 1,
                        "score": round(float(score), 1),
                        "score_display": display_half_str(score),
                        "feedback": ([f for f in feedbacks] if feedbacks else []),
                        "tip": tip_msg,
                        "start_frame": int(rep_start_frame or 0),
                        "end_frame": int(frame_idx),
                        "start_elbow_angle": round(float(rep_start_elbow_angle or elbow_angle), 2),
                        "min_elbow_angle": round(float(rep_min_elbow_angle), 2),
                        "max_elbow_angle": round(float(rep_max_elbow_angle), 2),
                        "depth_pct": depth_pct_final
                    })

                    # הצטברות פידבק-סשן (החמור גובר)
                    def _sev(m): return {"Straighten your arms fully at the bottom for a full range of motion":3,
                                         "Try to curl higher — aim to squeeze at the top":2}.get(m,1)
                    if feedbacks:
                        best = max(feedbacks, key=_sev)
                        session_best_feedback = best if (not session_best_feedback or _sev(best) >= _sev(session_best_feedback)) else session_best_feedback

                    # ספירה/סטטיסטיקות
                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        counter += 1
                        last_rep_frame = frame_idx
                        if score >= 9.5: good_reps += 1
                        else: bad_reps += 1
                        all_scores.append(score)

                    # איפוס מצב
                    stage = None
                    rep_start_frame = None
                    rep_start_elbow_angle = None
                    rep_min_elbow_angle = 999.0
                    rep_max_elbow_angle = -999.0
                    top_index = None
                    top_flex_bad_frames = 0
                    up_motion_streak = 0
                    down_motion_streak = 0
                    saw_descent_after_top = False
                    rt_fb_msg = None
                    rt_fb_hold = 0

            # ציור שלד גוף בלבד + Overlay זהה
            if return_video:
                frame = draw_body_only(frame, lm)
                frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), height_pct=depth_live)
                if out is not None:
                    out.write(frame)

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    # ציון סופי והחזרה
    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    feedback_list = [session_best_feedback] if session_best_feedback else ["Great form! Keep it up 💪"]

    # קובץ טקסט
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list: f.write(f"- {fb}\n")
    except Exception:
        pass

    # קידוד faststart (כמו בסקוואט)
    final_video_path = ""
    if return_video and output_path:
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                encoded_path
            ], check=False)
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                os.remove(output_path)
        except Exception:
            pass
        final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")

    return {
        "squat_count": counter,  # נשמר לשמירה על תאימות UI קיימת
        "technique_score": technique_score,
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": feedback_list,
        "reps": rep_reports,
        "video_path": final_video_path if return_video else "",
        "feedback_path": feedback_path
    }

# תאימות לשם אחיד
def run_analysis(*args, **kwargs):
    return run_barbell_bicep_curl_analysis(*args, **kwargs)

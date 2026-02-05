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

# פונטים — אותם גדלים כמו בסקוואט
REPS_FONT_SIZE        = 28
FEEDBACK_FONT_SIZE    = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE   = 18

def _load_font(size):
    # מנסה קודם את אותו קובץ פונט, אחרת fallback קבוע כדי למנוע שינוי מטריקה
    candidates = [
        "Roboto-VariableFont_wdth,wght.ttf",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Roboto-VariableFont_wdth,wght.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

REPS_FONT        = _load_font(REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(DEPTH_PCT_FONT_SIZE)

mp_pose = mp.solutions.pose

# ===================== OVERLAY (מועתק 1:1 מהסקוואט) =====================
def _draw_depth_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius   = int(radius)
    thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Reps שמאל-עליון; דונאט ימין-עליון; פידבק תחתון — בדיוק כמו בסקוואט."""
    h, w, _ = frame.shape

    # Reps box — שמאל עליון
    pil = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    reps_text = f"Reps: {reps}"
    inner_pad_x, inner_pad_y = 10, 6
    text_w = draw.textlength(reps_text, font=REPS_FONT)
    text_h = REPS_FONT.size
    x0, y0 = 0, 0
    x1 = int(text_w + 2*inner_pad_x); y1 = int(text_h + 2*inner_pad_y)
    top = frame.copy()
    cv2.rectangle(top, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((x0 + inner_pad_x, y0 + inner_pad_y - 1),
                             reps_text, font=REPS_FONT, fill=(255, 255, 255))
    frame = np.array(pil)

    # Donut — ימין עליון (אותו חישוב גודל/מיקום)
    ref_h  = max(int(h * 0.06), int(REPS_FONT_SIZE * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)
    frame = _draw_depth_donut(frame, (cx, cy), radius, thick, float(np.clip(depth_pct, 0, 1)))

    pil  = Image.fromarray(frame); draw = ImageDraw.Draw(pil)
    label_txt = "DEPTH"
    pct_txt   = f"{int(float(np.clip(depth_pct, 0, 1)) * 100)}%"
    label_w = draw.textlength(label_txt, font=DEPTH_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt,   font=DEPTH_PCT_FONT)
    gap     = max(2, int(radius * 0.10))
    base_y  = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size) // 2
    draw.text((cx - int(label_w // 2), base_y),                      label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx - int(pct_w // 2),   base_y + DEPTH_LABEL_FONT.size + gap),
              pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Feedback — תחתון (שתי שורות + …)
    if feedback:
        def wrap_to_two_lines(draw, text, font, max_width):
            words = text.split()
            if not words: return [""]
            lines, cur = [], ""
            for w_ in words:
                trial = (cur + " " + w_).strip()
                if draw.textlength(trial, font=font) <= max_width:
                    cur = trial
                else:
                    if cur: lines.append(cur)
                    cur = w_
                if len(lines) == 2: break
            if cur and len(lines) < 2: lines.append(cur)
            leftover = len(words) - sum(len(l.split()) for l in lines)
            if leftover > 0 and len(lines) >= 2:
                last = lines[-1] + "…"
                while draw.textlength(last, font=font) > max_width and len(last) > 1:
                    last = last[:-2] + "…"
                lines[-1] = last
            return lines

        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        safe_margin = max(6, int(h * 0.02))
        pad_x, pad_y, line_gap = 12, 8, 4
        max_text_w = int(w - 2*pad_x - 20)
        lines = wrap_to_two_lines(draw_fb, feedback, FEEDBACK_FONT, max_text_w)
        line_h = FEEDBACK_FONT.size + 6
        block_h = (2*pad_y) + len(lines)*line_h + (len(lines)-1)*line_gap
        y0 = max(0, h - safe_margin - block_h); y1 = h - safe_margin
        over = frame.copy()
        cv2.rectangle(over, (0, y0), (w, y1), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame); draw_fb = ImageDraw.Draw(pil_fb)
        ty = y0 + pad_y
        for ln in lines:
            tw = draw_fb.textlength(ln, font=FEEDBACK_FONT)
            tx = max(pad_x, (w - int(tw)) // 2)
            draw_fb.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255,255,255))
            ty += line_h + line_gap
        frame = np.array(pil_fb)

    return frame

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

    # עקיבות זווית/“עומק” (כאן "כיווץ")
    ext_elbow_ema = None
    EXT_ALPHA = 0.30
    depth_live = 0.0
    rep_start_frame = None
    rep_start_elbow_angle = None
    rep_min_elbow_angle = 999.0
    rep_max_elbow_angle = -999.0
    top_index = None
    last_angle = None
    top_flex_bad_frames = 0

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
                    frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
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
                    frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
                    if out is not None:
                        out.write(frame)
                continue
            elbow_angle = float(np.mean(angles))

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

            # התחלת רפ — מלמטה מתחילים לעלות (קונצנטרי)
            if (stage is None or stage == "down") and (deriv is not None) and (deriv < -1.5) \
               and (elbow_angle <= ELBOW_EXT_END_THR) and (movement_free_streak >= MOVEMENT_CLEAR_FRAMES):
                stage = "up"
                rep_start_frame = frame_idx
                rep_start_elbow_angle = elbow_angle
                rep_min_elbow_angle = 999.0
                rep_max_elbow_angle = -999.0
                top_index = None
                top_flex_bad_frames = 0
                rt_fb_msg = None
                rt_fb_hold = 0

            # תוך כדי רפ
            if stage in ("up", "down"):
                rep_min_elbow_angle = min(rep_min_elbow_angle, elbow_angle)
                rep_max_elbow_angle = max(rep_max_elbow_angle, elbow_angle)
                # מעבר לטופ → מתחילה ירידה
                if stage == "up" and (deriv is not None) and deriv > 0.8:
                    stage = "down"
                    top_index = frame_idx

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
            if (stage in ("up","down")) and (elbow_angle >= ELBOW_EXT_END_THR) and (movement_free_streak >= MOVEMENT_CLEAR_FRAMES):
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
                    rt_fb_msg = None
                    rt_fb_hold = 0

            # ציור שלד גוף בלבד + Overlay זהה
            if return_video:
                frame = draw_body_only(frame, lm)
                frame = draw_overlay(frame, reps=counter, feedback=(rt_fb_msg if rt_fb_hold>0 else None), depth_pct=depth_live)
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

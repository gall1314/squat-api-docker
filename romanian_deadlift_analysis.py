# -*- coding: utf-8 -*-
# romanian_deadlift_analysis_fixed.py — ספירת חזרות משופרת מכל זווית

import os
import cv2
import math
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / FONTS =====================
BAR_BG_ALPHA = 0.55
DONUT_RADIUS_SCALE = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR = (40, 200, 80)
DEPTH_RING_BG = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
_REF_H = 480.0
_REF_REPS_FONT_SIZE = 28
_REF_FEEDBACK_FONT_SIZE = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE = 18

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def _scaled_font_size(ref_size, canvas_h):
    return max(10, int(round(ref_size * (canvas_h / _REF_H))))

mp_pose = mp.solutions.pose

# ===================== FEEDBACK SEVERITY =====================
FB_SEVERITY = {
    "Go deeper - hinge more at the hips": 3,
    "Bend your knees a bit more": 3,
    "Too much knee bend": 3,
    "Try to keep your back neutral": 2,
}

FEEDBACK_CATEGORY = {
    "Go deeper - hinge more at the hips": "depth",
    "Bend your knees a bit more": "knees",
    "Too much knee bend": "knees",
    "Try to keep your back neutral": "back",
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
    if not cand:
        return global_best
    if not global_best:
        return cand
    return cand if FB_SEVERITY.get(cand, 1) >= FB_SEVERITY.get(global_best, 1) else global_best

def dedupe_feedback(feedback_list):
    seen = set()
    unique = []
    for fb in feedback_list or []:
        if fb in seen:
            continue
        seen.add(fb)
        unique.append(fb)
    return unique

# ===================== SCORE DISPLAY =====================
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

# ===================== OVERLAY =====================
def draw_depth_donut(frame, center, radius, thickness, pct):
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius = int(radius)
    thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90
    end_ang = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

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

def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """Reps בפינת שמאל-עליון; דונאט ימני-עליון; פידבק תחתון — זהה לסקוואט."""
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

# ===================== GEOMETRY =====================
def angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    nrm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = float(np.dot(ba, bc) / nrm)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))

def torso_angle_to_vertical(hip, shoulder):
    vec = shoulder - hip
    nrm = np.linalg.norm(vec) + 1e-9
    cosang = float(np.dot(vec, np.array([0.0, -1.0])) / nrm)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))

def analyze_back_curvature(shoulder, hip, head_like, max_angle_deg=45.0, min_head_dist_ratio=0.35):
    torso_vec = shoulder - hip
    head_vec = head_like - shoulder
    torso_nrm = np.linalg.norm(torso_vec) + 1e-9
    head_nrm = np.linalg.norm(head_vec) + 1e-9
    if head_nrm < (min_head_dist_ratio * torso_nrm):
        return 0.0, False
    cosang = float(np.dot(torso_vec, head_vec) / (torso_nrm * head_nrm))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    ang = math.degrees(math.acos(cosang))
    return ang, ang > max_angle_deg


# ===================== MULTI-ANGLE LANDMARKS =====================
def _get_all_landmarks(lm):
    """
    מחזיר landmarks משני הצדדים + ממוצע.
    זה מאפשר לנו לחשב סיגנלים שעובדים מכל זווית מצלמה.
    """
    PL = mp_pose.PoseLandmark

    def _pt(idx):
        return np.array([lm[idx.value].x, lm[idx.value].y, lm[idx.value].z])

    def _vis(idx):
        return lm[idx.value].visibility

    left = {
        "shoulder": _pt(PL.LEFT_SHOULDER),
        "hip": _pt(PL.LEFT_HIP),
        "knee": _pt(PL.LEFT_KNEE),
        "ankle": _pt(PL.LEFT_ANKLE),
        "ear": _pt(PL.LEFT_EAR),
    }
    right = {
        "shoulder": _pt(PL.RIGHT_SHOULDER),
        "hip": _pt(PL.RIGHT_HIP),
        "knee": _pt(PL.RIGHT_KNEE),
        "ankle": _pt(PL.RIGHT_ANKLE),
        "ear": _pt(PL.RIGHT_EAR),
    }

    # Visibility per side
    left_vis = sum(_vis(i) for i in [PL.LEFT_SHOULDER, PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE, PL.LEFT_EAR])
    right_vis = sum(_vis(i) for i in [PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE, PL.RIGHT_EAR])

    # Pick dominant side for side-view metrics
    if left_vis >= right_vis:
        dominant = left
        dominant_vis = left_vis
    else:
        dominant = right
        dominant_vis = right_vis

    # Midpoint (average both sides) — works well from front/back views
    mid = {}
    for key in left:
        mid[key] = (left[key] + right[key]) / 2.0

    return {
        "left": left,
        "right": right,
        "dominant": dominant,
        "dominant_vis": dominant_vis,
        "mid": mid,
        "left_vis": left_vis,
        "right_vis": right_vis,
    }


def _get_side_landmarks(lm):
    """Backward-compatible: returns 2D dominant side landmarks."""
    PL = mp_pose.PoseLandmark
    sides = {
        "left": [PL.LEFT_SHOULDER, PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE, PL.LEFT_EAR],
        "right": [PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE, PL.RIGHT_EAR],
    }
    vis = {}
    for side, idxs in sides.items():
        vis[side] = sum(lm[i.value].visibility for i in idxs)
    side = "left" if vis["left"] >= vis["right"] else "right"
    idxs = sides[side]
    return {
        "shoulder": np.array([lm[idxs[0].value].x, lm[idxs[0].value].y]),
        "hip": np.array([lm[idxs[1].value].x, lm[idxs[1].value].y]),
        "knee": np.array([lm[idxs[2].value].x, lm[idxs[2].value].y]),
        "ankle": np.array([lm[idxs[3].value].x, lm[idxs[3].value].y]),
        "ear": np.array([lm[idxs[4].value].x, lm[idxs[4].value].y]),
    }


# ===================== COMPOSITE MOVEMENT SIGNAL =====================
def compute_movement_signal(all_lm):
    """
    מחשב סיגנל תנועה מורכב שעובד מכל זווית מצלמה.
    
    הרעיון: במקום להסתמך רק על זווית הגו (שמשתנה דרמטית לפי זווית מצלמה),
    משלבים כמה סיגנלים:
    
    1. shoulder_drop: כמה הכתפיים ירדו ביחס לירכיים (Y axis) — עובד מכל זווית
    2. hip_shoulder_ratio: יחס המרחק האנכי כתף-ירך / גובה כולל — עובד מכל זווית
    3. torso_angle: זווית הגו מול אנך — עובד טוב מהצד
    4. shoulder_y_normalized: מיקום Y מנורמל של הכתפיים — עובד מכל זווית
    
    הסיגנל הסופי הוא ממוצע משוקלל שנותן משקל גבוה יותר לסיגנלים
    שרלוונטיים לזווית המצלמה הנוכחית.
    """
    mid = all_lm["mid"]
    dom = all_lm["dominant"]

    # Signal 1: Shoulder drop relative to hip (normalized)
    # בדדליפט רומני הכתפיים יורדות ביחס לירכיים
    hip_y = mid["hip"][1]
    shoulder_y = mid["shoulder"][1]
    ankle_y = mid["ankle"][1]

    # body_height = distance from ankle to shoulder at standing
    body_height = abs(ankle_y - shoulder_y) + 1e-6
    # shoulder_drop = how much shoulders dropped relative to hip
    # At standing: shoulder is above hip → negative value
    # At bottom: shoulder approaches hip level → value approaches 0 or positive
    shoulder_hip_diff = (shoulder_y - hip_y)  # positive = shoulder below hip
    shoulder_drop_signal = shoulder_hip_diff / body_height
    # Normalize: at standing ~= -0.3 to -0.4, at bottom ~= -0.05 to +0.1
    # Map to 0..1
    shoulder_drop_norm = float(np.clip((shoulder_drop_signal + 0.4) / 0.5, 0.0, 1.0))

    # Signal 2: Shoulder Y position (normalized to body)
    # Simple but effective — shoulders go down during RDL from any angle
    shoulder_y_norm = float(np.clip((shoulder_y - hip_y + 0.3) / 0.4, 0.0, 1.0))

    # Signal 3: Torso angle (2D, works best from side)
    dom_shoulder_2d = dom["shoulder"][:2]
    dom_hip_2d = dom["hip"][:2]
    torso_ang = torso_angle_to_vertical(dom_hip_2d, dom_shoulder_2d)
    torso_signal = float(np.clip(torso_ang / 80.0, 0.0, 1.0))  # 80° = very deep hinge

    # Signal 4: Hip-knee-ankle angle change (knee bend)
    dom_knee_2d = dom["knee"][:2]
    dom_ankle_2d = dom["ankle"][:2]
    knee_ang = angle_deg(dom_hip_2d, dom_knee_2d, dom_ankle_2d)
    # RDL: slight knee bend, ~160-170 standing, ~140-160 at bottom
    knee_signal = float(np.clip((180.0 - knee_ang) / 40.0, 0.0, 1.0))

    # Signal 5: 3D torso angle using Z coordinate
    # MediaPipe provides z (depth) — use it for better angle from front/back views
    torso_3d = dom["shoulder"] - dom["hip"]  # 3D vector
    up_3d = np.array([0.0, -1.0, 0.0])
    torso_3d_norm = np.linalg.norm(torso_3d) + 1e-9
    cos_3d = float(np.dot(torso_3d, up_3d) / torso_3d_norm)
    cos_3d = float(np.clip(cos_3d, -1.0, 1.0))
    torso_3d_angle = math.degrees(math.acos(cos_3d))
    torso_3d_signal = float(np.clip(torso_3d_angle / 80.0, 0.0, 1.0))

    # Determine camera angle to weight signals appropriately
    # If both sides are similarly visible → front/back view → rely more on Y-based signals
    # If one side is much more visible → side view → torso angle is reliable
    vis_ratio = min(all_lm["left_vis"], all_lm["right_vis"]) / (max(all_lm["left_vis"], all_lm["right_vis"]) + 1e-6)
    # vis_ratio close to 1.0 = front/back view, close to 0.0 = side view

    if vis_ratio > 0.7:
        # Front/back view — torso 2D angle is unreliable, use Y-based + 3D
        weights = {
            "shoulder_drop": 0.30,
            "shoulder_y": 0.20,
            "torso_2d": 0.05,
            "torso_3d": 0.30,
            "knee": 0.15,
        }
    elif vis_ratio > 0.4:
        # Diagonal view — blend everything
        weights = {
            "shoulder_drop": 0.20,
            "shoulder_y": 0.15,
            "torso_2d": 0.20,
            "torso_3d": 0.25,
            "knee": 0.20,
        }
    else:
        # Side view — torso 2D is most reliable
        weights = {
            "shoulder_drop": 0.10,
            "shoulder_y": 0.10,
            "torso_2d": 0.40,
            "torso_3d": 0.20,
            "knee": 0.20,
        }

    composite = (
        weights["shoulder_drop"] * shoulder_drop_norm +
        weights["shoulder_y"] * shoulder_y_norm +
        weights["torso_2d"] * torso_signal +
        weights["torso_3d"] * torso_3d_signal +
        weights["knee"] * knee_signal
    )

    composite = float(np.clip(composite, 0.0, 1.0))

    return {
        "composite": composite,
        "shoulder_drop": shoulder_drop_norm,
        "shoulder_y": shoulder_y_norm,
        "torso_2d": torso_signal,
        "torso_3d": torso_3d_signal,
        "torso_2d_angle": torso_ang,
        "torso_3d_angle": torso_3d_angle,
        "knee_signal": knee_signal,
        "knee_angle": knee_ang,
        "vis_ratio": vis_ratio,
        "weights": weights,
    }


# ===================== REP STATE MACHINE =====================
class RepCounter:
    """
    State machine לספירת חזרות עם peak/valley detection.

    עובד על סיגנל מורכב (0=עמידה, 1=תחתית) עם היסטרזיס:
    - מזהה מעבר לאזור "כניסה" (threshold_enter)
    - מחפש שיא (peak) של הסיגנל
    - מזהה חזרה לאזור "עמידה" (threshold_exit)
    - סופר חזרה רק אם ה-peak עבר סף מינימלי
    """

    # Adaptive thresholds — will be calibrated from first movements
    ENTER_THRESHOLD = 0.25       # סיגנל מעל לזה = התחלת תנועה
    EXIT_THRESHOLD = 0.15        # סיגנל מתחת לזה = חזרה לעמידה
    MIN_PEAK_FOR_REP = 0.40      # שיא מינימלי כדי לספור חזרה
    GOOD_DEPTH_PEAK = 0.55       # שיא שנחשב לעומק טוב
    MIN_FRAMES_BETWEEN = 5       # מינימום פריימים בין חזרות
    MAX_REP_FRAMES = 120         # timeout safety for noisy / oblique angles
    REBOUND_DELTA = 0.015        # rise after valley means movement turned down again
    MIN_RETURN_FROM_PEAK = 0.04  # require partial return toward top between reps
    NORMALIZED_PEAK_THRESHOLD = 0.58
    NORMALIZED_DROP_THRESHOLD = 0.10
    MIN_REP_DURATION_FRAMES = 6
    MIN_PEAK_DELTA = 0.07
    MIN_NORMALIZED_PEAK_DELTA = 0.18

    def __init__(self):
        self.state = "standing"  # standing | descending | ascending
        self.count = 0
        self.current_peak = 0.0
        self.frames_in_state = 0
        self.last_rep_frame = -999
        self.signal_history = []  # for smoothing
        self.smoothed = 0.0
        self.rep_start_frame = -1
        self.ascent_valley = 1.0
        self.rep_start_signal = 0.0

        # Per-rep metrics
        self.rep_max_torso_2d = 0.0
        self.rep_max_torso_3d = 0.0
        self.rep_min_knee = 999.0
        self.rep_max_knee = 0.0
        self.rep_back_issue = False
        self.rep_back_angle = 0.0

        # Adaptive calibration
        self.calibration_signals = []
        self.calibrated = False
        self.standing_baseline = 0.0
        self.dynamic_floor = 0.0
        self.dynamic_ceil = 0.6

    def _smooth(self, raw_signal):
        """EMA smoothing to reduce noise."""
        alpha = 0.35
        self.smoothed = self.smoothed + alpha * (raw_signal - self.smoothed)
        return self.smoothed

    def _normalize_by_session_range(self, smoothed_signal):
        """
        Normalize by evolving floor/ceiling so counting works better
        when camera angle compresses the raw signal range.
        """
        # Slow floor update, slightly faster ceiling update.
        self.dynamic_floor = 0.98 * self.dynamic_floor + 0.02 * min(self.dynamic_floor, smoothed_signal)
        self.dynamic_ceil = 0.95 * self.dynamic_ceil + 0.05 * max(self.dynamic_ceil, smoothed_signal)

        rng = max(0.20, self.dynamic_ceil - self.dynamic_floor)
        normalized = (smoothed_signal - self.dynamic_floor) / rng
        return float(np.clip(normalized, 0.0, 1.0))

    def _calibrate(self, signal):
        """
        Collect signals for first ~15 frames to establish standing baseline.
        """
        if self.calibrated:
            return
        self.calibration_signals.append(signal)
        if len(self.calibration_signals) >= 15:
            # Standing baseline = minimum of first signals (likely standing)
            self.standing_baseline = float(np.percentile(self.calibration_signals, 25))
            self.dynamic_floor = self.standing_baseline
            self.dynamic_ceil = max(self.standing_baseline + 0.35, float(np.percentile(self.calibration_signals, 90)))
            # Adjust thresholds relative to baseline
            self.ENTER_THRESHOLD = max(0.15, self.standing_baseline + 0.15)
            self.EXIT_THRESHOLD = max(0.10, self.standing_baseline + 0.08)
            self.calibrated = True

    def _start_rep_tracking(self, signal_data, frame_idx, smoothed):
        self.state = "descending"
        self.current_peak = smoothed
        self.frames_in_state = 0
        self.rep_start_frame = frame_idx
        self.ascent_valley = smoothed
        self.rep_start_signal = smoothed
        self.rep_max_torso_2d = signal_data.get("torso_2d_angle", 0)
        self.rep_max_torso_3d = signal_data.get("torso_3d_angle", 0)
        self.rep_min_knee = signal_data.get("knee_angle", 170)
        self.rep_max_knee = signal_data.get("knee_angle", 170)
        self.rep_back_issue = False
        self.rep_back_angle = 0.0

    def _normalized_peak(self):
        return (self.current_peak - self.dynamic_floor) / max(0.20, self.dynamic_ceil - self.dynamic_floor)

    def _has_meaningful_excursion(self):
        peak_delta = self.current_peak - self.rep_start_signal
        normalized_delta = peak_delta / max(0.20, self.dynamic_ceil - self.dynamic_floor)
        return (peak_delta >= self.MIN_PEAK_DELTA) or (normalized_delta >= self.MIN_NORMALIZED_PEAK_DELTA)

    def _build_rep_info(self):
        return {
            "rep": self.count,
            "peak_signal": self.current_peak,
            "good_depth": self.current_peak >= self.GOOD_DEPTH_PEAK,
            "max_torso_2d": self.rep_max_torso_2d,
            "max_torso_3d": self.rep_max_torso_3d,
            "min_knee": self.rep_min_knee,
            "max_knee": self.rep_max_knee,
            "back_issue": self.rep_back_issue,
            "back_angle": self.rep_back_angle,
        }

    def _reset_to_standing(self):
        self.state = "standing"
        self.frames_in_state = 0
        self.current_peak = 0.0
        self.rep_start_frame = -1
        self.ascent_valley = 1.0
        self.rep_start_signal = 0.0

    def finalize_pending_rep(self, frame_idx):
        """
        Count a likely last rep when video ends mid-ascent.
        Useful when the last lockout is cut off by the clip ending.
        """
        if self.state != "ascending":
            return None

        valley_drop = self.current_peak - self.ascent_valley
        normalized_drop = valley_drop / max(0.20, self.dynamic_ceil - self.dynamic_floor)
        is_valid_peak = (self.current_peak >= self.MIN_PEAK_FOR_REP) or (self._normalized_peak() >= self.NORMALIZED_PEAK_THRESHOLD)
        has_return = (valley_drop >= self.MIN_RETURN_FROM_PEAK * 0.7) or (normalized_drop >= self.NORMALIZED_DROP_THRESHOLD * 0.8)

        if is_valid_peak and has_return and self._has_meaningful_excursion() and (frame_idx - self.last_rep_frame) >= self.MIN_FRAMES_BETWEEN and (frame_idx - self.rep_start_frame) >= self.MIN_REP_DURATION_FRAMES:
            self.count += 1
            self.last_rep_frame = frame_idx
            rep_info = self._build_rep_info()
            self._reset_to_standing()
            return rep_info

        return None

    def update(self, signal_data, frame_idx):
        """
        מעדכן את ה-state machine עם סיגנל חדש.
        מחזיר dict אם חזרה הושלמה, None אחרת.
        """
        raw = signal_data["composite"]
        self._calibrate(raw)
        smoothed = self._smooth(raw)
        normalized = self._normalize_by_session_range(smoothed)
        self.frames_in_state += 1

        # Track per-rep metrics regardless of state
        if self.state in ("descending", "ascending"):
            self.rep_max_torso_2d = max(self.rep_max_torso_2d, signal_data.get("torso_2d_angle", 0))
            self.rep_max_torso_3d = max(self.rep_max_torso_3d, signal_data.get("torso_3d_angle", 0))
            knee_ang = signal_data.get("knee_angle", 170)
            self.rep_min_knee = min(self.rep_min_knee, knee_ang)
            self.rep_max_knee = max(self.rep_max_knee, knee_ang)

        if self.state == "standing":
            normalized_enter = normalized >= 0.34
            if smoothed >= self.ENTER_THRESHOLD or normalized_enter:
                self._start_rep_tracking(signal_data, frame_idx, smoothed)

        elif self.state == "descending":
            if smoothed > self.current_peak:
                self.current_peak = smoothed

            # If signal starts dropping, we're now ascending.
            if smoothed < self.current_peak - 0.015 and self.frames_in_state >= 2:
                self.state = "ascending"
                self.frames_in_state = 0
                self.ascent_valley = smoothed

            if self.rep_start_frame > 0 and (frame_idx - self.rep_start_frame) > self.MAX_REP_FRAMES:
                self._reset_to_standing()

        elif self.state == "ascending":
            self.ascent_valley = min(self.ascent_valley, smoothed)
            normalized_exit = normalized <= 0.28

            # Standard completion by returning near standing.
            if (smoothed <= self.EXIT_THRESHOLD or normalized_exit) and (frame_idx - self.last_rep_frame) >= self.MIN_FRAMES_BETWEEN:
                if (self.current_peak >= self.MIN_PEAK_FOR_REP or self._normalized_peak() >= self.NORMALIZED_PEAK_THRESHOLD) and self._has_meaningful_excursion() and (frame_idx - self.rep_start_frame) >= self.MIN_REP_DURATION_FRAMES:
                    self.count += 1
                    self.last_rep_frame = frame_idx
                    rep_info = self._build_rep_info()
                    self._reset_to_standing()
                    return rep_info
                self._reset_to_standing()

            # Touch-and-go completion: ascent reverses before full exit.
            valley_drop = self.current_peak - self.ascent_valley
            if smoothed > (self.ascent_valley + self.REBOUND_DELTA) and self.frames_in_state >= 2:
                normalized_drop = valley_drop / max(0.20, self.dynamic_ceil - self.dynamic_floor)
                is_valid_peak = (self.current_peak >= self.MIN_PEAK_FOR_REP) or (self._normalized_peak() >= self.NORMALIZED_PEAK_THRESHOLD)
                has_return = (valley_drop >= self.MIN_RETURN_FROM_PEAK) or (normalized_drop >= self.NORMALIZED_DROP_THRESHOLD)

                if is_valid_peak and has_return and self._has_meaningful_excursion() and (frame_idx - self.last_rep_frame) >= self.MIN_FRAMES_BETWEEN and (frame_idx - self.rep_start_frame) >= self.MIN_REP_DURATION_FRAMES:
                    self.count += 1
                    self.last_rep_frame = frame_idx
                    rep_info = self._build_rep_info()
                    self._start_rep_tracking(signal_data, frame_idx, smoothed)
                    return rep_info

            # Handle case where person goes much deeper again without clear valley.
            if smoothed > self.current_peak:
                self.current_peak = smoothed
                self.state = "descending"
                self.frames_in_state = 0

            if self.rep_start_frame > 0 and (frame_idx - self.rep_start_frame) > self.MAX_REP_FRAMES:
                self._reset_to_standing()

        return None


# ===================== PARAMETERS =====================
HINGE_BOTTOM_ANGLE = 55.0  # For depth quality check
KNEE_MIN_ANGLE = 172.0
KNEE_OPTIMAL_MIN = 160.0
KNEE_OPTIMAL_MAX = 170.0
KNEE_MAX_ANGLE = 125.0
BACK_MAX_ANGLE = 60.0

MIN_SCORE = 4.0
MAX_SCORE = 10.0

PROG_ALPHA = 0.3


def run_romanian_deadlift_analysis(video_path,
                                   frame_skip=3,
                                   scale=0.4,
                                   output_path="romanian_deadlift_analyzed.mp4",
                                   feedback_path="romanian_deadlift_feedback.txt",
                                   return_video=True,
                                   fast_mode=False):
    """
    Romanian Deadlift analysis — improved multi-angle rep counting:
    ✅ Composite movement signal from multiple body metrics
    ✅ Works from side, front, diagonal, and full profile views
    ✅ Adaptive calibration for standing baseline
    ✅ Peak/valley state machine with hysteresis
    ✅ Depth, knee, and back quality checks
    """
    import sys
    print(f"[RDL] Starting analysis: fast_mode={fast_mode}, return_video={return_video}", file=sys.stderr, flush=True)
    print(f"[RDL] Video path: {video_path}", file=sys.stderr, flush=True)

    mp_pose_mod = mp.solutions.pose

    if fast_mode is True:
        return_video = False
    create_video = bool(return_video) and (output_path is not None) and (output_path != "")

    print(f"[RDL] create_video={create_video}", file=sys.stderr, flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[RDL ERROR] Cannot open video: {video_path}", file=sys.stderr, flush=True)
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"[RDL] Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration", file=sys.stderr, flush=True)

    if total_frames < 10:
        print(f"[RDL ERROR] File has only {total_frames} frames - not a valid video!", file=sys.stderr, flush=True)
        cap.release()
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0,
            "feedback": [f"Invalid video file - only {total_frames} frame(s). Please upload an actual video."],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    if fast_mode:
        effective_frame_skip = frame_skip
        effective_scale = scale * 0.85
        model_complexity = 0
        print(f"[RDL FAST] frame_skip={effective_frame_skip}, scale={effective_scale:.2f}, model=lite", file=sys.stderr, flush=True)
    else:
        effective_frame_skip = frame_skip
        effective_scale = scale
        model_complexity = 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, effective_frame_skip))
    dt = 1.0 / float(effective_fps)

    # ✅ NEW: Use RepCounter state machine
    rep_counter = RepCounter()
    good_reps = bad_reps = 0
    all_scores = []
    rep_reports = []
    session_feedbacks = []
    session_feedback_by_cat = {}

    frame_idx = 0
    prog = 0.0

    rt_fb_msg = None
    rt_fb_hold = 0

    print(f"[RDL] Starting pose detection loop", file=sys.stderr, flush=True)

    import time
    loop_start_time = time.time()
    MAX_PROCESSING_TIME = 180
    frames_processed = 0

    with mp_pose_mod.Pose(model_complexity=model_complexity,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - loop_start_time
            if elapsed > MAX_PROCESSING_TIME:
                print(f"[RDL ERROR] Processing timeout after {elapsed:.1f}s (processed {frames_processed} frames)", file=sys.stderr, flush=True)
                break

            frame_idx += 1
            if frame_idx % effective_frame_skip != 0:
                continue

            frames_processed += 1

            if frames_processed % 30 == 0:
                print(f"[RDL] Progress: {frames_processed} frames, {rep_counter.count} reps, {elapsed:.1f}s", file=sys.stderr, flush=True)

            work = cv2.resize(frame, (0, 0), fx=effective_scale, fy=effective_scale) if effective_scale != 1.0 else frame
            if create_video and out is None:
                h0, w0 = work.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w0, h0))

            rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                if create_video and out is not None:
                    frame_drawn = draw_overlay(work.copy(), reps=rep_counter.count,
                                               feedback=(rt_fb_msg if rt_fb_hold > 0 else None),
                                               depth_pct=prog)
                    out.write(frame_drawn)
                continue

            lm = res.pose_landmarks.landmark

            # ✅ Get all landmarks (both sides + mid)
            all_lm = _get_all_landmarks(lm)

            # ✅ Compute composite movement signal
            signal_data = compute_movement_signal(all_lm)

            # Smooth depth progress for display
            prog = prog + PROG_ALPHA * (signal_data["composite"] - prog)

            # ✅ Back curvature check (using dominant side 2D)
            pts = _get_side_landmarks(lm)
            back_angle, back_bad = analyze_back_curvature(
                pts["shoulder"], pts["hip"], pts["ear"], max_angle_deg=BACK_MAX_ANGLE
            )

            # Track back issues in rep counter
            if rep_counter.state in ("descending", "ascending") and back_bad:
                rep_counter.rep_back_issue = True
                rep_counter.rep_back_angle = max(rep_counter.rep_back_angle, back_angle)

            # ✅ Update state machine
            rep_result = rep_counter.update(signal_data, frame_idx)

            if rep_result is not None:
                # Rep completed! Evaluate quality
                feedback = []
                score = MAX_SCORE

                # Depth check — use the best torso angle available
                best_torso = max(rep_result["max_torso_2d"], rep_result["max_torso_3d"])
                if not rep_result["good_depth"] and best_torso < HINGE_BOTTOM_ANGLE:
                    feedback.append("Go deeper - hinge more at the hips")
                    score -= 2.0

                # Knee check
                if rep_result["max_knee"] > KNEE_MIN_ANGLE:
                    feedback.append("Bend your knees a bit more")
                    score -= 1.5
                elif rep_result["min_knee"] < KNEE_MAX_ANGLE:
                    feedback.append("Too much knee bend")
                    score -= 2.0

                # Back check
                if rep_result["back_issue"] and rep_result["back_angle"] > (BACK_MAX_ANGLE + 5.0):
                    feedback.append("Try to keep your back neutral")
                    score -= 1.0

                score = float(max(MIN_SCORE, min(MAX_SCORE, score)))
                all_scores.append(score)

                if score >= 9.0:
                    good_reps += 1
                else:
                    bad_reps += 1

                session_feedbacks.extend(feedback)
                _, session_feedback_by_cat = pick_strongest_per_category(session_feedbacks)

                rep_reports.append({
                    "rep": rep_result["rep"],
                    "score": round(score, 2),
                    "score_display": display_half_str(score),
                    "label": score_label(score),
                    "feedback": feedback,
                    "metrics": {
                        "peak_signal": round(rep_result["peak_signal"], 3),
                        "max_torso_2d_angle": round(rep_result["max_torso_2d"], 2),
                        "max_torso_3d_angle": round(rep_result["max_torso_3d"], 2),
                        "min_knee_angle": round(rep_result["min_knee"], 2),
                        "max_knee_angle": round(rep_result["max_knee"], 2),
                        "back_angle": round(rep_result["back_angle"], 2),
                        "view_type": "front/back" if signal_data["vis_ratio"] > 0.7 else (
                            "diagonal" if signal_data["vis_ratio"] > 0.4 else "side"
                        ),
                    }
                })

                print(f"[RDL] Rep {rep_result['rep']}: peak={rep_result['peak_signal']:.3f}, "
                      f"torso2d={rep_result['max_torso_2d']:.1f}°, torso3d={rep_result['max_torso_3d']:.1f}°, "
                      f"knee=[{rep_result['min_knee']:.1f}-{rep_result['max_knee']:.1f}]°, "
                      f"view_ratio={signal_data['vis_ratio']:.2f}",
                      file=sys.stderr, flush=True)

                rt_fb_msg = pick_strongest_feedback(feedback)
                rt_fb_hold = int(0.7 / dt)

            if rt_fb_hold > 0:
                rt_fb_hold -= 1

            if create_video and out is not None:
                frame_drawn = draw_overlay(work.copy(), reps=rep_counter.count,
                                           feedback=(rt_fb_msg if rt_fb_hold > 0 else None),
                                           depth_pct=prog)
                out.write(frame_drawn)

    # Finalize likely last rep if video ended mid-lockout
    rep_result = rep_counter.finalize_pending_rep(frame_idx)
    if rep_result is not None:
        feedback = []
        score = MAX_SCORE

        best_torso = max(rep_result["max_torso_2d"], rep_result["max_torso_3d"])
        if not rep_result["good_depth"] and best_torso < HINGE_BOTTOM_ANGLE:
            feedback.append("Go deeper - hinge more at the hips")
            score -= 2.0

        if rep_result["max_knee"] > KNEE_MIN_ANGLE:
            feedback.append("Bend your knees a bit more")
            score -= 1.5
        elif rep_result["min_knee"] < KNEE_MAX_ANGLE:
            feedback.append("Too much knee bend")
            score -= 2.0

        if rep_result["back_issue"] and rep_result["back_angle"] > (BACK_MAX_ANGLE + 5.0):
            feedback.append("Try to keep your back neutral")
            score -= 1.0

        score = float(max(MIN_SCORE, min(MAX_SCORE, score)))
        all_scores.append(score)
        if score >= 9.0:
            good_reps += 1
        else:
            bad_reps += 1

        session_feedbacks.extend(feedback)
        _, session_feedback_by_cat = pick_strongest_per_category(session_feedbacks)

        rep_reports.append({
            "rep": rep_result["rep"],
            "score": round(score, 2),
            "score_display": display_half_str(score),
            "label": score_label(score),
            "feedback": feedback,
            "metrics": {
                "peak_signal": round(rep_result["peak_signal"], 3),
                "max_torso_2d_angle": round(rep_result["max_torso_2d"], 2),
                "max_torso_3d_angle": round(rep_result["max_torso_3d"], 2),
                "min_knee_angle": round(rep_result["min_knee"], 2),
                "max_knee_angle": round(rep_result["max_knee"], 2),
                "back_angle": round(rep_result["back_angle"], 2),
                "view_type": "end_of_clip",
            }
        })

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    counter = rep_counter.count
    print(f"[RDL] Video processing complete. Reps={counter}", file=sys.stderr, flush=True)

    avg = np.mean(all_scores) if all_scores else 0.0
    if np.isnan(avg) or np.isinf(avg):
        avg = 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    if session_feedbacks and len(session_feedbacks) > 0:
        technique_score = min(technique_score, 9.5)

    feedback_list = dedupe_feedback(list(session_feedback_by_cat.values())) if session_feedback_by_cat else (["Perfect form! 🔥"] if not session_feedbacks else dedupe_feedback(session_feedbacks))

    session_tip = None
    if session_feedback_by_cat:
        if "depth" in session_feedback_by_cat:
            session_tip = "Focus on pushing your hips back further to reach full depth"
        elif "knees" in session_feedback_by_cat:
            session_tip = "Romanian deadlifts need a soft knee bend throughout the movement"
        elif "back" in session_feedback_by_cat:
            session_tip = "Keep your core tight and chest up"
    else:
        session_tip = "Great technique! Keep building that posterior chain 💪"

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
        print(f"[RDL] Starting FFmpeg encoding", file=sys.stderr, flush=True)
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                encoded_path
            ], check=False, capture_output=True)

            if os.path.exists(output_path) and os.path.exists(encoded_path):
                os.remove(output_path)
            final_video_path = encoded_path if os.path.exists(encoded_path) else (
                output_path if os.path.exists(output_path) else "")
            print(f"[RDL] FFmpeg encoding complete", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Warning: FFmpeg error: {e}")
            final_video_path = output_path if os.path.exists(output_path) else ""
    else:
        print(f"[RDL] Skipping video encoding (create_video={create_video})", file=sys.stderr, flush=True)

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

    print(f"[RDL] Returning result (video_path length={len(final_video_path)})", file=sys.stderr, flush=True)

    return result


def run_analysis(*args, **kwargs):
    return run_romanian_deadlift_analysis(*args, **kwargs)

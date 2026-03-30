# -*- coding: utf-8 -*-
# squat_analysis.py — Romanian Deadlift analysis
# FIXED: restored RepCounter + DepthNormalizer + composite signal from v5.1
#        (two-pass architecture with correct rep counting logic)
# KEPT:  efficient frame-resolution overlay from current version

import os
import cv2
import math
import json
import sys
import time
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== STYLE / FONTS =====================
BAR_BG_ALPHA         = 0.55
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)
DEPTH_RING_BG        = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"

_REF_H = 480.0
_REF_REPS_FONT_SIZE        = 28
_REF_FEEDBACK_FONT_SIZE    = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE   = 18

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

# ===================== FEEDBACK =====================
FB_SEVERITY = {
    "Go deeper - hinge more at the hips": 3,
    "Bend your knees a bit more":         3,
    "Your torso is going too far forward": 2,
}
FEEDBACK_CATEGORY = {
    "Go deeper - hinge more at the hips": "depth",
    "Bend your knees a bit more":         "knees",
    "Your torso is going too far forward": "back",
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

def dedupe_feedback(feedback_list):
    seen, unique = set(), []
    for fb in feedback_list or []:
        if fb not in seen:
            seen.add(fb)
            unique.append(fb)
    return unique

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
    lines, cur = [], ""
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
    _FB_F   = _get_font(FONT_PATH, _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, h))
    _DL_F   = _get_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, h))
    _DP_F   = _get_font(FONT_PATH, _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, h))

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
_FACE_LMS         = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS      = tuple()

def _init_skeleton_data():
    global _FACE_LMS, _BODY_CONNECTIONS, _BODY_POINTS
    if not _BODY_CONNECTIONS:
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
        _BODY_POINTS = tuple(sorted({i for c in _BODY_CONNECTIONS for i in c}))

def _dyn_thickness(h):
    return max(2, int(round(h * 0.003))), max(3, int(round(h * 0.005)))

def draw_body_only(frame, lms, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    line, dot = _dyn_thickness(h)
    for a, b in _BODY_CONNECTIONS:
        pa, pb = lms[a], lms[b]
        cv2.line(frame,
                 (int(pa.x * w), int(pa.y * h)),
                 (int(pb.x * w), int(pb.y * h)),
                 color, line, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = lms[i]
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), dot, color, -1, cv2.LINE_AA)
    return frame

# ===================== GEOMETRY =====================
def angle_deg(a, b, c):
    ba = a - b
    bc = c - b
    nrm = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return float(math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / nrm, -1.0, 1.0)))))

def torso_angle_to_vertical(hip, shoulder):
    vec = shoulder - hip
    nrm = np.linalg.norm(vec) + 1e-9
    return float(math.degrees(math.acos(
        float(np.clip(np.dot(vec, np.array([0.0, -1.0])) / nrm, -1.0, 1.0)))))

def check_torso_angle_rdl(hip, shoulder, max_torso_angle=85.0):
    """Check if torso angle is excessive for RDL.
    In a proper RDL the torso hinges forward but shouldn't go
    near-horizontal (>85 deg from vertical). Returns (angle, is_too_far)."""
    torso_ang = torso_angle_to_vertical(hip, shoulder)
    return torso_ang, torso_ang > max_torso_angle

# ===================== LANDMARK EXTRACTION =====================
def _get_all_landmarks(lm):
    PL = mp_pose.PoseLandmark
    def _pt(i): return np.array([lm[i.value].x, lm[i.value].y, lm[i.value].z])
    def _v(i):  return lm[i.value].visibility

    left  = {"shoulder": _pt(PL.LEFT_SHOULDER),  "hip":   _pt(PL.LEFT_HIP),
              "knee":    _pt(PL.LEFT_KNEE),        "ankle": _pt(PL.LEFT_ANKLE),
              "ear":     _pt(PL.LEFT_EAR)}
    right = {"shoulder": _pt(PL.RIGHT_SHOULDER), "hip":   _pt(PL.RIGHT_HIP),
              "knee":    _pt(PL.RIGHT_KNEE),       "ankle": _pt(PL.RIGHT_ANKLE),
              "ear":     _pt(PL.RIGHT_EAR)}

    lv = sum(_v(i) for i in [PL.LEFT_SHOULDER, PL.LEFT_HIP, PL.LEFT_KNEE,
                               PL.LEFT_ANKLE, PL.LEFT_EAR])
    rv = sum(_v(i) for i in [PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE,
                               PL.RIGHT_ANKLE, PL.RIGHT_EAR])
    mid = {k: (left[k] + right[k]) / 2.0 for k in left}
    return {"left": left, "right": right,
            "dominant": left if lv >= rv else right,
            "dominant_vis": max(lv, rv),
            "mid": mid, "left_vis": lv, "right_vis": rv}

def _get_side_landmarks(lm):
    PL = mp_pose.PoseLandmark
    L  = [PL.LEFT_SHOULDER, PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE, PL.LEFT_EAR]
    R  = [PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE, PL.RIGHT_EAR]
    lv = sum(lm[i.value].visibility for i in L)
    rv = sum(lm[i.value].visibility for i in R)
    idx = L if lv >= rv else R
    keys = ["shoulder", "hip", "knee", "ankle", "ear"]
    return {k: np.array([lm[idx[i].value].x, lm[idx[i].value].y])
            for i, k in enumerate(keys)}

# ===================== COMPOSITE MOVEMENT SIGNAL =====================
def compute_movement_signal(all_lm):
    mid = all_lm["mid"]
    dom = all_lm["dominant"]

    hy = mid["hip"][1]
    sy = mid["shoulder"][1]
    ay = mid["ankle"][1]
    bh = abs(ay - sy) + 1e-6

    sd = float(np.clip(((sy - hy) / bh + 0.4) / 0.5, 0.0, 1.0))
    sn = float(np.clip((sy - hy + 0.3) / 0.4, 0.0, 1.0))

    sh2 = dom["shoulder"][:2]
    hi2 = dom["hip"][:2]
    kn2 = dom["knee"][:2]
    an2 = dom["ankle"][:2]

    ta = torso_angle_to_vertical(hi2, sh2)
    ts = float(np.clip(ta / 80.0, 0.0, 1.0))
    ka = angle_deg(hi2, kn2, an2)
    ks = float(np.clip((180.0 - ka) / 40.0, 0.0, 1.0))

    t3  = dom["shoulder"] - dom["hip"]
    t3n = np.linalg.norm(t3) + 1e-9
    t3a = math.degrees(math.acos(
        float(np.clip(np.dot(t3, np.array([0.0, -1.0, 0.0])) / t3n, -1.0, 1.0))))
    t3s = float(np.clip(t3a / 80.0, 0.0, 1.0))

    vr = (min(all_lm["left_vis"], all_lm["right_vis"])
          / (max(all_lm["left_vis"], all_lm["right_vis"]) + 1e-6))

    if   vr > 0.7: w = {"sd": .30, "sn": .20, "ts": .05, "t3": .30, "ks": .15}
    elif vr > 0.4: w = {"sd": .20, "sn": .15, "ts": .20, "t3": .25, "ks": .20}
    else:          w = {"sd": .10, "sn": .10, "ts": .40, "t3": .20, "ks": .20}

    c = float(np.clip(
        w["sd"]*sd + w["sn"]*sn + w["ts"]*ts + w["t3"]*t3s + w["ks"]*ks,
        0.0, 1.0))

    return {"composite": c, "shoulder_drop": sd, "shoulder_y": sn,
            "torso_2d": ts, "torso_3d": t3s,
            "torso_2d_angle": ta, "torso_3d_angle": t3a,
            "knee_signal": ks, "knee_angle": ka,
            "vis_ratio": vr, "weights": w}

# ===================== DEPTH NORMALIZER =====================
class DepthNormalizer:
    def __init__(self):
        self.floor   = 0.50
        self.ceiling = 0.50
        self.n       = 0

    def update(self, raw):
        self.n += 1
        if raw < self.floor:
            self.floor = 0.85 * self.floor + 0.15 * raw
        else:
            self.floor = 0.995 * self.floor + 0.005 * raw

        if raw > self.ceiling:
            self.ceiling = 0.80 * self.ceiling + 0.20 * raw
        else:
            self.ceiling = 0.995 * self.ceiling + 0.005 * raw

        rng = max(0.12, self.ceiling - self.floor)
        return float(np.clip((raw - self.floor) / rng, 0.0, 1.0))

# ===================== REP COUNTER (from working v5.1) =====================
class RepCounter:
    ENTER_THRESHOLD           = 0.30
    EXIT_THRESHOLD            = 0.18
    MIN_PEAK_FOR_REP          = 0.45
    GOOD_DEPTH_PEAK           = 0.58
    MIN_FRAMES_BETWEEN        = 8
    MAX_REP_FRAMES            = 120
    REBOUND_DELTA             = 0.06
    MIN_RETURN_FROM_PEAK      = 0.10
    NORMALIZED_PEAK_THRESHOLD = 0.62
    NORMALIZED_DROP_THRESHOLD = 0.18
    MIN_REP_DURATION_FRAMES   = 10
    MIN_PEAK_DELTA            = 0.12
    MIN_NORMALIZED_PEAK_DELTA = 0.25

    def __init__(self):
        self.state             = "standing"
        self.count             = 0
        self.current_peak      = 0.0
        self.frames_in_state   = 0
        self.last_rep_frame    = -999
        self.smoothed          = 0.0
        self.rep_start_frame   = -1
        self.ascent_valley     = 1.0
        self.rep_start_signal  = 0.0
        self.rep_max_torso_2d  = 0.0
        self.rep_max_torso_3d  = 0.0
        self.rep_min_knee      = 999.0
        self.rep_max_knee      = 0.0
        self.rep_back_issue    = False
        self.rep_back_angle    = 0.0
        self.calibration_signals = []
        self.calibrated        = False
        self.standing_baseline = 0.0
        self.dynamic_floor     = 0.0
        self.dynamic_ceil      = 0.6

    def _smooth(self, raw):
        self.smoothed += 0.35 * (raw - self.smoothed)
        return self.smoothed

    def _normalize(self, s):
        self.dynamic_floor = 0.98 * self.dynamic_floor + 0.02 * min(self.dynamic_floor, s)
        self.dynamic_ceil  = 0.95 * self.dynamic_ceil  + 0.05 * max(self.dynamic_ceil, s)
        return float(np.clip(
            (s - self.dynamic_floor) / max(0.20, self.dynamic_ceil - self.dynamic_floor),
            0.0, 1.0))

    def _calibrate(self, signal):
        if self.calibrated:
            return
        self.calibration_signals.append(signal)
        if len(self.calibration_signals) >= 25:
            self.standing_baseline = float(np.percentile(self.calibration_signals, 40))
            self.dynamic_floor     = self.standing_baseline
            p95 = float(np.percentile(self.calibration_signals, 95))
            # Cap ceil to avoid inflated range from a single early rep
            self.dynamic_ceil      = min(max(self.standing_baseline + 0.40, p95),
                                         self.standing_baseline + 0.65)
            self.ENTER_THRESHOLD   = max(0.20, self.standing_baseline + 0.20)
            self.EXIT_THRESHOLD    = max(0.14, self.standing_baseline + 0.10)
            self.calibrated        = True

    def _start_rep(self, sd, fi, sm):
        self.state            = "descending"
        self.current_peak     = sm
        self.frames_in_state  = 0
        self.rep_start_frame  = fi
        self.ascent_valley    = sm
        self.rep_start_signal = sm
        self.rep_max_torso_2d = sd.get("torso_2d_angle", 0)
        self.rep_max_torso_3d = sd.get("torso_3d_angle", 0)
        self.rep_min_knee     = sd.get("knee_angle", 170)
        self.rep_max_knee     = sd.get("knee_angle", 170)
        self.rep_back_issue   = False
        self.rep_back_angle   = 0.0

    def _norm_peak(self):
        return ((self.current_peak - self.dynamic_floor)
                / max(0.20, self.dynamic_ceil - self.dynamic_floor))

    def _meaningful(self):
        d  = self.current_peak - self.rep_start_signal
        nd = d / max(0.20, self.dynamic_ceil - self.dynamic_floor)
        return d >= self.MIN_PEAK_DELTA or nd >= self.MIN_NORMALIZED_PEAK_DELTA

    def _rep_info(self):
        return {"rep":          self.count,
                "peak_signal":  self.current_peak,
                "good_depth":   self.current_peak >= self.GOOD_DEPTH_PEAK,
                "max_torso_2d": self.rep_max_torso_2d,
                "max_torso_3d": self.rep_max_torso_3d,
                "min_knee":     self.rep_min_knee,
                "max_knee":     self.rep_max_knee,
                "back_issue":   self.rep_back_issue,
                "back_angle":   self.rep_back_angle}

    def _reset(self):
        self.state            = "standing"
        self.frames_in_state  = 0
        self.current_peak     = 0.0
        self.rep_start_frame  = -1
        self.ascent_valley    = 1.0
        self.rep_start_signal = 0.0

    def finalize_pending_rep(self, fi):
        # Disabled: end-of-clip motion (putting bar down, standing up)
        # was being falsely counted as a rep
        return None

    def update(self, sd, fi):
        raw = sd["composite"]
        self._calibrate(raw)
        sm = self._smooth(raw)
        nm = self._normalize(sm)
        self.frames_in_state += 1

        if self.state in ("descending", "ascending"):
            self.rep_max_torso_2d = max(self.rep_max_torso_2d, sd.get("torso_2d_angle", 0))
            self.rep_max_torso_3d = max(self.rep_max_torso_3d, sd.get("torso_3d_angle", 0))
            ka = sd.get("knee_angle", 170)
            self.rep_min_knee = min(self.rep_min_knee, ka)
            self.rep_max_knee = max(self.rep_max_knee, ka)

        if self.state == "standing":
            if sm >= self.ENTER_THRESHOLD or nm >= 0.28:
                self._start_rep(sd, fi, sm)

        elif self.state == "descending":
            if sm > self.current_peak:
                self.current_peak = sm
            if sm < self.current_peak - 0.015 and self.frames_in_state >= 2:
                self.state           = "ascending"
                self.frames_in_state = 0
                self.ascent_valley   = sm
            if (self.rep_start_frame > 0
                    and (fi - self.rep_start_frame) > self.MAX_REP_FRAMES):
                self._reset()

        elif self.state == "ascending":
            self.ascent_valley = min(self.ascent_valley, sm)

            if ((sm <= self.EXIT_THRESHOLD or nm <= 0.28)
                    and (fi - self.last_rep_frame) >= self.MIN_FRAMES_BETWEEN):
                if ((self.current_peak >= self.MIN_PEAK_FOR_REP
                     or self._norm_peak() >= self.NORMALIZED_PEAK_THRESHOLD)
                        and self._meaningful()
                        and (fi - self.rep_start_frame) >= self.MIN_REP_DURATION_FRAMES):
                    self.count += 1
                    self.last_rep_frame = fi
                    info = self._rep_info()
                    self._reset()
                    return info
                self._reset()

            vd  = self.current_peak - self.ascent_valley
            nvd = vd / max(0.20, self.dynamic_ceil - self.dynamic_floor)
            if sm > (self.ascent_valley + self.REBOUND_DELTA) and self.frames_in_state >= 2:
                ok_p = (self.current_peak >= self.MIN_PEAK_FOR_REP
                        or self._norm_peak() >= self.NORMALIZED_PEAK_THRESHOLD)
                ok_r = (vd >= self.MIN_RETURN_FROM_PEAK
                        or nvd >= self.NORMALIZED_DROP_THRESHOLD)
                if (ok_p and ok_r and self._meaningful()
                        and (fi - self.last_rep_frame) >= self.MIN_FRAMES_BETWEEN
                        and (fi - self.rep_start_frame) >= self.MIN_REP_DURATION_FRAMES):
                    self.count += 1
                    self.last_rep_frame = fi
                    info = self._rep_info()
                    self._start_rep(sd, fi, sm)
                    return info

            if sm > self.current_peak:
                self.current_peak    = sm
                self.state           = "descending"
                self.frames_in_state = 0
            if (self.rep_start_frame > 0
                    and (fi - self.rep_start_frame) > self.MAX_REP_FRAMES):
                self._reset()

        return None

# ===================== REP EVALUATION =====================
HINGE_BOTTOM_ANGLE    = 55.0
KNEE_MIN_ANGLE        = 172.0
TORSO_MAX_ANGLE       = 95.0   # only flag if torso goes past horizontal
MIN_SCORE             = 4.0
MAX_SCORE             = 10.0

def _evaluate_rep(rep_result):
    feedback = []
    score    = MAX_SCORE
    best_t   = max(rep_result["max_torso_2d"], rep_result["max_torso_3d"])

    if not rep_result["good_depth"] and best_t < HINGE_BOTTOM_ANGLE:
        feedback.append("Go deeper - hinge more at the hips")
        score -= 2.0
    if rep_result["max_knee"] > KNEE_MIN_ANGLE:
        feedback.append("Bend your knees a bit more")
        score -= 1.5
    if rep_result["back_issue"] and rep_result["back_angle"] > TORSO_MAX_ANGLE:
        feedback.append("Your torso is going too far forward")
        score -= 1.0

    return float(max(MIN_SCORE, min(MAX_SCORE, score))), feedback

# ===================== SNAPSHOT LANDMARKS =====================
def _snapshot_lm(lm):
    """Freeze mediapipe landmarks into a plain list of objects."""
    if lm is None:
        return None
    class _P:
        __slots__ = ("x", "y", "z", "visibility")
        def __init__(self, src):
            self.x = src.x; self.y = src.y
            self.z = src.z; self.visibility = src.visibility
    return [_P(p) for p in lm]

# ===================== VIDEO ROTATION =====================
def _get_rotation_angle(video_path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream_tags=rotate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=10)
        s = r.stdout.strip()
        if s:
            return int(s)
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_streams", "-of", "json", video_path],
            capture_output=True, text=True, timeout=10)
        for st in json.loads(r.stdout).get("streams", []):
            if st.get("codec_type") == "video":
                if "rotate" in st.get("tags", {}):
                    return int(st["tags"]["rotate"])
                for sd in st.get("side_data_list", []):
                    if "rotation" in sd:
                        a = int(float(sd["rotation"]))
                        return 360 + a if a < 0 else a
    except Exception:
        pass
    return 0

def _apply_rotation(frame, rotation):
    if rotation == 90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

# =====================================================================
# PASS 1 — analysis (mediapipe), saves per-frame data
# =====================================================================
def _analysis_pass(video_path, rotation, frame_skip, scale, fps_in):
    effective_skip = frame_skip
    effective_fps  = max(1.0, fps_in / max(1, effective_skip))
    dt             = 1.0 / float(effective_fps)

    rep_counter          = RepCounter()
    depth_norm           = DepthNormalizer()
    good_reps = bad_reps = 0
    all_scores              = []
    rep_reports             = []
    session_feedbacks       = []
    session_feedback_by_cat = {}

    frame_idx     = 0
    depth_pct     = 0.0
    rt_fb_msg     = None
    rt_fb_hold    = 0
    frame_data    = {}
    last_valid_lm = None

    cap = cv2.VideoCapture(video_path)
    t0  = time.time()

    with mp_pose.Pose(model_complexity=0,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        _init_skeleton_data()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if time.time() - t0 > 180:
                print("[RDL] Pass1 timeout", file=sys.stderr, flush=True)
                break

            frame_idx += 1
            if frame_idx % effective_skip != 0:
                continue

            frame = _apply_rotation(frame, rotation)
            work  = (cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                     if scale != 1.0 else frame)

            res = pose.process(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))

            def _store(snap):
                frame_data[frame_idx] = {
                    "lm":   snap,
                    "reps": rep_counter.count,
                    "fb":   rt_fb_msg if rt_fb_hold > 0 else None,
                    "prog": depth_pct,
                }

            if not res.pose_landmarks:
                _store(last_valid_lm)
                if rt_fb_hold > 0:
                    rt_fb_hold -= 1
                continue

            lm   = res.pose_landmarks.landmark
            snap = _snapshot_lm(lm)
            last_valid_lm = snap

            all_lm      = _get_all_landmarks(lm)
            signal_data = compute_movement_signal(all_lm)

            depth_pct = depth_norm.update(signal_data["composite"])

            pts = _get_side_landmarks(lm)
            back_angle, back_bad = check_torso_angle_rdl(
                pts["hip"], pts["shoulder"], TORSO_MAX_ANGLE)
            if rep_counter.state in ("descending", "ascending") and back_bad:
                rep_counter.rep_back_issue = True
                rep_counter.rep_back_angle = max(rep_counter.rep_back_angle, back_angle)

            rep_result = rep_counter.update(signal_data, frame_idx)

            # Debug: log signal every frame to diagnose missed reps
            if frame_idx % (effective_skip * 3) == 0:  # log every ~9th raw frame
                print(f"[RDL-DBG] f={frame_idx} state={rep_counter.state} "
                      f"composite={signal_data['composite']:.3f} "
                      f"smoothed={rep_counter.smoothed:.3f} "
                      f"peak={rep_counter.current_peak:.3f} "
                      f"floor={rep_counter.dynamic_floor:.3f} "
                      f"ceil={rep_counter.dynamic_ceil:.3f} "
                      f"torso2d={signal_data['torso_2d_angle']:.1f} "
                      f"vis={signal_data['vis_ratio']:.2f}",
                      file=sys.stderr, flush=True)
            if rep_result is not None:
                score, feedback = _evaluate_rep(rep_result)
                all_scores.append(score)
                if score >= 9.0: good_reps += 1
                else:            bad_reps  += 1
                session_feedbacks.extend(feedback)
                _, session_feedback_by_cat = pick_strongest_per_category(session_feedbacks)
                vt = ("front/back" if signal_data["vis_ratio"] > 0.7 else
                      ("diagonal"  if signal_data["vis_ratio"] > 0.4 else "side"))
                rep_reports.append({
                    "rep":   rep_result["rep"],
                    "score": round(score, 2),
                    "score_display": display_half_str(score),
                    "label": score_label(score),
                    "feedback": feedback,
                    "metrics": {
                        "peak_signal":        round(rep_result["peak_signal"], 3),
                        "max_torso_2d_angle": round(rep_result["max_torso_2d"], 2),
                        "max_torso_3d_angle": round(rep_result["max_torso_3d"], 2),
                        "min_knee_angle":     round(rep_result["min_knee"], 2),
                        "max_knee_angle":     round(rep_result["max_knee"], 2),
                        "back_angle":         round(rep_result["back_angle"], 2),
                        "view_type":          vt,
                    }
                })
                print(f"[RDL] Rep {rep_result['rep']} score={score:.1f} "
                      f"peak={rep_result['peak_signal']:.3f}",
                      file=sys.stderr, flush=True)
                rt_fb_msg  = pick_strongest_feedback(feedback)
                rt_fb_hold = int(0.7 / dt)

            if rt_fb_hold > 0:
                rt_fb_hold -= 1

            _store(snap)

    # finalize pending rep at end of clip
    rep_result = rep_counter.finalize_pending_rep(frame_idx)
    if rep_result is not None:
        score, feedback = _evaluate_rep(rep_result)
        all_scores.append(score)
        if score >= 9.0: good_reps += 1
        else:            bad_reps  += 1
        session_feedbacks.extend(feedback)
        _, session_feedback_by_cat = pick_strongest_per_category(session_feedbacks)
        rep_reports.append({
            "rep":   rep_result["rep"],
            "score": round(score, 2),
            "score_display": display_half_str(score),
            "label": score_label(score),
            "feedback": feedback,
            "metrics": {
                "peak_signal":        round(rep_result["peak_signal"], 3),
                "max_torso_2d_angle": round(rep_result["max_torso_2d"], 2),
                "max_torso_3d_angle": round(rep_result["max_torso_3d"], 2),
                "min_knee_angle":     round(rep_result["min_knee"], 2),
                "max_knee_angle":     round(rep_result["max_knee"], 2),
                "back_angle":         round(rep_result["back_angle"], 2),
                "view_type":          "end_of_clip",
            }
        })

    cap.release()
    counter = rep_counter.count
    print(f"[RDL] Pass1 done reps={counter}", file=sys.stderr, flush=True)

    avg = float(np.mean(all_scores)) if all_scores else 0.0
    if not np.isfinite(avg):
        avg = 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    if session_feedbacks:
        technique_score = min(technique_score, 9.5)

    feedback_list = (
        dedupe_feedback(list(session_feedback_by_cat.values()))
        if session_feedback_by_cat
        else (["Perfect form! \U0001f525"] if not session_feedbacks
              else dedupe_feedback(session_feedbacks))
    )

    if session_feedback_by_cat:
        if "depth" in session_feedback_by_cat:
            tip = "Focus on pushing your hips back further to reach full depth"
        elif "knees" in session_feedback_by_cat:
            tip = "Romanian deadlifts need a soft knee bend throughout the movement"
        else:
            tip = "Keep your core tight and chest up"
    else:
        tip = "Great technique! Keep building that posterior chain \U0001f4aa"

    return {
        "squat_count":             int(counter),
        "technique_score":         float(technique_score),
        "technique_score_display": str(display_half_str(technique_score)),
        "technique_label":         str(score_label(technique_score)),
        "good_reps":               int(good_reps),
        "bad_reps":                int(bad_reps),
        "feedback":                [str(f) for f in feedback_list],
        "tip":                     str(tip),
        "reps":                    rep_reports,
    }, frame_data, effective_fps

# =====================================================================
# PASS 2 — video rendering, NO mediapipe
# =====================================================================
def _render_pass(video_path, rotation, frame_skip, output_path,
                 out_w, out_h, fps_in, frame_data):
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    cap    = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, effective_fps, (out_w, out_h))
    frame_idx = written = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        frame = _apply_rotation(frame, rotation)
        fd    = frame_data.get(frame_idx)

        if fd is None:
            out.write(draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0))
        else:
            f = frame.copy()
            if fd["lm"] is not None:
                f = draw_body_only(f, fd["lm"])
            out.write(draw_overlay(f, reps=fd["reps"],
                                   feedback=fd["fb"], depth_pct=fd["prog"]))
        written += 1

    cap.release()
    out.release()
    print(f"[RDL] Pass2 done frames_written={written}", file=sys.stderr, flush=True)

# =====================================================================
# PUBLIC ENTRY POINT
# =====================================================================
def run_squat_analysis(video_path,
                       frame_skip=3,
                       scale=0.4,
                       output_path="squat_analyzed.mp4",
                       feedback_path="squat_feedback.txt",
                       fast_mode=False,
                       return_video=True):
    print(f"[RDL] Start fast_mode={fast_mode} return_video={return_video}",
          file=sys.stderr, flush=True)

    if fast_mode:
        return_video = False
    create_video = bool(return_video) and bool(output_path)

    rotation = _get_rotation_angle(video_path)
    print(f"[RDL] rotation={rotation}deg", file=sys.stderr, flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"squat_count": 0, "technique_score": 0.0,
                "technique_score_display": "0", "technique_label": score_label(0),
                "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
                "tip": None, "reps": [], "video_path": "", "feedback_path": feedback_path}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames < 10:
        return {"squat_count": 0, "technique_score": 0.0,
                "technique_score_display": "0", "technique_label": score_label(0),
                "good_reps": 0, "bad_reps": 0,
                "feedback": [f"Invalid video - only {total_frames} frames"],
                "tip": None, "reps": [], "video_path": "", "feedback_path": feedback_path}

    out_w, out_h = (orig_h, orig_w) if rotation in (90, 270) else (orig_w, orig_h)
    print(f"[RDL] {total_frames}fr @ {fps_in:.1f}fps  out={out_w}x{out_h}",
          file=sys.stderr, flush=True)

    # PASS 1 — analysis
    analysis, frame_data, _ = _analysis_pass(
        video_path, rotation, frame_skip, scale, fps_in)

    # Write feedback file
    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {analysis['squat_count']}\n"
                    f"Technique Score: {analysis['technique_score']}/10\n")
            for fb in analysis["feedback"]:
                f.write(f"- {fb}\n")
    except Exception as e:
        print(f"Warning feedback file: {e}")

    # PASS 2 — video rendering
    final_video_path = ""
    if create_video:
        print("[RDL] Pass2 rendering...", file=sys.stderr, flush=True)
        _render_pass(video_path, rotation, frame_skip, output_path,
                     out_w, out_h, fps_in, frame_data)

        encoded = output_path.replace(".mp4", "_encoded.mp4")
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", output_path,
                 "-c:v", "libx264", "-preset", "fast",
                 "-movflags", "+faststart", "-pix_fmt", "yuv420p", encoded],
                capture_output=True, timeout=300)
            print(f"[RDL] ffmpeg rc={proc.returncode}", file=sys.stderr, flush=True)
            if proc.returncode != 0:
                print(f"[RDL] ffmpeg stderr: {proc.stderr.decode(errors='replace')[-500:]}",
                      file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[RDL] ffmpeg exception: {e}", file=sys.stderr, flush=True)

        encoded_ok = os.path.exists(encoded) and os.path.getsize(encoded) > 1000
        if encoded_ok:
            try: os.remove(output_path)
            except Exception: pass
            final_video_path = encoded
        elif os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            final_video_path = output_path

        print(f"[RDL] video='{final_video_path}' "
              f"exists={bool(final_video_path and os.path.exists(final_video_path))}",
              file=sys.stderr, flush=True)

    analysis["video_path"]    = str(final_video_path)
    analysis["feedback_path"] = str(feedback_path)
    return analysis

def run_analysis(video_path,
                 frame_skip=3,
                 scale=0.4,
                 output_path="squat_analyzed.mp4",
                 feedback_path="squat_feedback.txt",
                 fast_mode=False,
                 return_video=True):
    return run_squat_analysis(video_path, frame_skip, scale, output_path, feedback_path, fast_mode, return_video)

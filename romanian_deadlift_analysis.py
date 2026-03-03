# -*- coding: utf-8 -*-
# romanian_deadlift_analysis_fixed.py
# FIXES:
# 1. Consistent rep counting: fast_mode and video mode use same frame_skip/scale
# 2. PersonLocker: tighter thresholds - skeleton won't jump to background people
# 3. Skeleton drawn on ORIGINAL full-res frame (not upscaled small work frame)
# 4. Video rotation handling preserved

import os
import cv2
import math
import numpy as np
import subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

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

_FACE_LMS = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS = tuple()


def _init_skeleton_data():
    global _FACE_LMS, _BODY_CONNECTIONS, _BODY_POINTS
    if not _BODY_CONNECTIONS:
        _FACE_LMS = {
            mp_pose.PoseLandmark.NOSE.value,
            mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value,
            mp_pose.PoseLandmark.LEFT_EYE_OUTER.value, mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,
            mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
            mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
            mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
        }
        _BODY_CONNECTIONS = tuple(
            (a, b) for (a, b) in mp_pose.POSE_CONNECTIONS
            if a not in _FACE_LMS and b not in _FACE_LMS
        )
        _BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))


def _dyn_thickness(h):
    return max(2, int(round(h * 0.003))), max(3, int(round(h * 0.005)))


def draw_body_only(frame, lms, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    line, dot = _dyn_thickness(h)
    for a, b in _BODY_CONNECTIONS:
        pa, pb = lms[a], lms[b]
        ax, ay = int(pa.x * w), int(pa.y * h)
        bx, by = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, line, cv2.LINE_AA)
    for i in _BODY_POINTS:
        p = lms[i]
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), dot, color, -1, cv2.LINE_AA)
    return frame


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


def dedupe_feedback(feedback_list):
    seen = set()
    unique = []
    for fb in feedback_list or []:
        if fb in seen:
            continue
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
        last = lines[-1] + "..."
        while draw.textlength(last, font=font) > max_width and len(last) > 1:
            last = last[:-2] + "..."
        lines[-1] = last
    return lines


def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    h, w, _ = frame.shape
    HD_H = 1080
    hd_scale = HD_H / float(h)
    HD_W = max(1, int(round(w * hd_scale)))

    reps_font_size      = _scaled_font_size(_REF_REPS_FONT_SIZE, HD_H)
    feedback_font_size  = _scaled_font_size(_REF_FEEDBACK_FONT_SIZE, HD_H)
    depth_label_size    = _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, HD_H)
    depth_pct_size      = _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE, HD_H)

    REPS_FONT        = _load_font(FONT_PATH, reps_font_size)
    FEEDBACK_FONT    = _load_font(FONT_PATH, feedback_font_size)
    DEPTH_LABEL_FONT = _load_font(FONT_PATH, depth_label_size)
    DEPTH_PCT_FONT   = _load_font(FONT_PATH, depth_pct_size)

    pct = float(np.clip(depth_pct, 0, 1))
    bg_alpha_val = int(round(255 * BAR_BG_ALPHA))

    ref_h   = max(int(HD_H * 0.06), int(reps_font_size * 1.6))
    radius  = int(ref_h * DONUT_RADIUS_SCALE)
    thick   = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin  = int(12 * hd_scale)
    cx = HD_W - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)

    overlay_np = np.zeros((HD_H, HD_W, 4), dtype=np.uint8)

    pad_x, pad_y = int(10 * hd_scale), int(6 * hd_scale)
    tmp_pil  = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_pil)
    txt  = f"Reps: {int(reps)}"
    tw   = tmp_draw.textlength(txt, font=REPS_FONT)
    box_w = int(tw + 2 * pad_x)
    box_h = int(REPS_FONT.size + 2 * pad_y)
    cv2.rectangle(overlay_np, (0, 0), (box_w, box_h), (0, 0, 0, bg_alpha_val), -1)

    cv2.circle(overlay_np, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    if end_ang != start_ang:
        cv2.ellipse(overlay_np, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                    (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)

    fb_y0 = 0
    fb_lines = []
    fb_pad_x = fb_pad_y = line_gap = line_h = 0
    if feedback:
        safe_margin = max(int(6 * hd_scale), int(HD_H * 0.02))
        fb_pad_x  = int(12 * hd_scale)
        fb_pad_y  = int(8  * hd_scale)
        line_gap  = int(4  * hd_scale)
        max_text_w = int(HD_W - 2 * fb_pad_x - int(20 * hd_scale))
        fb_lines  = _wrap_two_lines(tmp_draw, feedback, FEEDBACK_FONT, max_text_w)
        line_h    = FEEDBACK_FONT.size + int(6 * hd_scale)
        block_h   = 2 * fb_pad_y + len(fb_lines) * line_h + (len(fb_lines) - 1) * line_gap
        fb_y0     = max(0, HD_H - safe_margin - block_h)
        cv2.rectangle(overlay_np, (0, fb_y0), (HD_W, HD_H - safe_margin),
                      (0, 0, 0, bg_alpha_val), -1)

    overlay_pil = Image.fromarray(overlay_np, mode="RGBA")
    draw = ImageDraw.Draw(overlay_pil)
    draw.text((pad_x, pad_y - 1), txt, font=REPS_FONT, fill=(255, 255, 255, 255))

    gap    = max(2, int(radius * 0.10))
    by     = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size) // 2
    lbl    = "DEPTH"
    pct_t  = f"{int(pct * 100)}%"
    lw     = draw.textlength(lbl,  font=DEPTH_LABEL_FONT)
    pw     = draw.textlength(pct_t, font=DEPTH_PCT_FONT)
    draw.text((cx - int(lw // 2), by),
              lbl, font=DEPTH_LABEL_FONT, fill=(255, 255, 255, 255))
    draw.text((cx - int(pw // 2), by + DEPTH_LABEL_FONT.size + gap),
              pct_t, font=DEPTH_PCT_FONT, fill=(255, 255, 255, 255))

    if feedback and fb_lines:
        ty = fb_y0 + fb_pad_y
        for ln in fb_lines:
            tw2 = draw.textlength(ln, font=FEEDBACK_FONT)
            tx  = max(fb_pad_x, (HD_W - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=FEEDBACK_FONT, fill=(255, 255, 255, 255))
            ty += line_h + line_gap

    overlay_rgba  = np.array(overlay_pil)
    overlay_small = cv2.resize(overlay_rgba, (w, h), interpolation=cv2.INTER_AREA)
    alpha         = overlay_small[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr   = overlay_small[:, :, [2, 1, 0]].astype(np.float32)
    result        = frame.astype(np.float32) * (1.0 - alpha) + overlay_bgr * alpha
    return result.astype(np.uint8)


def angle_deg(a, b, c):
    ba  = a - b
    bc  = c - b
    nrm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    return float(math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / nrm, -1.0, 1.0)))))


def torso_angle_to_vertical(hip, shoulder):
    vec = shoulder - hip
    nrm = np.linalg.norm(vec) + 1e-9
    return float(math.degrees(math.acos(float(
        np.clip(np.dot(vec, np.array([0.0, -1.0])) / nrm, -1.0, 1.0)
    ))))


def analyze_back_curvature(shoulder, hip, head_like,
                            max_angle_deg=45.0, min_head_dist_ratio=0.35):
    torso_vec  = shoulder - hip
    head_vec   = head_like - shoulder
    torso_nrm  = np.linalg.norm(torso_vec) + 1e-9
    head_nrm   = np.linalg.norm(head_vec)  + 1e-9
    if head_nrm < (min_head_dist_ratio * torso_nrm):
        return 0.0, False
    cosang = float(np.clip(np.dot(torso_vec, head_vec) / (torso_nrm * head_nrm), -1.0, 1.0))
    ang    = math.degrees(math.acos(cosang))
    return ang, ang > max_angle_deg


def _get_all_landmarks(lm):
    PL = mp_pose.PoseLandmark

    def _pt(idx):
        return np.array([lm[idx.value].x, lm[idx.value].y, lm[idx.value].z])

    def _vis(idx):
        return lm[idx.value].visibility

    left = {
        "shoulder": _pt(PL.LEFT_SHOULDER),  "hip":   _pt(PL.LEFT_HIP),
        "knee":     _pt(PL.LEFT_KNEE),       "ankle": _pt(PL.LEFT_ANKLE),
        "ear":      _pt(PL.LEFT_EAR),
    }
    right = {
        "shoulder": _pt(PL.RIGHT_SHOULDER), "hip":   _pt(PL.RIGHT_HIP),
        "knee":     _pt(PL.RIGHT_KNEE),      "ankle": _pt(PL.RIGHT_ANKLE),
        "ear":      _pt(PL.RIGHT_EAR),
    }
    left_vis  = sum(_vis(i) for i in [PL.LEFT_SHOULDER,  PL.LEFT_HIP,  PL.LEFT_KNEE,  PL.LEFT_ANKLE,  PL.LEFT_EAR])
    right_vis = sum(_vis(i) for i in [PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE, PL.RIGHT_EAR])
    dominant  = left if left_vis >= right_vis else right
    mid       = {k: (left[k] + right[k]) / 2.0 for k in left}
    return {
        "left": left, "right": right, "dominant": dominant,
        "dominant_vis": max(left_vis, right_vis),
        "mid": mid, "left_vis": left_vis, "right_vis": right_vis,
    }


def _get_side_landmarks(lm):
    PL   = mp_pose.PoseLandmark
    L    = [PL.LEFT_SHOULDER,  PL.LEFT_HIP,  PL.LEFT_KNEE,  PL.LEFT_ANKLE,  PL.LEFT_EAR]
    R    = [PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE, PL.RIGHT_EAR]
    lv   = sum(lm[i.value].visibility for i in L)
    rv   = sum(lm[i.value].visibility for i in R)
    idxs = L if lv >= rv else R
    keys = ["shoulder", "hip", "knee", "ankle", "ear"]
    return {k: np.array([lm[idxs[i].value].x, lm[idxs[i].value].y]) for i, k in enumerate(keys)}


def compute_movement_signal(all_lm):
    mid = all_lm["mid"]
    dom = all_lm["dominant"]

    hip_y      = mid["hip"][1]
    shoulder_y = mid["shoulder"][1]
    ankle_y    = mid["ankle"][1]
    body_h     = abs(ankle_y - shoulder_y) + 1e-6

    sd_norm  = float(np.clip(((shoulder_y - hip_y) / body_h + 0.4) / 0.5, 0.0, 1.0))
    sy_norm  = float(np.clip((shoulder_y - hip_y + 0.3) / 0.4,            0.0, 1.0))

    sh2d   = dom["shoulder"][:2]
    hip2d  = dom["hip"][:2]
    kn2d   = dom["knee"][:2]
    ank2d  = dom["ankle"][:2]

    torso_ang   = torso_angle_to_vertical(hip2d, sh2d)
    torso_sig   = float(np.clip(torso_ang / 80.0,          0.0, 1.0))
    knee_ang    = angle_deg(hip2d, kn2d, ank2d)
    knee_sig    = float(np.clip((180.0 - knee_ang) / 40.0, 0.0, 1.0))

    t3d      = dom["shoulder"] - dom["hip"]
    t3d_nrm  = np.linalg.norm(t3d) + 1e-9
    cos3d    = float(np.clip(np.dot(t3d, np.array([0.0, -1.0, 0.0])) / t3d_nrm, -1.0, 1.0))
    t3d_ang  = math.degrees(math.acos(cos3d))
    t3d_sig  = float(np.clip(t3d_ang / 80.0, 0.0, 1.0))

    vis_ratio = (min(all_lm["left_vis"], all_lm["right_vis"])
                 / (max(all_lm["left_vis"], all_lm["right_vis"]) + 1e-6))

    if vis_ratio > 0.7:
        w = {"sd": 0.30, "sy": 0.20, "t2": 0.05, "t3": 0.30, "kn": 0.15}
    elif vis_ratio > 0.4:
        w = {"sd": 0.20, "sy": 0.15, "t2": 0.20, "t3": 0.25, "kn": 0.20}
    else:
        w = {"sd": 0.10, "sy": 0.10, "t2": 0.40, "t3": 0.20, "kn": 0.20}

    composite = float(np.clip(
        w["sd"] * sd_norm + w["sy"] * sy_norm +
        w["t2"] * torso_sig + w["t3"] * t3d_sig + w["kn"] * knee_sig,
        0.0, 1.0
    ))
    return {
        "composite":       composite,
        "shoulder_drop":   sd_norm,
        "shoulder_y":      sy_norm,
        "torso_2d":        torso_sig,
        "torso_3d":        t3d_sig,
        "torso_2d_angle":  torso_ang,
        "torso_3d_angle":  t3d_ang,
        "knee_signal":     knee_sig,
        "knee_angle":      knee_ang,
        "vis_ratio":       vis_ratio,
        "weights":         w,
    }


class RepCounter:
    ENTER_THRESHOLD          = 0.25
    EXIT_THRESHOLD           = 0.15
    MIN_PEAK_FOR_REP         = 0.40
    GOOD_DEPTH_PEAK          = 0.55
    MIN_FRAMES_BETWEEN       = 5
    MAX_REP_FRAMES           = 120
    REBOUND_DELTA            = 0.015
    MIN_RETURN_FROM_PEAK     = 0.04
    NORMALIZED_PEAK_THRESHOLD= 0.58
    NORMALIZED_DROP_THRESHOLD= 0.10
    MIN_REP_DURATION_FRAMES  = 6
    MIN_PEAK_DELTA           = 0.07
    MIN_NORMALIZED_PEAK_DELTA= 0.18

    def __init__(self):
        self.state            = "standing"
        self.count            = 0
        self.current_peak     = 0.0
        self.frames_in_state  = 0
        self.last_rep_frame   = -999
        self.smoothed         = 0.0
        self.rep_start_frame  = -1
        self.ascent_valley    = 1.0
        self.rep_start_signal = 0.0
        self.rep_max_torso_2d = 0.0
        self.rep_max_torso_3d = 0.0
        self.rep_min_knee     = 999.0
        self.rep_max_knee     = 0.0
        self.rep_back_issue   = False
        self.rep_back_angle   = 0.0
        self.calibration_signals = []
        self.calibrated       = False
        self.standing_baseline= 0.0
        self.dynamic_floor    = 0.0
        self.dynamic_ceil     = 0.6

    def _smooth(self, raw):
        self.smoothed += 0.35 * (raw - self.smoothed)
        return self.smoothed

    def _normalize(self, s):
        self.dynamic_floor = 0.98 * self.dynamic_floor + 0.02 * min(self.dynamic_floor, s)
        self.dynamic_ceil  = 0.95 * self.dynamic_ceil  + 0.05 * max(self.dynamic_ceil,  s)
        rng = max(0.20, self.dynamic_ceil - self.dynamic_floor)
        return float(np.clip((s - self.dynamic_floor) / rng, 0.0, 1.0))

    def _calibrate(self, signal):
        if self.calibrated:
            return
        self.calibration_signals.append(signal)
        if len(self.calibration_signals) >= 15:
            self.standing_baseline  = float(np.percentile(self.calibration_signals, 25))
            self.dynamic_floor      = self.standing_baseline
            self.dynamic_ceil       = max(self.standing_baseline + 0.35,
                                          float(np.percentile(self.calibration_signals, 90)))
            self.ENTER_THRESHOLD    = max(0.15, self.standing_baseline + 0.15)
            self.EXIT_THRESHOLD     = max(0.10, self.standing_baseline + 0.08)
            self.calibrated         = True

    def _start_rep(self, sd, frame_idx, smoothed):
        self.state            = "descending"
        self.current_peak     = smoothed
        self.frames_in_state  = 0
        self.rep_start_frame  = frame_idx
        self.ascent_valley    = smoothed
        self.rep_start_signal = smoothed
        self.rep_max_torso_2d = sd.get("torso_2d_angle", 0)
        self.rep_max_torso_3d = sd.get("torso_3d_angle", 0)
        self.rep_min_knee     = sd.get("knee_angle", 170)
        self.rep_max_knee     = sd.get("knee_angle", 170)
        self.rep_back_issue   = False
        self.rep_back_angle   = 0.0

    def _norm_peak(self):
        return (self.current_peak - self.dynamic_floor) / max(0.20, self.dynamic_ceil - self.dynamic_floor)

    def _meaningful(self):
        d  = self.current_peak - self.rep_start_signal
        nd = d / max(0.20, self.dynamic_ceil - self.dynamic_floor)
        return d >= self.MIN_PEAK_DELTA or nd >= self.MIN_NORMALIZED_PEAK_DELTA

    def _rep_info(self):
        return {
            "rep":        self.count,
            "peak_signal":self.current_peak,
            "good_depth": self.current_peak >= self.GOOD_DEPTH_PEAK,
            "max_torso_2d": self.rep_max_torso_2d,
            "max_torso_3d": self.rep_max_torso_3d,
            "min_knee":   self.rep_min_knee,
            "max_knee":   self.rep_max_knee,
            "back_issue": self.rep_back_issue,
            "back_angle": self.rep_back_angle,
        }

    def _reset(self):
        self.state           = "standing"
        self.frames_in_state = 0
        self.current_peak    = 0.0
        self.rep_start_frame = -1
        self.ascent_valley   = 1.0
        self.rep_start_signal= 0.0

    def finalize_pending_rep(self, frame_idx):
        if self.state != "ascending":
            return None
        vd  = self.current_peak - self.ascent_valley
        nvd = vd / max(0.20, self.dynamic_ceil - self.dynamic_floor)
        ok_peak   = self.current_peak >= self.MIN_PEAK_FOR_REP or self._norm_peak() >= self.NORMALIZED_PEAK_THRESHOLD
        ok_return = vd >= self.MIN_RETURN_FROM_PEAK * 0.7   or nvd >= self.NORMALIZED_DROP_THRESHOLD * 0.8
        if (ok_peak and ok_return and self._meaningful()
                and (frame_idx - self.last_rep_frame)   >= self.MIN_FRAMES_BETWEEN
                and (frame_idx - self.rep_start_frame)  >= self.MIN_REP_DURATION_FRAMES):
            self.count += 1
            self.last_rep_frame = frame_idx
            info = self._rep_info()
            self._reset()
            return info
        return None

    def update(self, sd, frame_idx):
        raw      = sd["composite"]
        self._calibrate(raw)
        smoothed = self._smooth(raw)
        norm     = self._normalize(smoothed)
        self.frames_in_state += 1

        if self.state in ("descending", "ascending"):
            self.rep_max_torso_2d = max(self.rep_max_torso_2d, sd.get("torso_2d_angle", 0))
            self.rep_max_torso_3d = max(self.rep_max_torso_3d, sd.get("torso_3d_angle", 0))
            ka = sd.get("knee_angle", 170)
            self.rep_min_knee = min(self.rep_min_knee, ka)
            self.rep_max_knee = max(self.rep_max_knee, ka)

        if self.state == "standing":
            if smoothed >= self.ENTER_THRESHOLD or norm >= 0.34:
                self._start_rep(sd, frame_idx, smoothed)

        elif self.state == "descending":
            if smoothed > self.current_peak:
                self.current_peak = smoothed
            if smoothed < self.current_peak - 0.015 and self.frames_in_state >= 2:
                self.state = "ascending"
                self.frames_in_state = 0
                self.ascent_valley   = smoothed
            if self.rep_start_frame > 0 and (frame_idx - self.rep_start_frame) > self.MAX_REP_FRAMES:
                self._reset()

        elif self.state == "ascending":
            self.ascent_valley = min(self.ascent_valley, smoothed)

            if (smoothed <= self.EXIT_THRESHOLD or norm <= 0.28) and (frame_idx - self.last_rep_frame) >= self.MIN_FRAMES_BETWEEN:
                if ((self.current_peak >= self.MIN_PEAK_FOR_REP or self._norm_peak() >= self.NORMALIZED_PEAK_THRESHOLD)
                        and self._meaningful()
                        and (frame_idx - self.rep_start_frame) >= self.MIN_REP_DURATION_FRAMES):
                    self.count += 1
                    self.last_rep_frame = frame_idx
                    info = self._rep_info()
                    self._reset()
                    return info
                self._reset()

            vd  = self.current_peak - self.ascent_valley
            nvd = vd / max(0.20, self.dynamic_ceil - self.dynamic_floor)
            if smoothed > (self.ascent_valley + self.REBOUND_DELTA) and self.frames_in_state >= 2:
                ok_p = self.current_peak >= self.MIN_PEAK_FOR_REP or self._norm_peak() >= self.NORMALIZED_PEAK_THRESHOLD
                ok_r = vd >= self.MIN_RETURN_FROM_PEAK or nvd >= self.NORMALIZED_DROP_THRESHOLD
                if (ok_p and ok_r and self._meaningful()
                        and (frame_idx - self.last_rep_frame)  >= self.MIN_FRAMES_BETWEEN
                        and (frame_idx - self.rep_start_frame) >= self.MIN_REP_DURATION_FRAMES):
                    self.count += 1
                    self.last_rep_frame = frame_idx
                    info = self._rep_info()
                    self._start_rep(sd, frame_idx, smoothed)
                    return info

            if smoothed > self.current_peak:
                self.current_peak = smoothed
                self.state        = "descending"
                self.frames_in_state = 0
            if self.rep_start_frame > 0 and (frame_idx - self.rep_start_frame) > self.MAX_REP_FRAMES:
                self._reset()

        return None


HINGE_BOTTOM_ANGLE = 55.0
KNEE_MIN_ANGLE     = 172.0
KNEE_MAX_ANGLE     = 125.0
BACK_MAX_ANGLE     = 60.0
MIN_SCORE          = 4.0
MAX_SCORE          = 10.0
PROG_ALPHA         = 0.3


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
        import json
        for stream in json.loads(r.stdout).get("streams", []):
            if stream.get("codec_type") == "video":
                if "rotate" in stream.get("tags", {}):
                    return int(stream["tags"]["rotate"])
                for sd in stream.get("side_data_list", []):
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


class PersonLocker:
    """
    FIXED vs original:
      MAX_JUMP_RATIO      0.60  ->  0.35   tighter spatial lock
      RELOCK_AFTER_REJECTS  8  ->  20     much harder to snap to new person
      MIN_VIS             0.30  ->  0.45   require confident detection
      WARMUP_FRAMES          3  ->   5
      NEW: size_ratio check — rejects detections <50% of locked person's size
           (background people appear smaller in frame)
    """

    def __init__(self):
        self.locked_centroid     = None
        self.locked_body_size    = None
        self.consecutive_rejects = 0
        self.frames_since_lock   = 0
        self.lock_established    = False
        self.RELOCK_AFTER_REJECTS = 20
        self.MAX_JUMP_RATIO       = 0.35
        self.MIN_JUMP_THRESHOLD   = 0.10
        self.WARMUP_FRAMES        = 5
        self.MIN_VIS              = 0.45
        self.MIN_SIZE_RATIO       = 0.50

    def _centroid_size_vis(self, lm):
        PL   = mp_pose.PoseLandmark
        lh   = lm[PL.LEFT_HIP.value];      rh  = lm[PL.RIGHT_HIP.value]
        ls   = lm[PL.LEFT_SHOULDER.value]; rs  = lm[PL.RIGHT_SHOULDER.value]
        la   = lm[PL.LEFT_ANKLE.value];    ra  = lm[PL.RIGHT_ANKLE.value]
        cx   = (lh.x + rh.x) / 2.0;       cy  = (lh.y + rh.y) / 2.0
        sh_y = (ls.y + rs.y) / 2.0;       ak_y= (la.y + ra.y) / 2.0
        vis  = (lh.visibility + rh.visibility + ls.visibility + rs.visibility) / 4.0
        return (cx, cy), abs(ak_y - sh_y), vis

    def check(self, lm):
        centroid, body_size, vis = self._centroid_size_vis(lm)

        if vis < self.MIN_VIS:
            return False

        if not self.lock_established:
            self.locked_centroid  = centroid
            self.locked_body_size = body_size
            self.frames_since_lock += 1
            if self.frames_since_lock >= self.WARMUP_FRAMES:
                self.lock_established = True
            return True

        # Reject if detected body is much smaller than locked person
        if self.locked_body_size and body_size > 0.05:
            if body_size / (self.locked_body_size + 1e-6) < self.MIN_SIZE_RATIO:
                self.consecutive_rejects += 1
                return False

        dx   = centroid[0] - self.locked_centroid[0]
        dy   = centroid[1] - self.locked_centroid[1]
        dist = math.sqrt(dx * dx + dy * dy)
        ref  = max(self.locked_body_size or 0.3, 0.15)
        thr  = max(self.MIN_JUMP_THRESHOLD, ref * self.MAX_JUMP_RATIO)

        if dist <= thr:
            a = 0.3
            self.locked_centroid  = (
                self.locked_centroid[0] + a * dx,
                self.locked_centroid[1] + a * dy,
            )
            self.locked_body_size += a * (body_size - self.locked_body_size)
            self.consecutive_rejects = 0
            return True

        self.consecutive_rejects += 1
        if self.consecutive_rejects >= self.RELOCK_AFTER_REJECTS:
            self.locked_centroid  = centroid
            self.locked_body_size = body_size
            self.consecutive_rejects = 0
            return True
        return False


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
    elif rep_result["min_knee"] < KNEE_MAX_ANGLE:
        feedback.append("Too much knee bend")
        score -= 2.0
    if rep_result["back_issue"] and rep_result["back_angle"] > (BACK_MAX_ANGLE + 5.0):
        feedback.append("Try to keep your back neutral")
        score -= 1.0

    return float(max(MIN_SCORE, min(MAX_SCORE, score))), feedback


def run_romanian_deadlift_analysis(video_path,
                                   frame_skip=3,
                                   scale=0.4,
                                   output_path="romanian_deadlift_analyzed.mp4",
                                   feedback_path="romanian_deadlift_feedback.txt",
                                   return_video=True,
                                   fast_mode=False):
    """
    Romanian Deadlift analysis — fixed version.

    FIX 1: Consistent rep counting — fast_mode and video mode use SAME
            frame_skip and scale (only model_complexity differs).
    FIX 2: PersonLocker with tighter thresholds — skeleton stays on main
            person even when background people walk through the frame.
    FIX 3: Skeleton drawn on ORIGINAL full-resolution frame (not on the
            upscaled-from-small work frame — which was blurry).
    FIX 4: Video rotation metadata handled before any processing.
    """
    import sys
    print(f"[RDL] Starting: fast_mode={fast_mode}, return_video={return_video}",
          file=sys.stderr, flush=True)

    mp_pose_mod = mp.solutions.pose

    if fast_mode:
        return_video = False
    create_video = bool(return_video) and bool(output_path)

    rotation = _get_rotation_angle(video_path)
    print(f"[RDL] Rotation metadata: {rotation}deg", file=sys.stderr, flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"squat_count": 0, "technique_score": 0.0,
                "technique_score_display": "0", "technique_label": score_label(0),
                "good_reps": 0, "bad_reps": 0,
                "feedback": ["Could not open video"],
                "reps": [], "video_path": "", "feedback_path": feedback_path}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25

    if total_frames < 10:
        cap.release()
        return {"squat_count": 0, "technique_score": 0.0,
                "technique_score_display": "0", "technique_label": score_label(0),
                "good_reps": 0, "bad_reps": 0,
                "feedback": [f"Invalid video – only {total_frames} frames"],
                "reps": [], "video_path": "", "feedback_path": feedback_path}

    # FIX 1: Both modes share the same frame_skip and scale.
    #         Only model_complexity (speed vs accuracy) differs.
    effective_frame_skip = frame_skip
    effective_scale      = scale
    model_complexity     = 0 if fast_mode else 1

    effective_fps = max(1.0, fps_in / max(1, effective_frame_skip))
    dt            = 1.0 / float(effective_fps)

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = (orig_h, orig_w) if rotation in (90, 270) else (orig_w, orig_h)

    print(f"[RDL] {total_frames} frames @ {fps_in:.1f} fps, output {out_w}x{out_h}",
          file=sys.stderr, flush=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = None

    rep_counter    = RepCounter()
    person_locker  = PersonLocker()
    last_valid_lm  = None
    good_reps = bad_reps = 0
    all_scores             = []
    rep_reports            = []
    session_feedbacks      = []
    session_feedback_by_cat= {}

    frame_idx  = 0
    prog       = 0.0
    rt_fb_msg  = None
    rt_fb_hold = 0

    import time
    t0 = time.time()

    with mp_pose_mod.Pose(model_complexity=model_complexity,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
        _init_skeleton_data()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if time.time() - t0 > 180:
                print("[RDL] Timeout", file=sys.stderr, flush=True)
                break

            frame_idx += 1
            if frame_idx % effective_frame_skip != 0:
                continue

            # Rotate so frame is always out_w x out_h
            frame = _apply_rotation(frame, rotation)

            # Scale down only for pose detection (speed)
            work = (cv2.resize(frame, (0, 0), fx=effective_scale, fy=effective_scale)
                    if effective_scale != 1.0 else frame)

            if create_video and out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (out_w, out_h))

            res = pose.process(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))

            # Helper: write one frame to video
            def _write(lm_draw):
                if out is None:
                    return
                # FIX 3: copy ORIGINAL full-res frame, draw skeleton on that
                f = frame.copy()
                if lm_draw is not None:
                    f = draw_body_only(f, lm_draw)
                out.write(draw_overlay(f, reps=rep_counter.count,
                                       feedback=(rt_fb_msg if rt_fb_hold > 0 else None),
                                       depth_pct=prog))

            if not res.pose_landmarks:
                if create_video:
                    _write(last_valid_lm)
                continue

            lm = res.pose_landmarks.landmark

            # FIX 2: PersonLocker — reject jumps to background people
            if not person_locker.check(lm):
                if create_video:
                    _write(last_valid_lm)   # keep showing last valid person
                continue

            last_valid_lm = lm

            all_lm      = _get_all_landmarks(lm)
            signal_data = compute_movement_signal(all_lm)
            prog        = prog + PROG_ALPHA * (signal_data["composite"] - prog)

            pts = _get_side_landmarks(lm)
            back_angle, back_bad = analyze_back_curvature(
                pts["shoulder"], pts["hip"], pts["ear"], BACK_MAX_ANGLE)
            if rep_counter.state in ("descending", "ascending") and back_bad:
                rep_counter.rep_back_issue = True
                rep_counter.rep_back_angle = max(rep_counter.rep_back_angle, back_angle)

            rep_result = rep_counter.update(signal_data, frame_idx)

            if rep_result is not None:
                score, feedback = _evaluate_rep(rep_result)
                all_scores.append(score)
                if score >= 9.0:
                    good_reps += 1
                else:
                    bad_reps  += 1

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
                        "peak_signal":       round(rep_result["peak_signal"], 3),
                        "max_torso_2d_angle":round(rep_result["max_torso_2d"], 2),
                        "max_torso_3d_angle":round(rep_result["max_torso_3d"], 2),
                        "min_knee_angle":    round(rep_result["min_knee"], 2),
                        "max_knee_angle":    round(rep_result["max_knee"], 2),
                        "back_angle":        round(rep_result["back_angle"], 2),
                        "view_type":         vt,
                    },
                })
                print(f"[RDL] Rep {rep_result['rep']}: score={score:.1f}  "
                      f"peak={rep_result['peak_signal']:.3f}  view={vt}",
                      file=sys.stderr, flush=True)

                rt_fb_msg  = pick_strongest_feedback(feedback)
                rt_fb_hold = int(0.7 / dt)

            if rt_fb_hold > 0:
                rt_fb_hold -= 1

            if create_video:
                _write(lm)

    # Finalise last rep if video ended mid-ascent
    rep_result = rep_counter.finalize_pending_rep(frame_idx)
    if rep_result is not None:
        score, feedback = _evaluate_rep(rep_result)
        all_scores.append(score)
        if score >= 9.0: good_reps += 1
        else:             bad_reps  += 1
        session_feedbacks.extend(feedback)
        _, session_feedback_by_cat = pick_strongest_per_category(session_feedbacks)
        rep_reports.append({
            "rep":   rep_result["rep"],
            "score": round(score, 2),
            "score_display": display_half_str(score),
            "label": score_label(score),
            "feedback": feedback,
            "metrics": {
                "peak_signal":       round(rep_result["peak_signal"], 3),
                "max_torso_2d_angle":round(rep_result["max_torso_2d"], 2),
                "max_torso_3d_angle":round(rep_result["max_torso_3d"], 2),
                "min_knee_angle":    round(rep_result["min_knee"], 2),
                "max_knee_angle":    round(rep_result["max_knee"], 2),
                "back_angle":        round(rep_result["back_angle"], 2),
                "view_type":         "end_of_clip",
            },
        })

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    counter = rep_counter.count
    print(f"[RDL] Done — reps={counter}", file=sys.stderr, flush=True)

    avg = float(np.mean(all_scores)) if all_scores else 0.0
    if not np.isfinite(avg):
        avg = 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    if session_feedbacks:
        technique_score = min(technique_score, 9.5)

    feedback_list = (
        dedupe_feedback(list(session_feedback_by_cat.values()))
        if session_feedback_by_cat
        else (["Perfect form! 🔥"] if not session_feedbacks
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
        tip = "Great technique! Keep building that posterior chain 💪"

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            for fb in feedback_list:
                f.write(f"- {fb}\n")
    except Exception as e:
        print(f"Warning (feedback file): {e}")

    final_video_path = ""
    if create_video and output_path:
        encoded = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", output_path,
                 "-c:v", "libx264", "-preset", "fast",
                 "-movflags", "+faststart", "-pix_fmt", "yuv420p", encoded],
                check=False, capture_output=True)
            if os.path.exists(output_path) and os.path.exists(encoded):
                os.remove(output_path)
            final_video_path = (encoded if os.path.exists(encoded)
                                else output_path if os.path.exists(output_path) else "")
        except Exception as e:
            print(f"Warning (ffmpeg): {e}")
            final_video_path = output_path if os.path.exists(output_path) else ""

    return {
        "squat_count":            int(counter),
        "technique_score":        float(technique_score),
        "technique_score_display":str(display_half_str(technique_score)),
        "technique_label":        str(score_label(technique_score)),
        "good_reps":              int(good_reps),
        "bad_reps":               int(bad_reps),
        "feedback":               [str(f) for f in feedback_list],
        "tip":                    str(tip),
        "reps":                   rep_reports,
        "video_path":             str(final_video_path),
        "feedback_path":          str(feedback_path),
    }


def run_analysis(*args, **kwargs):
    return run_romanian_deadlift_analysis(*args, **kwargs)

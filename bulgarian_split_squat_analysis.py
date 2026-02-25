# -*- coding: utf-8 -*-
# bulgarian_split_squat_analysis.py
# ============================================================================
# Production-ready Bulgarian Split Squat analyzer.
#
# DESIGNED TO WORK WITH: frame_skip=3, scale=0.4 (from app.py)
# This means: ~8-10 effective FPS, small image → noisy landmarks.
# Thresholds are tuned for these real-world conditions.
#
# FIXES vs original:
# 1. draw_overlay: fixed NameError (wrap_to_two_lines → _wrap_two_lines)
# 2. draw_overlay: works at frame resolution (no 1080p upscale/downscale)
#    with a per-resolution cache for static layers — same approach as pullup.
# 3. Rotation: reads video metadata once and rotates pixels before processing,
#    so portrait/landscape videos always display upright.
# ============================================================================

import os, math, subprocess, re, collections
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

mp_pose = mp.solutions.pose

# ========================== CONSTANTS ==========================
ANGLE_DOWN_THRESH    = 105
ANGLE_UP_THRESH      = 150
MIN_RANGE_DELTA_DEG  = 10
MIN_DOWN_FRAMES      = 2
FAST_REP_MIN_DEPTH_DELTA = 18

GOOD_REP_MIN_SCORE   = 8.0
PERFECT_MIN_KNEE     = 70
TORSO_LEAN_MIN       = 112
TORSO_MARGIN_DEG     = 3
TORSO_BAD_MIN_FRAMES = 8
VALGUS_X_TOL         = 0.03
VALGUS_BAD_MIN_FRAMES = 2

EMA_ALPHA            = 0.7
REP_DEBOUNCE_FRAMES  = 3

HIP_VEL_THRESH_PCT   = 0.018
ANKLE_VEL_THRESH_PCT  = 0.022
MOTION_EMA_ALPHA     = 0.55
MOVEMENT_CLEAR_FRAMES = 1

NOPOSE_STOP_SEC      = 1.5
NO_MOVEMENT_STOP_SEC = 1.5

# Overlay style
BAR_BG_ALPHA          = 0.55
FONT_PATH             = "Roboto-VariableFont_wdth,wght.ttf"
RT_FB_HOLD_SEC        = 0.8
DONUT_RADIUS_SCALE    = 0.72
DONUT_THICKNESS_FRAC  = 0.28
DEPTH_COLOR           = (40, 200, 80)
DEPTH_RING_BG         = (70, 70, 70)

# Reference sizes (same as pullup so overlay looks identical)
_REF_H                = 480.0
_REF_REPS_FONT_SIZE   = 28
_REF_FEEDBACK_FONT_SIZE = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE   = 18


# ========================== ROTATION ==========================

def _read_source_rotation_degrees(video_path):
    """Read rotation metadata (0/90/180/270). OpenCV ignores this metadata."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        if proc.returncode == 0 and proc.stdout:
            for line in proc.stdout.splitlines():
                m = re.search(r"-?\d+(?:\.\d+)?", line.strip())
                if not m: continue
                try:
                    rot = int(round(float(m.group(0)))) % 360
                    return min((0, 90, 180, 270), key=lambda x: abs(x - rot))
                except Exception:
                    continue
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", video_path],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        m = re.search(r"rotate\s*:\s*(-?\d+)", proc.stderr or "")
        if m:
            rot = int(m.group(1)) % 360
            return min((0, 90, 180, 270), key=lambda x: abs(x - rot))
    except Exception:
        pass
    return 0


def _rotate_frame(frame, rotation_degrees):
    """
    Rotate pixels to compensate for video metadata rotation.
      90  → rotate CCW 90°
      180 → rotate 180°
      270 → rotate CW 90°
      0   → no-op
    """
    if rotation_degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


# ========================== OVERLAY CACHE ==========================
# Pre-build static background layers once per (frame_w, frame_h).
# Only dynamic elements (text, arc) are drawn each frame.

_OVERLAY_CACHE = {}

def _load_font(p, s):
    try:
        return ImageFont.truetype(p, s)
    except Exception:
        pass
    for fallback in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(fallback, s)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=s)
    except TypeError:
        return ImageFont.load_default()

def _scaled_font_size(ref_size, canvas_h):
    return max(10, int(round(ref_size * (canvas_h / _REF_H))))

def _get_overlay_cache(w, h):
    key = (int(w), int(h))
    cached = _OVERLAY_CACHE.get(key)
    if cached is not None:
        return cached

    fw, fh = int(w), int(h)

    reps_font_size        = _scaled_font_size(_REF_REPS_FONT_SIZE,        fh)
    feedback_font_size    = _scaled_font_size(_REF_FEEDBACK_FONT_SIZE,    fh)
    depth_label_font_size = _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, fh)
    depth_pct_font_size   = _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE,   fh)

    reps_font        = _load_font(FONT_PATH, reps_font_size)
    feedback_font    = _load_font(FONT_PATH, feedback_font_size)
    depth_label_font = _load_font(FONT_PATH, depth_label_font_size)
    depth_pct_font   = _load_font(FONT_PATH, depth_pct_font_size)

    _tmp  = Image.new("RGBA", (1, 1))
    _tdraw = ImageDraw.Draw(_tmp)
    sample_txt = "Reps: 00"
    pad_x = max(6, int(fw * 0.013))
    pad_y = max(4, int(fh * 0.010))
    tw = _tdraw.textlength(sample_txt, font=reps_font)
    rep_box_w = int(tw + 2 * pad_x)
    rep_box_h = int(reps_font.size + 2 * pad_y)

    ref_h_donut = max(int(fh * 0.06), int(reps_font_size * 1.6))
    radius  = int(ref_h_donut * DONUT_RADIUS_SCALE)
    thick   = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin  = max(8, int(fw * 0.016))
    cx      = fw - margin - radius
    cy      = max(ref_h_donut + radius // 8, radius + thick // 2 + 2)

    safe_margin = max(4, int(fh * 0.012))
    fb_pad_x    = max(8, int(fw * 0.016))
    fb_pad_y    = max(4, int(fh * 0.010))
    line_gap    = max(2, int(fh * 0.006))
    max_text_w  = fw - 2 * fb_pad_x - max(8, int(fw * 0.015))
    line_h      = feedback_font.size + max(4, int(fh * 0.008))
    block_h     = 2 * fb_pad_y + 2 * line_h + line_gap
    fb_y0       = max(0, fh - safe_margin - block_h)
    fb_y1       = fh - safe_margin

    bg_alpha_val = int(round(255 * BAR_BG_ALPHA))

    # Static rep-count background
    rep_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.rectangle(rep_bg, (0, 0), (rep_box_w, rep_box_h), (0, 0, 0, bg_alpha_val), -1)

    # Static feedback background
    fb_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.rectangle(fb_bg, (0, fb_y0), (fw, fb_y1), (0, 0, 0, bg_alpha_val), -1)

    # Static donut ring background (grey circle)
    donut_bg = np.zeros((fh, fw, 4), dtype=np.uint8)
    cv2.circle(donut_bg, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)

    label_gap     = max(2, int(radius * 0.10))
    label_block_h = depth_label_font.size + label_gap + depth_pct_font.size
    label_by      = cy - label_block_h // 2

    cache = {
        "fw": fw, "fh": fh,
        "reps_font": reps_font,
        "feedback_font": feedback_font,
        "depth_label_font": depth_label_font,
        "depth_pct_font": depth_pct_font,
        "radius": radius, "thick": thick, "cx": cx, "cy": cy,
        "rep_box_h": rep_box_h,
        "rep_txt_x": pad_x, "rep_txt_y": pad_y - 1,
        "pad_x": pad_x, "pad_y": pad_y,
        "fb_pad_x": fb_pad_x, "fb_pad_y": fb_pad_y,
        "fb_y0": fb_y0, "fb_y1": fb_y1,
        "line_h": line_h, "line_gap": line_gap,
        "max_text_w": max_text_w,
        "label_by": label_by, "label_gap": label_gap,
        "rep_bg_pil":   Image.fromarray(rep_bg,   mode="RGBA"),
        "fb_bg_pil":    Image.fromarray(fb_bg,    mode="RGBA"),
        "donut_bg_pil": Image.fromarray(donut_bg, mode="RGBA"),
    }
    _OVERLAY_CACHE[key] = cache
    return cache


def _wrap_two_lines(draw, text, font, max_width):
    """Wrap text to at most 2 lines, truncating with … if needed."""
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur: lines.append(cur)
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
    return lines if lines else [""]


def draw_overlay(frame, reps=0, feedback=None, depth_pct=0.0):
    """
    Fast overlay — works at frame resolution, uses cached static layers.
    No 1080p upscale/downscale per frame.
    """
    h, w, _ = frame.shape
    c = _get_overlay_cache(w, h)

    pct = float(np.clip(depth_pct, 0, 1))
    fw, fh = c["fw"], c["fh"]
    cx, cy, radius, thick = c["cx"], c["cy"], c["radius"], c["thick"]

    canvas = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))

    # Static layers
    canvas.alpha_composite(c["rep_bg_pil"])
    if feedback:
        canvas.alpha_composite(c["fb_bg_pil"])
    canvas.alpha_composite(c["donut_bg_pil"])

    # Dynamic arc
    if pct > 0:
        arc_np = np.zeros((fh, fw, 4), dtype=np.uint8)
        start_ang = -90
        end_ang   = start_ang + int(360 * pct)
        cv2.ellipse(arc_np, (cx, cy), (radius, radius),
                    0, start_ang, end_ang, (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)
        canvas.alpha_composite(Image.fromarray(arc_np, mode="RGBA"))

    draw = ImageDraw.Draw(canvas)

    # Reps text
    txt = f"Reps: {int(reps)}"
    draw.text((c["rep_txt_x"], c["rep_txt_y"]), txt,
              font=c["reps_font"], fill=(255, 255, 255, 255))

    # Donut labels (DEPTH + %)
    label   = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label,   font=c["depth_label_font"])
    pw = draw.textlength(pct_txt, font=c["depth_pct_font"])
    draw.text((cx - int(lw // 2), c["label_by"]),
              label, font=c["depth_label_font"], fill=(255, 255, 255, 255))
    draw.text((cx - int(pw // 2), c["label_by"] + c["depth_label_font"].size + c["label_gap"]),
              pct_txt, font=c["depth_pct_font"], fill=(255, 255, 255, 255))

    # Feedback text
    if feedback:
        fb_lines = _wrap_two_lines(draw, feedback, c["feedback_font"], c["max_text_w"])
        ty = c["fb_y0"] + c["fb_pad_y"]
        for ln in fb_lines:
            tw2 = draw.textlength(ln, font=c["feedback_font"])
            tx  = max(c["fb_pad_x"], (fw - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=c["feedback_font"], fill=(255, 255, 255, 255))
            ty += c["line_h"] + c["line_gap"]

    canvas_np   = np.array(canvas)
    alpha       = canvas_np[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = canvas_np[:, :, [2, 1, 0]].astype(np.float32)
    result      = frame.astype(np.float32) * (1.0 - alpha) + overlay_bgr * alpha
    return result.astype(np.uint8)


# ========================== GEOMETRY ==========================

def angle_3pt(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def lm_xy(landmarks, idx, w, h):
    return (landmarks[idx].x * w, landmarks[idx].y * h)

def detect_active_leg(landmarks):
    left_y  = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    return 'right' if left_y < right_y else 'left'

def check_valgus(landmarks, side, tol):
    knee_x  = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x
    ankle_x = landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x
    return not (knee_x < ankle_x - tol)


# ========================== EMA ==========================

class AngleEMA:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = float(alpha)
        self.knee = self.torso = None

    def update(self, knee_angle, torso_angle):
        ka, ta = float(knee_angle), float(torso_angle)
        if self.knee is None:
            self.knee, self.torso = ka, ta
        else:
            a = self.alpha
            self.knee  = a * ka + (1.0 - a) * self.knee
            self.torso = a * ta + (1.0 - a) * self.torso
        return self.knee, self.torso


# ========================== SKELETON ==========================

_FACE_LMS = {
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value,
    mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value,
    mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
    mp_pose.PoseLandmark.LEFT_EAR.value,  mp_pose.PoseLandmark.RIGHT_EAR.value,
    mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
}
_BODY_CONNECTIONS = tuple(
    (a, b) for (a, b) in mp_pose.POSE_CONNECTIONS
    if a not in _FACE_LMS and b not in _FACE_LMS
)
_BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))


class LandmarkStabilizer:
    def __init__(self, quality_thr=0.55, min_fraction=0.6, jump_thr=0.12, max_hold=6):
        self.body_points   = _BODY_POINTS
        self.quality_thr   = quality_thr
        self.min_fraction  = min_fraction
        self.jump_thr      = jump_thr
        self.max_hold      = max_hold
        self.last_good     = None
        self.hold          = 0

    def stabilize(self, lms):
        if lms is None:
            if self.last_good is not None and self.hold < self.max_hold:
                self.hold += 1; return self.last_good
            self.hold = 0; return None

        frac = self._quality(lms)
        if frac < self.min_fraction:
            if self.last_good is not None and self.hold < self.max_hold:
                self.hold += 1; return self.last_good
            self.hold = 0; return None

        avg_d = self._avg_disp(lms)
        if avg_d > self.jump_thr and frac < 0.8 and self.last_good is not None:
            if self.hold < self.max_hold:
                self.hold += 1; return self.last_good

        self.last_good = [type('P', (), {'x': float(lms[i].x), 'y': float(lms[i].y)})
                          for i in range(len(lms))]
        self.hold = 0
        return self.last_good

    def _quality(self, lms):
        ok = total = 0
        for i in self.body_points:
            if i >= len(lms): continue
            total += 1
            if (getattr(lms[i], 'visibility', 1.0) or 0.0) >= self.quality_thr:
                ok += 1
        return (ok / max(1, total)) if total else 1.0

    def _avg_disp(self, lms):
        if not self.last_good: return 0.0
        n = min(len(self.last_good), len(lms))
        s = c = 0.0
        for i in self.body_points:
            if i >= n: continue
            s += math.hypot(float(lms[i].x) - float(self.last_good[i].x),
                            float(lms[i].y) - float(self.last_good[i].y))
            c += 1
        return s / max(1, c)


def draw_body_only(frame, landmarks, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    for a, b in _BODY_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks): continue
        pa, pb = landmarks[a], landmarks[b]
        cv2.line(frame, (int(pa.x*w), int(pa.y*h)), (int(pb.x*w), int(pb.y*h)),
                 color, 2, cv2.LINE_AA)
    for i in _BODY_POINTS:
        if i >= len(landmarks): continue
        p = landmarks[i]
        cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, color, -1, cv2.LINE_AA)
    return frame


# ========================== SCORE HELPERS ==========================

def _display_half(x):
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

def _score_label(s):
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"


# ========================== REP COUNTER ==========================

class BulgarianRepCounter:
    def __init__(self):
        self.count = 0
        self.stage = None
        self.rep_reports = []
        self.rep_index = 1
        self.rep_start_frame = None
        self.good_reps = 0
        self.bad_reps  = 0
        self.all_feedback = collections.Counter()

        self._start_knee_angle  = None
        self._curr_min_knee     = 999.0
        self._curr_max_knee     = -999.0
        self._curr_min_torso    = 999.0
        self._curr_valgus_bad   = 0
        self._torso_bad_frames  = 0
        self._valgus_bad_frames = 0
        self._down_frames       = 0
        self._last_depth_for_ui = 0.0
        self._last_rep_end_frame = -100
        self._last_standing_knee = 170.0

        self._down_thresh = ANGLE_DOWN_THRESH
        self._up_thresh   = ANGLE_UP_THRESH

        self._cal_samples  = []
        self._cal_done     = False
        self._standing_angle = None

    def calibrate_standing(self, knee_angle, is_good_frame):
        if self._cal_done: return
        if is_good_frame and knee_angle > 130:
            self._cal_samples.append(knee_angle)
        if len(self._cal_samples) >= 10:
            self._standing_angle = float(np.median(self._cal_samples))
            new_down = self._standing_angle - 65
            new_up   = self._standing_angle - 15
            self._down_thresh = float(np.clip(new_down, 75, 120))
            self._up_thresh   = float(np.clip(new_up,  140, 170))
            self._cal_done = True

    def _start_rep(self, frame_no, start_knee_angle):
        if frame_no - self._last_rep_end_frame < REP_DEBOUNCE_FRAMES:
            return False
        self.rep_start_frame     = frame_no
        self._start_knee_angle   = float(start_knee_angle)
        self._curr_min_knee      = 999.0
        self._curr_max_knee      = -999.0
        self._curr_min_torso     = 999.0
        self._curr_valgus_bad    = 0
        self._torso_bad_frames   = 0
        self._valgus_bad_frames  = 0
        self._down_frames        = 0
        return True

    def _finish_rep(self, frame_no, score, feedback, extra=None):
        score_q = round(float(score) * 2) / 2.0
        if score_q >= GOOD_REP_MIN_SCORE: self.good_reps += 1
        else:                              self.bad_reps  += 1
        for fb in (feedback or []):
            self.all_feedback[fb] += 1

        report = {
            "rep_index":        self.rep_index,
            "score":            float(score_q),
            "score_display":    _display_half(score_q),
            "feedback":         feedback or [],
            "start_frame":      self.rep_start_frame or 0,
            "end_frame":        frame_no,
            "start_knee_angle": round(float(self._start_knee_angle or 0), 2),
            "min_knee_angle":   round(self._curr_min_knee, 2),
            "max_knee_angle":   round(self._curr_max_knee, 2),
            "torso_min_angle":  round(self._curr_min_torso, 2),
        }
        if extra: report.update(extra)
        self.rep_reports.append(report)
        self.rep_index          += 1
        self.rep_start_frame     = None
        self._start_knee_angle   = None
        self._last_depth_for_ui  = 0.0
        self._last_rep_end_frame = frame_no

    def evaluate_form(self):
        feedback = []
        score    = 10.0
        start    = self._start_knee_angle or 170
        min_k    = self._curr_min_knee if self._curr_min_knee < 900 else start
        denom    = max(10.0, start - PERFECT_MIN_KNEE)
        depth_pct = float(np.clip((start - min_k) / denom, 0, 1))

        if depth_pct < 0.6:
            feedback.append("Go deeper – aim for 90° knee angle"); score -= 3
        elif depth_pct < 0.8:
            feedback.append("Go a bit deeper"); score -= 1.5
        elif depth_pct < 0.9:
            feedback.append("Try to reach a bit more depth for full ROM"); score -= 0.5

        if self._torso_bad_frames >= TORSO_BAD_MIN_FRAMES:
            feedback.append("Keep your back straight"); score -= 2

        if self._curr_valgus_bad >= VALGUS_BAD_MIN_FRAMES:
            feedback.append("Avoid knee collapse"); score -= 2

        score = float(np.clip(score, 0, 10))
        if score < 10.0 and not feedback:
            feedback.append("Good rep, but there is still room for cleaner form")
        return score, feedback, depth_pct

    def update(self, knee_angle, torso_angle, valgus_ok_flag, frame_no):
        if knee_angle > self._up_thresh:
            self._last_standing_knee = max(self._last_standing_knee, float(knee_angle))

        # ENTER DOWN
        if knee_angle < self._down_thresh:
            if self.stage != 'down':
                self.stage = 'down'
                if not self._start_rep(frame_no, self._last_standing_knee):
                    self.stage = 'up'
                    return
            self._down_frames += 1

        # ENTER UP (potential rep complete)
        elif knee_angle > (self._up_thresh - 5) and self.stage == 'down':
            start_angle  = self._start_knee_angle or 0
            min_knee     = self._curr_min_knee if self._curr_min_knee < 900 else 0
            depth_delta  = start_angle - min_knee
            moved_enough = depth_delta >= MIN_RANGE_DELTA_DEG
            enough_frames = (
                self._down_frames >= MIN_DOWN_FRAMES
                or (self._down_frames >= 1 and depth_delta >= FAST_REP_MIN_DEPTH_DELTA)
            )
            if enough_frames and moved_enough:
                score, fb, depth = self.evaluate_form()
                self.count += 1
                self._finish_rep(frame_no, score, fb, extra={"depth_pct": float(depth)})
                self._last_standing_knee = float(knee_angle)
            else:
                self._last_depth_for_ui = 0.0
                self.rep_start_frame    = None
                self._start_knee_angle  = None
            self.stage = 'up'

        # TRACK FORM DURING DOWN
        if self.stage == 'down' and self.rep_start_frame:
            self._curr_min_knee  = min(self._curr_min_knee,  knee_angle)
            self._curr_max_knee  = max(self._curr_max_knee,  knee_angle)
            self._curr_min_torso = min(self._curr_min_torso, torso_angle)

            if torso_angle < (TORSO_LEAN_MIN - TORSO_MARGIN_DEG):
                self._torso_bad_frames += 1
            else:
                self._torso_bad_frames = 0

            if not valgus_ok_flag:
                self._valgus_bad_frames += 1
                self._curr_valgus_bad   += 1
            else:
                self._valgus_bad_frames = 0

            denom = max(10.0, self._start_knee_angle - PERFECT_MIN_KNEE)
            self._last_depth_for_ui = float(np.clip(
                (self._start_knee_angle - self._curr_min_knee) / denom, 0, 1))

    def depth_for_overlay(self):
        return float(self._last_depth_for_ui)

    def result(self):
        avg = np.mean([float(r["score"]) for r in self.rep_reports]) if self.rep_reports else 0.0
        ts  = round(float(avg) * 2) / 2.0
        aggregated_feedback = list(self.all_feedback.keys())
        return {
            "squat_count":             self.count,
            "technique_score":         float(ts),
            "technique_score_display": _display_half(ts),
            "technique_label":         _score_label(ts),
            "good_reps":               self.good_reps,
            "bad_reps":                self.bad_reps,
            "feedback":                aggregated_feedback if aggregated_feedback else ["Great form! Keep it up 💪"],
            "reps":                    self.rep_reports,
        }


# ========================== TIPS ==========================

BULGARIAN_TIPS = [
    "Keep your front shin vertical at the bottom",
    "Drive through the front heel for power",
    "Brace your core before the descent",
    "Keep hips square – avoid rotation",
    "Control the eccentric; go down a bit slower",
    "Pause 1–2s at the bottom to build stability",
]

def choose_session_tip(counter):
    if counter.all_feedback.get("Avoid knee collapse", 0) >= 2:
        return "Track your knee over your toes"
    if counter.all_feedback.get("Keep your back straight", 0) >= 2:
        return "Brace your core and keep chest up"
    return BULGARIAN_TIPS[1]


# ========================== MAIN ANALYSIS ==========================

def run_bulgarian_analysis(video_path, frame_skip=1, scale=1.0,
                           output_path="analyzed_output.mp4",
                           feedback_path="feedback_summary.txt",
                           return_video=True,
                           fast_mode=None):
    if fast_mode is True:
        return_video = False
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open: {video_path}", "squat_count": 0,
                "technique_score": 0.0, "technique_score_display": "0",
                "technique_label": "N/A", "reps": []}

    # Read rotation ONCE — OpenCV ignores metadata, so we rotate pixels manually.
    source_rotation = _read_source_rotation_degrees(video_path)

    counter = BulgarianRepCounter()
    frame_no = 0
    active_leg = None
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    ema     = AngleEMA(alpha=EMA_ALPHA)
    lm_stab = LandmarkStabilizer()

    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt           = 1.0 / float(effective_fps)

    NOPOSE_STOP_FRAMES      = int(NOPOSE_STOP_SEC * effective_fps)
    NO_MOVEMENT_STOP_FRAMES = int(NO_MOVEMENT_STOP_SEC * effective_fps)
    nopose_since_rep  = 0
    no_movement_frames = 0

    stand_knee_ema    = None
    STAND_KNEE_ALPHA  = 0.30

    RT_FB_HOLD_FRAMES = max(2, int(RT_FB_HOLD_SEC / dt))
    rt_fb_msg = None
    rt_fb_hold = 0

    prev_hip = prev_la = prev_ra = None
    hip_vel_ema = ankle_vel_ema = 0.0
    movement_free_streak = 0

    def _euclid(a, b, norm):
        return math.hypot(a[0]-b[0], a[1]-b[1]) / max(1, norm)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_skip > 1 and (frame_no % frame_skip) != 0:
            continue

        # Fix orientation before any processing or writing
        if source_rotation != 0:
            frame = _rotate_frame(frame, source_rotation)

        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        h, w = frame.shape[:2]
        if return_video and out is None:
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = pose.process(image_rgb)
        depth_live = 0.0

        if not results.pose_landmarks:
            if counter.count > 0:
                nopose_since_rep += 1
            else:
                nopose_since_rep = 0
            if counter.count > 0 and nopose_since_rep >= NOPOSE_STOP_FRAMES:
                break
            if rt_fb_hold > 0:
                rt_fb_hold -= 1
            no_movement_frames = 0
            stab_lms = lm_stab.stabilize(None)
        else:
            nopose_since_rep = 0
            lms = results.pose_landmarks.landmark

            if active_leg is None:
                active_leg = detect_active_leg(lms)
            side = "RIGHT" if active_leg == "right" else "LEFT"

            # Walking filter
            hip_px   = (lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x * w,
                        lms[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y * h)
            l_ankle_px = (lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                          lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h)
            r_ankle_px = (lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                          lms[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)
            if prev_hip is None:
                prev_hip, prev_la, prev_ra = hip_px, l_ankle_px, r_ankle_px
            norm     = max(h, w)
            hip_vel  = _euclid(hip_px, prev_hip, norm)
            an_vel   = max(_euclid(l_ankle_px, prev_la, norm), _euclid(r_ankle_px, prev_ra, norm))
            hip_vel_ema   = MOTION_EMA_ALPHA * hip_vel  + (1 - MOTION_EMA_ALPHA) * hip_vel_ema
            ankle_vel_ema = MOTION_EMA_ALPHA * an_vel   + (1 - MOTION_EMA_ALPHA) * ankle_vel_ema
            prev_hip, prev_la, prev_ra = hip_px, l_ankle_px, r_ankle_px

            movement_block = (hip_vel_ema > HIP_VEL_THRESH_PCT) or (ankle_vel_ema > ANKLE_VEL_THRESH_PCT)
            if movement_block:
                movement_free_streak = 0
                no_movement_frames   = 0
            else:
                movement_free_streak = min(MOVEMENT_CLEAR_FRAMES, movement_free_streak + 1)
                no_movement_frames  += 1

            if counter.count > 0 and no_movement_frames >= NO_MOVEMENT_STOP_FRAMES:
                break

            # Angles
            hip      = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_HIP").value, w, h)
            knee     = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value, w, h)
            ankle    = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value, w, h)
            shoulder = lm_xy(lms, getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value, w, h)
            knee_angle_raw  = angle_3pt(hip, knee, ankle)
            torso_angle_raw = angle_3pt(shoulder, hip, knee)
            knee_angle, torso_angle = ema.update(knee_angle_raw, torso_angle_raw)
            v_ok = check_valgus(lms, side, VALGUS_X_TOL)

            # Calibration
            vis = getattr(lms[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value], 'visibility', 0) or 0
            counter.calibrate_standing(knee_angle, vis > 0.5)

            counter.update(knee_angle, torso_angle, v_ok, frame_no)

            # Live depth
            if knee_angle > counter._up_thresh - 5 and movement_free_streak >= 1:
                stand_knee_ema = knee_angle if stand_knee_ema is None else (
                    STAND_KNEE_ALPHA * knee_angle + (1 - STAND_KNEE_ALPHA) * stand_knee_ema)
            if stand_knee_ema is not None:
                denom_live = max(10.0, stand_knee_ema - PERFECT_MIN_KNEE)
                depth_live = float(np.clip((stand_knee_ema - knee_angle) / denom_live, 0, 1))
            else:
                depth_live = counter.depth_for_overlay()

            # RT feedback
            live_msgs = []
            if counter.stage == 'down':
                if counter._torso_bad_frames >= TORSO_BAD_MIN_FRAMES:
                    live_msgs.append("Keep your back straight")
                if counter._valgus_bad_frames >= VALGUS_BAD_MIN_FRAMES:
                    live_msgs.append("Avoid knee collapse")
            new_msg = " | ".join(live_msgs) if live_msgs else None
            if new_msg:
                if new_msg != rt_fb_msg:
                    rt_fb_msg = new_msg; rt_fb_hold = RT_FB_HOLD_FRAMES
                else:
                    rt_fb_hold = max(rt_fb_hold, RT_FB_HOLD_FRAMES)
            else:
                if rt_fb_hold > 0: rt_fb_hold -= 1
                if rt_fb_hold == 0: rt_fb_msg = None

            stab_lms = lm_stab.stabilize(lms)

        # Draw
        if return_video:
            if stab_lms:
                frame = draw_body_only(frame, stab_lms)
            frame = draw_overlay(frame, reps=counter.count,
                                 feedback=(rt_fb_msg if rt_fb_hold > 0 else None),
                                 depth_pct=depth_live)
            if out is not None:
                out.write(frame)

    pose.close()
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    result = counter.result()
    session_tip       = choose_session_tip(counter)
    result["tips"]     = [session_tip]
    result["form_tip"] = session_tip

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {result['squat_count']}\n")
            f.write(f"Technique Score: {result['technique_score_display']} / 10  ({result['technique_label']})\n")
            f.write(f"Form Tip: {session_tip}\n")
            if result.get("feedback"):
                f.write("Feedback:\n")
                for fb in result["feedback"]:
                    f.write(f"- {fb}\n")
    except Exception:
        pass

    # ffmpeg encode — pixels already rotated, clear stale metadata
    final_path = ""
    if return_video and output_path:
        encoded_path = output_path.replace('.mp4', '_encoded.mp4')
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', output_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-movflags', '+faststart', '-pix_fmt', 'yuv420p',
                '-metadata:s:v:0', 'rotate=0',
                encoded_path
            ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            final_path = encoded_path if os.path.isfile(encoded_path) else output_path
        except Exception:
            final_path = output_path if os.path.isfile(output_path) else ""
        if not os.path.isfile(final_path) and os.path.isfile(output_path):
            final_path = output_path

    result["video_path"]    = final_path if return_video else ""
    result["feedback_path"] = feedback_path
    return result


# Backward compatibility
run_analysis = run_bulgarian_analysis

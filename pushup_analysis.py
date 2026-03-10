# -*- coding: utf-8 -*-
# pushup_analysis.py — V4: Single-pass + ffmpeg pipe output (no slow re-read)
# Key speed improvements:
#   1. NO render pass 2 — frames rendered during analysis, piped to ffmpeg
#   2. Pure OpenCV overlay (no PIL per-frame)
#   3. Better fast-rep detection thresholds
#   4. ffmpeg pipe with ultrafast preset

import os, cv2, math, numpy as np, subprocess, sys, time, json
from collections import deque

# ============ Styles ============
BAR_BG_ALPHA = 0.55
DONUT_RADIUS_SCALE = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR = (40, 200, 80)
DEPTH_RING_BG = (70, 70, 70)

def _scaled_font_size(ref_size, canvas_h):
    return max(10, int(round(ref_size * (canvas_h / 480.0))))

def display_half_str(x):
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

def score_label(s):
    s = float(s)
    if s >= 9.0: return "Excellent"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

# ============ MediaPipe ============
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception:
    mp_pose = None

# ============ Helpers ============
def _ang(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cos = float(np.clip(np.dot(ba, bc) / den, -1, 1))
    return float(np.degrees(np.arccos(cos)))

def _ema(prev, new, a):
    return float(new) if prev is None else (a * float(new) + (1 - a) * float(prev))

def _half_floor10(x):
    return max(0.0, min(10.0, math.floor(x * 2.0) / 2.0))

def _dyn_thickness(h):
    return max(2, int(round(h * 0.003))), max(3, int(round(h * 0.005)))

# ============ Body skeleton ============
_FACE_LMS = set()
_BODY_CONNECTIONS = tuple()
_BODY_POINTS = tuple()
if mp_pose:
    _FACE_LMS = {
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value,
        mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
        mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
        mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
    }
    _BODY_CONNECTIONS = tuple(
        (a, b) for (a, b) in mp_pose.POSE_CONNECTIONS
        if a not in _FACE_LMS and b not in _FACE_LMS
    )
    _BODY_POINTS = tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))


def _draw_skeleton(frame, lms, color=(255, 255, 255)):
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


# ============ Pure OpenCV Overlay (no PIL) ============
class OverlayRenderer:
    """Pre-computes layout per resolution. Pure OpenCV drawing — very fast."""

    def __init__(self, w, h):
        self.w, self.h = w, h

        reps_h = max(12, int(h * 0.045))
        self.reps_scale = max(0.3, reps_h / 22.0)
        self.reps_thick = max(1, int(h * 0.004))

        fb_h = max(10, int(h * 0.035))
        self.fb_scale = max(0.3, fb_h / 22.0)
        self.fb_thick = max(1, int(h * 0.003))

        dl_h = max(8, int(h * 0.022))
        self.dl_scale = max(0.3, dl_h / 22.0)
        self.dl_thick = max(1, int(h * 0.002))

        dp_h = max(10, int(h * 0.03))
        self.dp_scale = max(0.3, dp_h / 22.0)
        self.dp_thick = max(1, int(h * 0.003))

        ref_h = max(int(h * 0.06), reps_h * 2)
        self.radius = int(ref_h * DONUT_RADIUS_SCALE)
        self.thick = max(3, int(self.radius * DONUT_THICKNESS_FRAC))
        self.cx = w - 12 - self.radius
        self.cy = max(ref_h + self.radius // 8, self.radius + self.thick // 2 + 2)
        self.bg_alpha = BAR_BG_ALPHA

    def draw(self, frame, reps=0, feedback=None, depth_pct=0.0):
        h, w = self.h, self.w
        pct = float(np.clip(depth_pct, 0, 1))

        # Reps box
        txt = f"Reps: {int(reps)}"
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, self.reps_scale, self.reps_thick)
        box_w = min(tw + 16, w)
        box_h = min(th + bl + 12, h)
        if box_w > 0 and box_h > 0:
            sub = frame[0:box_h, 0:box_w]
            frame[0:box_h, 0:box_w] = (sub.astype(np.float32) * (1.0 - self.bg_alpha)).astype(np.uint8)
        cv2.putText(frame, txt, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    self.reps_scale, (255, 255, 255), self.reps_thick, cv2.LINE_AA)

        # Donut ring
        cv2.circle(frame, (self.cx, self.cy), self.radius, DEPTH_RING_BG, self.thick, cv2.LINE_AA)
        if pct > 0.001:
            sa = -90
            ea = sa + int(360 * pct)
            cv2.ellipse(frame, (self.cx, self.cy), (self.radius, self.radius), 0,
                        sa, ea, DEPTH_COLOR, self.thick, cv2.LINE_AA)

        # Donut center labels
        label = "DEPTH"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.dl_scale, self.dl_thick)
        pct_txt = f"{int(pct * 100)}%"
        (pw, ph), _ = cv2.getTextSize(pct_txt, cv2.FONT_HERSHEY_SIMPLEX, self.dp_scale, self.dp_thick)
        gap = max(2, int(self.radius * 0.08))
        total_h = lh + gap + ph
        by = self.cy - total_h // 2
        cv2.putText(frame, label, (self.cx - lw // 2, by + lh),
                    cv2.FONT_HERSHEY_SIMPLEX, self.dl_scale, (255, 255, 255), self.dl_thick, cv2.LINE_AA)
        cv2.putText(frame, pct_txt, (self.cx - pw // 2, by + lh + gap + ph),
                    cv2.FONT_HERSHEY_SIMPLEX, self.dp_scale, (255, 255, 255), self.dp_thick, cv2.LINE_AA)

        # Feedback bar
        if feedback:
            max_chars = max(20, int(w / max(1, int(self.fb_scale * 12))))
            if len(feedback) > max_chars:
                feedback = feedback[:max_chars - 1] + "..."
            (ftw, fth), fbl = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, self.fb_scale, self.fb_thick)
            bar_h = fth + fbl + 16
            bar_y0 = max(0, h - int(h * 0.02) - bar_h)
            bar_y1 = min(h, h - int(h * 0.02))
            if bar_y1 > bar_y0:
                sub = frame[bar_y0:bar_y1, 0:w]
                frame[bar_y0:bar_y1, 0:w] = (sub.astype(np.float32) * (1.0 - self.bg_alpha)).astype(np.uint8)
            tx = max(12, (w - ftw) // 2)
            ty = bar_y0 + 8 + fth
            cv2.putText(frame, feedback, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        self.fb_scale, (255, 255, 255), self.fb_thick, cv2.LINE_AA)

        return frame


_RENDERER_CACHE = {}
def _get_renderer(w, h):
    key = (w, h)
    if key not in _RENDERER_CACHE:
        _RENDERER_CACHE[key] = OverlayRenderer(w, h)
    return _RENDERER_CACHE[key]


# ============ Motion Detection ============
BASE_FRAME_SKIP = 2
ACTIVE_FRAME_SKIP = 1
MOTION_DETECTION_WINDOW = 10
MOTION_VEL_THRESHOLD = 0.0008
MOTION_ACCEL_THRESHOLD = 0.0005
ELBOW_CHANGE_THRESHOLD = 4.0
COOLDOWN_FRAMES = 18
MIN_VEL_FOR_MOTION = 0.0003
FAST_REP_ELBOW_RANGE = 55.0


class MotionDetector:
    def __init__(self):
        self.shoulder_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.elbow_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.velocity_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.raw_elbow_history = deque(maxlen=MOTION_DETECTION_WINDOW)
        self.is_active = False
        self.cooldown_counter = 0
        self.last_process_frame = -999
        self.activation_count = 0
        self.last_activation_reason = ""

    def add_sample(self, shoulder_y, elbow_angle, raw_elbow_min, frame_idx):
        self.shoulder_history.append(shoulder_y)
        self.elbow_history.append(elbow_angle)
        self.raw_elbow_history.append(raw_elbow_min)

        motion_detected = False
        reason = ""

        if len(self.shoulder_history) >= 2:
            vel = abs(self.shoulder_history[-1] - self.shoulder_history[-2])
            self.velocity_history.append(vel)

            if len(self.velocity_history) >= 3:
                max_vel = max(self.velocity_history)
                recent_avg = sum(list(self.velocity_history)[-3:]) / 3
                accel = abs(self.velocity_history[-1] - self.velocity_history[-2])

                if max_vel > MOTION_VEL_THRESHOLD:
                    motion_detected = True; reason = "high_vel"
                elif accel > MOTION_ACCEL_THRESHOLD:
                    motion_detected = True; reason = "accel"
                elif recent_avg > MOTION_VEL_THRESHOLD * 0.55:
                    motion_detected = True; reason = "sustained"

        if len(self.elbow_history) >= 3:
            elbow_change = abs(self.elbow_history[-1] - self.elbow_history[-3])
            elbow_vel = abs(self.elbow_history[-1] - self.elbow_history[-2])
            if elbow_change > ELBOW_CHANGE_THRESHOLD:
                motion_detected = True; reason = "elbow_change"
            elif elbow_vel > ELBOW_CHANGE_THRESHOLD * 0.45:
                motion_detected = True; reason = "elbow_vel"

        if len(self.raw_elbow_history) >= 3:
            raw_change = abs(self.raw_elbow_history[-1] - self.raw_elbow_history[-3])
            raw_vel = abs(self.raw_elbow_history[-1] - self.raw_elbow_history[-2])
            if raw_change > 8.0:
                motion_detected = True; reason = "raw_spike"
            elif raw_vel > 5.0:
                motion_detected = True; reason = "raw_vel"

        # V-pattern for fast reps
        if len(self.raw_elbow_history) >= 5:
            elbows = list(self.raw_elbow_history)
            if (elbows[-5] - elbows[-3] > 10) and (elbows[-1] - elbows[-3] > 10):
                motion_detected = True; reason = "V_pattern"

        # Short V-pattern (3-sample) for very fast reps
        if len(self.raw_elbow_history) >= 3:
            elbows = list(self.raw_elbow_history)
            d1 = elbows[-3] - elbows[-2]
            d2 = elbows[-1] - elbows[-2]
            if d1 > 6 and d2 > 6:
                motion_detected = True; reason = "fast_V"

        if len(self.shoulder_history) >= 5:
            diffs = [self.shoulder_history[i + 1] - self.shoulder_history[i]
                     for i in range(len(self.shoulder_history) - 1)]
            if len(diffs) >= 4:
                sign_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
                max_diff = max(abs(d) for d in diffs)
                if sign_changes >= 1 and max_diff > MIN_VEL_FOR_MOTION:
                    motion_detected = True; reason = "direction_change"

        # Wide elbow range = active motion
        if len(self.raw_elbow_history) >= 4:
            rng = max(self.raw_elbow_history) - min(self.raw_elbow_history)
            if rng > 20:
                motion_detected = True; reason = "elbow_range"

        if motion_detected:
            self.activate(reason)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.cooldown_counter == 0:
                self.is_active = False

    def activate(self, reason=""):
        if not self.is_active:
            self.is_active = True
            self.activation_count += 1
            self.last_activation_reason = reason
        self.cooldown_counter = COOLDOWN_FRAMES

    def should_process(self, frame_idx):
        skip = ACTIVE_FRAME_SKIP if (self.is_active or self.cooldown_counter > 0) else BASE_FRAME_SKIP
        should = (frame_idx - self.last_process_frame) >= skip
        if should:
            self.last_process_frame = frame_idx
        return should


# ============ Analysis Parameters ============
ELBOW_BENT_ANGLE = 102.0
SHOULDER_MIN_DESCENT = 0.036
RESET_ASCENT = 0.024
RESET_ELBOW = 153.0
REFRACTORY_FRAMES = 1
ELBOW_EMA_ALPHA = 0.72
SHOULDER_EMA_ALPHA = 0.67
VIS_THR_STRICT = 0.29
PLANK_BODY_ANGLE_MAX = 26.0
HANDS_BELOW_SHOULDERS = 0.035
ONPUSHUP_MIN_FRAMES = 2
OFFPUSHUP_MIN_FRAMES = 5
AUTO_STOP_AFTER_EXIT_SEC = 1.5
TAIL_NOPOSE_STOP_SEC = 1.0

FB_ERROR_DEPTH = "Go deeper (chest to floor)"
FB_ERROR_HIPS = "Keep hips level (don't sag or pike)"
FB_ERROR_LOCKOUT = "Fully lockout arms at top"
FB_ERROR_ELBOWS = "Keep elbows at 45\u00b0 (not flared)"
PERF_TIP_SLOW_DOWN = "Lower slowly for better control"
PERF_TIP_TEMPO = "Try 2-1-2 tempo (down-pause-up)"
PERF_TIP_BREATHING = "Breathe: inhale down, exhale up"
PERF_TIP_CORE = "Engage core throughout movement"

FB_W_DEPTH = 1.2; FB_W_HIPS = 1.0; FB_W_LOCKOUT = 0.9; FB_W_ELBOWS = 0.7
FB_WEIGHTS = {FB_ERROR_DEPTH: FB_W_DEPTH, FB_ERROR_HIPS: FB_W_HIPS,
              FB_ERROR_LOCKOUT: FB_W_LOCKOUT, FB_ERROR_ELBOWS: FB_W_ELBOWS}
FB_DEFAULT_WEIGHT = 0.5; PENALTY_MIN_IF_ANY = 0.5
FORM_ERROR_PRIORITY = [FB_ERROR_DEPTH, FB_ERROR_LOCKOUT, FB_ERROR_HIPS, FB_ERROR_ELBOWS]
PERF_TIP_PRIORITY = [PERF_TIP_SLOW_DOWN, PERF_TIP_TEMPO, PERF_TIP_BREATHING, PERF_TIP_CORE]

DEPTH_EXCELLENT_ANGLE = 85.0; DEPTH_GOOD_ANGLE = 95.0; DEPTH_FAIR_ANGLE = 105.0; DEPTH_POOR_ANGLE = 115.0
HIP_EXCELLENT = 8.0; HIP_GOOD = 15.0; HIP_FAIR = 22.0; HIP_POOR = 30.0
LOCKOUT_EXCELLENT = 175.0; LOCKOUT_GOOD = 170.0; LOCKOUT_FAIR = 165.0; LOCKOUT_POOR = 160.0
FLARE_EXCELLENT = 45.0; FLARE_GOOD = 55.0; FLARE_FAIR = 65.0; FLARE_POOR = 75.0
DESCENT_SPEED_FAST = 0.0012

DEPTH_FAIL_MIN_REPS = 2; HIPS_FAIL_MIN_REPS = 2; LOCKOUT_FAIL_MIN_REPS = 2
FLARE_FAIL_MIN_REPS = 2; TEMPO_CHECK_MIN_REPS = 1
DEPTH_ERROR_ANGLE = 110.0; LOCKOUT_ERROR_ANGLE = 165.0

BURST_FRAMES = 5
INFLECT_VEL_THR = 0.0027
MIN_CYCLE_ELBOW_SAMPLES = 4
ROBUST_BOTTOM_PERCENTILE = 25; ROBUST_TOP_PERCENTILE = 75; ROBUST_CONFIRMED_PERCENTILE = 10


# ============ Geometry helpers ============
def _calculate_body_angle(lms, LSH, RSH, LH, RH, LA, RA):
    mid_sh = ((lms[LSH].x + lms[RSH].x) / 2.0, (lms[LSH].y + lms[RSH].y) / 2.0)
    mid_ank = ((lms[LA].x + lms[RA].x) / 2.0, (lms[LA].y + lms[RA].y) / 2.0)
    dx = mid_sh[0] - mid_ank[0]; dy = mid_sh[1] - mid_ank[1]
    return abs(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-9)))

def _calculate_hip_misalignment(lms, LSH, RSH, LH, RH, LA, RA):
    mid_sh = ((lms[LSH].x + lms[RSH].x) / 2.0, (lms[LSH].y + lms[RSH].y) / 2.0)
    mid_hp = ((lms[LH].x + lms[RH].x) / 2.0, (lms[LH].y + lms[RH].y) / 2.0)
    mid_ank = ((lms[LA].x + lms[RA].x) / 2.0, (lms[LA].y + lms[RA].y) / 2.0)
    return abs(180.0 - _ang(mid_sh, mid_hp, mid_ank))

def _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW):
    mid_sh = ((lms[LSH].x + lms[RSH].x) / 2.0, (lms[LSH].y + lms[RSH].y) / 2.0)
    left_vec_sh = (mid_sh[0] - lms[LSH].x, mid_sh[1] - lms[LSH].y)
    left_vec_elb = (lms[LE].x - lms[LSH].x, lms[LE].y - lms[LSH].y)
    left_angle = abs(math.degrees(math.atan2(
        left_vec_sh[0] * left_vec_elb[1] - left_vec_sh[1] * left_vec_elb[0],
        left_vec_sh[0] * left_vec_elb[0] + left_vec_sh[1] * left_vec_elb[1])))
    right_vec_sh = (mid_sh[0] - lms[RSH].x, mid_sh[1] - lms[RSH].y)
    right_vec_elb = (lms[RE].x - lms[RSH].x, lms[RE].y - lms[RSH].y)
    right_angle = abs(math.degrees(math.atan2(
        right_vec_sh[0] * right_vec_elb[1] - right_vec_sh[1] * right_vec_elb[0],
        right_vec_sh[0] * right_vec_elb[0] + right_vec_sh[1] * right_vec_elb[1])))
    return max(left_angle, right_angle)


def _robust_cycle_elbows(bottom_samples, top_samples, fallback_bottom=None, fallback_top=None, confirmed_bottom=None):
    robust_bottom = fallback_bottom
    robust_top = fallback_top
    if confirmed_bottom and len(confirmed_bottom) >= 1:
        robust_bottom = float(np.percentile(confirmed_bottom, ROBUST_CONFIRMED_PERCENTILE))
    elif bottom_samples:
        robust_bottom = float(np.percentile(bottom_samples, ROBUST_BOTTOM_PERCENTILE))
    if fallback_bottom is not None and robust_bottom is not None:
        robust_bottom = min(robust_bottom, fallback_bottom)
    if top_samples:
        robust_top = float(np.percentile(top_samples, ROBUST_TOP_PERCENTILE))
    return robust_bottom, robust_top


def _evaluate_cycle_form(bottom_elbow, top_elbow, hip_mis, flare, desc_vel,
                         dfc, hfc, lfc, ffc, fdc, dar, har, lar, far, tar,
                         sfe, spt, rc, n_bottom, n_top, ctrs):
    has_depth = has_lockout = has_hips = has_flare = False

    if bottom_elbow is not None and n_bottom >= MIN_CYCLE_ELBOW_SAMPLES:
        if bottom_elbow > DEPTH_ERROR_ANGLE:
            has_depth = True
            ctrs['depth_fail_count'] += 1
            if ctrs['depth_fail_count'] >= DEPTH_FAIL_MIN_REPS and not dar:
                sfe.add(FB_ERROR_DEPTH); ctrs['depth_already_reported'] = True

    if top_elbow is not None and n_top >= MIN_CYCLE_ELBOW_SAMPLES:
        if top_elbow < LOCKOUT_ERROR_ANGLE:
            has_lockout = True
            ctrs['lockout_fail_count'] += 1
            if ctrs['lockout_fail_count'] >= LOCKOUT_FAIL_MIN_REPS and not lar:
                sfe.add(FB_ERROR_LOCKOUT); ctrs['lockout_already_reported'] = True

    if hip_mis is not None and hip_mis > HIP_FAIR:
        has_hips = True
        ctrs['hips_fail_count'] += 1
        if ctrs['hips_fail_count'] >= HIPS_FAIL_MIN_REPS and not har:
            sfe.add(FB_ERROR_HIPS); ctrs['hips_already_reported'] = True

    if flare is not None and flare > FLARE_FAIR:
        has_flare = True
        ctrs['flare_fail_count'] += 1
        if ctrs['flare_fail_count'] >= FLARE_FAIL_MIN_REPS and not far:
            sfe.add(FB_ERROR_ELBOWS); ctrs['flare_already_reported'] = True

    if rc >= TEMPO_CHECK_MIN_REPS and not tar:
        if desc_vel > DESCENT_SPEED_FAST:
            ctrs['fast_descent_count'] += 1
            if ctrs['fast_descent_count'] >= 1:
                spt.add(PERF_TIP_SLOW_DOWN); spt.add(PERF_TIP_TEMPO)
                ctrs['tempo_already_reported'] = True

    return has_depth or has_lockout or has_hips or has_flare


def _count_rep(rr, rc, bottom_elbow, descent_from, bottom_sy, asc, has_issues,
               bpme, tpme, hip_mis, flare):
    ds = ls = hs = fs = 10.0
    if bpme:
        if bpme <= DEPTH_EXCELLENT_ANGLE: ds = 10.0
        elif bpme <= DEPTH_GOOD_ANGLE: ds = 9.0
        elif bpme <= DEPTH_FAIR_ANGLE: ds = 7.5
        elif bpme <= DEPTH_POOR_ANGLE: ds = 5.0
        else: ds = 3.0
    if tpme:
        if tpme >= LOCKOUT_EXCELLENT: ls = 10.0
        elif tpme >= LOCKOUT_GOOD: ls = 9.0
        elif tpme >= LOCKOUT_FAIR: ls = 7.5
        elif tpme >= LOCKOUT_POOR: ls = 5.0
        else: ls = 3.0
    if hip_mis is not None:
        if hip_mis <= HIP_EXCELLENT: hs = 10.0
        elif hip_mis <= HIP_GOOD: hs = 9.0
        elif hip_mis <= HIP_FAIR: hs = 7.5
        elif hip_mis <= HIP_POOR: hs = 5.0
        else: hs = 3.0
    if flare is not None:
        if flare <= FLARE_EXCELLENT: fs = 10.0
        elif flare <= FLARE_GOOD: fs = 9.0
        elif flare <= FLARE_FAIR: fs = 7.5
        elif flare <= FLARE_POOR: fs = 5.0
        else: fs = 3.0

    score = round((ds * 0.35 + ls * 0.25 + hs * 0.25 + fs * 0.15) * 2) / 2
    asc.append(score)
    rr.append({
        "rep_index": int(rc + 1), "score": float(score), "good": bool(score >= 9.0),
        "bottom_elbow": float(bottom_elbow), "descent_from": float(descent_from),
        "bottom_shoulder_y": float(bottom_sy),
        "detailed_scores": {"depth": float(ds), "lockout": float(ls), "hips": float(hs), "flare": float(fs)},
        "measurements": {
            "bottom_elbow_angle": float(bpme) if bpme else None,
            "top_elbow_angle": float(tpme) if tpme else None,
            "hip_misalignment": float(hip_mis) if hip_mis else None,
            "elbow_flare": float(flare) if flare else None}
    })


# ============ Video rotation ============
def _get_video_rotation(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for s in data.get('streams', []):
                if s.get('codec_type') != 'video': continue
                tags = s.get('tags', {})
                if 'rotate' in tags: return int(tags['rotate'])
                for sd in s.get('side_data_list', []):
                    if 'rotation' in sd: return (-int(sd['rotation'])) % 360
    except Exception: pass
    return 0

def _apply_rotation(frame, angle):
    angle = angle % 360
    if angle == 90: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ============ ffmpeg pipe writer ============
class FFmpegPipeWriter:
    """Write frames directly to ffmpeg via stdin pipe. No intermediate file needed."""

    def __init__(self, output_path, width, height, fps, out_w, out_h, crf=28):
        self.output_path = output_path
        self.proc = None

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(fps),
            '-i', 'pipe:0',
            '-vf', f'scale={out_w}:{out_h}:flags=bilinear',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', str(crf),
            '-threads', '2',
            '-movflags', '+faststart', '-pix_fmt', 'yuv420p',
            output_path
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"[PUSHUP] ffmpeg pipe failed: {e}", file=sys.stderr, flush=True)

    def write(self, frame):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                pass

    def release(self):
        if not self.proc: return ""
        try: self.proc.stdin.close()
        except Exception: pass
        try: self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait()

        rc = self.proc.returncode
        if rc != 0:
            try:
                err = self.proc.stderr.read().decode(errors='replace')[-500:]
                print(f"[PUSHUP] ffmpeg rc={rc}: {err}", file=sys.stderr, flush=True)
            except Exception: pass

        if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 1000:
            return self.output_path
        return ""


# ============ SINGLE-PASS: Analysis + render simultaneously ============
def run_pushup_analysis(video_path,
                        frame_skip=None,
                        scale=0.4,
                        output_path="pushup_analyzed.mp4",
                        feedback_path="pushup_feedback.txt",
                        preserve_quality=False,
                        encode_crf=None,
                        return_video=True,
                        fast_mode=None):
    if mp_pose is None:
        return _ret_err("Mediapipe not available", feedback_path)

    t_start = time.time()

    if fast_mode:
        return_video = False
        scale = min(scale, 0.35)
    if preserve_quality:
        scale = 1.0
        encode_crf = 18 if encode_crf is None else encode_crf
    else:
        encode_crf = 28 if encode_crf is None else encode_crf

    create_video = bool(return_video) and bool(output_path)
    model_complexity = 0 if fast_mode else 1

    rotation = _get_video_rotation(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _ret_err("Could not open video", feedback_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if rotation % 360 in (90, 270):
        out_w, out_h = orig_h, orig_w
    else:
        out_w, out_h = orig_w, orig_h

    work_w = int(out_w * scale) if scale != 1.0 else out_w
    work_h = int(out_h * scale) if scale != 1.0 else out_h
    work_w = work_w if work_w % 2 == 0 else work_w + 1
    work_h = work_h if work_h % 2 == 0 else work_h + 1

    effective_fps = max(1.0, fps_in / max(1, BASE_FRAME_SKIP))
    sec_to_frames = lambda s: max(1, int(s * effective_fps))

    print(f"[PUSHUP] V4 single-pass | orig={orig_w}x{orig_h} work={work_w}x{work_h} "
          f"rot={rotation} video={create_video}", file=sys.stderr, flush=True)

    # Setup video writer (ffmpeg pipe — writes during analysis, no pass 2)
    writer = None
    renderer = None
    if create_video:
        writer = FFmpegPipeWriter(output_path, work_w, work_h, effective_fps, out_w, out_h, encode_crf)
        if writer.proc is None:
            writer = None
            print("[PUSHUP] WARNING: ffmpeg pipe failed, no video output", file=sys.stderr, flush=True)
        renderer = _get_renderer(work_w, work_h)

    # ===== Analysis state =====
    cap = cv2.VideoCapture(video_path)
    fi = 0
    md = MotionDetector()
    fp = 0; fs = 0; burst = 0

    rc = 0; gr = 0; br = 0; rr = []; asc = []
    sfe = set(); spt = set()

    LSH = mp_pose.PoseLandmark.LEFT_SHOULDER.value; RSH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LE = mp_pose.PoseLandmark.LEFT_ELBOW.value; RE = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW = mp_pose.PoseLandmark.LEFT_WRIST.value; RW = mp_pose.PoseLandmark.RIGHT_WRIST.value
    LH = mp_pose.PoseLandmark.LEFT_HIP.value; RH = mp_pose.PoseLandmark.RIGHT_HIP.value
    LA = mp_pose.PoseLandmark.LEFT_ANKLE.value; RA = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    ee = None; se = None; sp = None; svp = None; bsy = None
    dsb = None; anb = True; lbf = -10 ** 9
    cmd = 0.0; cme = 999.0; ctc = False
    op = False; ops = 0; offs = 0
    offr = 0; npr = 0
    rfm = None; rfh = 0

    cbs = []; cts = []; ccs = []
    bpme = None; tpme = None; cmhm = None; cmf = None; cmdv = 0.0; idp = False
    ctrs = {
        'depth_fail_count': 0, 'hips_fail_count': 0, 'lockout_fail_count': 0,
        'flare_fail_count': 0, 'fast_descent_count': 0,
        'depth_already_reported': False, 'hips_already_reported': False,
        'lockout_already_reported': False, 'flare_already_reported': False,
        'tempo_already_reported': False,
    }
    rcr = []
    is_fast_rep_mode = False

    OFFPUSHUP_STOP = sec_to_frames(AUTO_STOP_AFTER_EXIT_SEC)
    NOPOSE_STOP = sec_to_frames(TAIL_NOPOSE_STOP_SEC)
    RFHF = sec_to_frames(0.8)
    ERA = max(RESET_ASCENT * 0.58, 0.014)
    EFR = REFRACTORY_FRAMES

    with mp_pose.Pose(model_complexity=model_complexity, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break

            if time.time() - t_start > 180:
                print("[PUSHUP] timeout 180s", file=sys.stderr, flush=True); break

            fi += 1
            process_now = False
            if burst > 0:
                process_now = True; burst -= 1
            elif md.should_process(fi):
                process_now = True

            if not process_now:
                fs += 1; continue
            fp += 1

            frame = _apply_rotation(frame, rotation)

            # Scale for pose detection + video output (same resolution)
            if scale != 1.0:
                work = cv2.resize(frame, (work_w, work_h))
            else:
                work = frame
                if work.shape[1] != work_w or work.shape[0] != work_h:
                    work = cv2.resize(frame, (work_w, work_h))

            res = pose.process(cv2.cvtColor(work, cv2.COLOR_BGR2RGB))
            dl = 0.0

            if not res.pose_landmarks:
                npr = (npr + 1) if rc > 0 else 0
                if rc > 0 and npr >= NOPOSE_STOP: break
                if rfh > 0: rfh -= 1
                if create_video and writer and renderer:
                    renderer.draw(work, reps=rc, feedback=rfm if rfh > 0 else None, depth_pct=0.0)
                    writer.write(work)
                continue

            npr = 0
            lms = res.pose_landmarks.landmark

            vL = lms[LSH].visibility + lms[LE].visibility + lms[LW].visibility
            vR = lms[RSH].visibility + lms[RE].visibility + lms[RW].visibility
            if vL >= vR:
                side, S, E, W = "LEFT", LSH, LE, LW
            else:
                side, S, E, W = "RIGHT", RSH, RE, RW

            min_vis = min(lms[S].visibility, lms[E].visibility, lms[W].visibility,
                          lms[LH].visibility, lms[RH].visibility)
            vis_ok = (min_vis >= VIS_THR_STRICT)

            sr = float(lms[S].y)
            reL = _ang((lms[LSH].x, lms[LSH].y), (lms[LE].x, lms[LE].y), (lms[LW].x, lms[LW].y))
            reR = _ang((lms[RSH].x, lms[RSH].y), (lms[RE].x, lms[RE].y), (lms[RW].x, lms[RW].y))
            re = reL if side == "LEFT" else reR
            rem = min(reL, reR)

            ee = _ema(ee, re, ELBOW_EMA_ALPHA)
            se = _ema(se, sr, SHOULDER_EMA_ALPHA)
            sy = se; ea = ee

            md.add_sample(sy, ea, rem, fi)

            if bsy is None: bsy = sy
            dl = float(np.clip((sy - bsy) / max(0.10, SHOULDER_MIN_DESCENT * 1.2), 0.0, 1.0))

            ba = _calculate_body_angle(lms, LSH, RSH, LH, RH, LA, RA)
            hp = (lms[LW].y > lms[LSH].y - HANDS_BELOW_SHOULDERS) and \
                 (lms[RW].y > lms[RSH].y - HANDS_BELOW_SHOULDERS)
            in_plank = (ba <= PLANK_BODY_ANGLE_MAX) and hp

            if vis_ok and in_plank:
                ops += 1; offs = 0
            else:
                offs += 1; ops = 0

            if (not op) and ops >= ONPUSHUP_MIN_FRAMES:
                op = True
                dsb = None; anb = True; cmd = 0.0; cme = 999.0; ctc = False
                cbs = []; cts = []; ccs = []; bpme = None; tpme = None
                cmhm = None; cmf = None; cmdv = 0.0; idp = False
                md.activate("enter_plank")

            if op and offs >= OFFPUSHUP_MIN_FRAMES:
                rob, rot = _robust_cycle_elbows(cbs, cts, bpme, tpme, confirmed_bottom=ccs)
                chi = _evaluate_cycle_form(rob, rot, cmhm, cmf, cmdv,
                    ctrs['depth_fail_count'], ctrs['hips_fail_count'], ctrs['lockout_fail_count'],
                    ctrs['flare_fail_count'], ctrs['fast_descent_count'],
                    ctrs['depth_already_reported'], ctrs['hips_already_reported'],
                    ctrs['lockout_already_reported'], ctrs['flare_already_reported'],
                    ctrs['tempo_already_reported'], sfe, spt, rc, len(cbs), len(cts), ctrs)
                if (not ctc) and (cmd >= SHOULDER_MIN_DESCENT) and (cme <= ELBOW_BENT_ANGLE):
                    _count_rep(rr, rc, cme, dsb if dsb is not None else sy,
                               bsy + cmd if bsy is not None else sy, asc, chi, rob, rot, cmhm, cmf)
                    rc += 1
                    if chi: br += 1
                    else: gr += 1
                op = False; offr = 0
                dsb = None; cmd = 0.0; cme = 999.0; ctc = False
                cbs = []; cts = []; ccs = []; bpme = None; tpme = None
                cmhm = None; cmf = None; cmdv = 0.0

            if (not op) and rc > 0:
                offr += 1
                if offr >= OFFPUSHUP_STOP: break

            sv = 0.0 if sp is None else (sy - sp)
            crt = None

            if op and (dsb is not None):
                near_inf = (abs(sv) <= INFLECT_VEL_THR)
                sign_flip = (svp is not None) and ((svp < 0 and sv >= 0) or (svp > 0 and sv <= 0))
                if near_inf or sign_flip:
                    burst = max(burst, BURST_FRAMES)
                    md.activate("inflection")
            svp = sv

            if op and vis_ok:
                if dsb is None:
                    if sv > abs(INFLECT_VEL_THR):
                        dsb = sy; cmd = 0.0; cme = ea; ctc = False
                        cbs = []; cts = []; ccs = []; bpme = None; tpme = None
                        cmhm = None; cmf = None; cmdv = 0.0; idp = True
                        md.activate("start_descent")
                else:
                    cmd = max(cmd, sy - dsb)
                    cme = min(cme, ea)
                    va = abs(sv)
                    if sv > 0 and idp: cmdv = max(cmdv, va)
                    if va > 0.0005: cmdv = max(cmdv, va)

                    men = min(reL, reR)
                    cbs.append(men)
                    if len(cbs) > 40: cbs.pop(0)
                    if bpme is None: bpme = men
                    else: bpme = min(bpme, men)

                    xen = max(reL, reR)
                    cts.append(xen)
                    if len(cts) > 40: cts.pop(0)
                    if tpme is None: tpme = xen
                    else: tpme = max(tpme, xen)

                    hm = _calculate_hip_misalignment(lms, LSH, RSH, LH, RH, LA, RA)
                    if cmhm is None: cmhm = hm
                    else: cmhm = max(cmhm, hm)

                    ef_ = _calculate_elbow_flare(lms, LSH, RSH, LE, RE, LW, RW)
                    if cmf is None: cmf = ef_
                    else: cmf = max(cmf, ef_)

                    rba = (dsb is not None) and ((dsb - sy) >= RESET_ASCENT)
                    rbe_ = (ea >= RESET_ELBOW)
                    if sv < 0 and idp: idp = False

                    if rba or rbe_:
                        rob, rot = _robust_cycle_elbows(cbs, cts, bpme, tpme, confirmed_bottom=ccs)
                        chi = _evaluate_cycle_form(rob, rot, cmhm, cmf, cmdv,
                            ctrs['depth_fail_count'], ctrs['hips_fail_count'], ctrs['lockout_fail_count'],
                            ctrs['flare_fail_count'], ctrs['fast_descent_count'],
                            ctrs['depth_already_reported'], ctrs['hips_already_reported'],
                            ctrs['lockout_already_reported'], ctrs['flare_already_reported'],
                            ctrs['tempo_already_reported'], sfe, spt, rc, len(cbs), len(cts), ctrs)
                        if (not ctc) and (cmd >= SHOULDER_MIN_DESCENT) and (cme <= ELBOW_BENT_ANGLE):
                            _count_rep(rr, rc, ea, dsb, bsy + cmd if bsy is not None else sy,
                                       asc, chi, rob, rot, cmhm, cmf)
                            rc += 1
                            if chi: br += 1
                            else: gr += 1
                        dsb = sy; cmd = 0.0; cme = ea; ctc = False; anb = True
                        cbs = []; cts = []; ccs = []; bpme = None; tpme = None
                        cmhm = None; cmf = None; cmdv = 0.0; idp = True
                        md.activate("reset")

                    da = 0.0 if dsb is None else (sy - dsb)
                    ab = (ea <= ELBOW_BENT_ANGLE) and (da >= SHOULDER_MIN_DESCENT)
                    rb_ = (rem <= (ELBOW_BENT_ANGLE + 9.0)) and (da >= SHOULDER_MIN_DESCENT * 0.87)
                    ab = ab or rb_
                    cc = (fi - lbf) >= EFR

                    if ab:
                        ccs.append(rem)
                        if len(ccs) > 15: ccs.pop(0)

                    if ab and anb and cc and (not ctc):
                        rhi = False
                        rob, rot = _robust_cycle_elbows(cbs, cts, bpme, tpme, confirmed_bottom=ccs)
                        if rob and rob > DEPTH_ERROR_ANGLE: rhi = True
                        if rot and rot < LOCKOUT_ERROR_ANGLE: rhi = True
                        if cmhm and cmhm > HIP_FAIR: rhi = True
                        if cmf and cmf > FLARE_FAIR: rhi = True

                        if rob and rot: rcr.append(rot - rob)

                        _count_rep(rr, rc, ea, dsb if dsb is not None else sy, sy,
                                   asc, rhi, rob or rem, rot or max(reL, reR), cmhm or 0.0, cmf or 0.0)
                        rc += 1
                        if rhi: br += 1
                        else: gr += 1
                        lbf = fi; anb = False; ctc = True; tpme = max(reL, reR); idp = False
                        md.activate("rep")
                        if len(rcr) >= 2:
                            is_fast_rep_mode = (sum(rcr) / len(rcr)) < FAST_REP_ELBOW_RANGE

                    if (not anb) and (lbf > 0):
                        if sp is not None and (sp - sy) > 0 and dsb is not None:
                            if ((dsb + cmd) - sy) >= ERA: anb = True

                    if ab:
                        rob, _ = _robust_cycle_elbows(cbs, cts, bpme, tpme, confirmed_bottom=ccs)
                        if rob and rob > DEPTH_ERROR_ANGLE:
                            ctrs['depth_fail_count'] += 1
                            if ctrs['depth_fail_count'] >= DEPTH_FAIL_MIN_REPS and not ctrs['depth_already_reported']:
                                sfe.add(FB_ERROR_DEPTH); ctrs['depth_already_reported'] = True
                                crt = FB_ERROR_DEPTH
            else:
                dsb = None; anb = True

            if crt:
                if crt != rfm: rfm = crt; rfh = RFHF
                else: rfh = max(rfh, RFHF)
            else:
                if rfh > 0: rfh -= 1

            # ===== RENDER FRAME INLINE (no pass 2!) =====
            if create_video and writer and renderer:
                _draw_skeleton(work, lms)
                renderer.draw(work, reps=rc, feedback=rfm if rfh > 0 else None, depth_pct=dl)
                writer.write(work)

            if sy is not None: sp = sy

    # Final uncounted cycle
    if op and (not ctc) and (cmd >= SHOULDER_MIN_DESCENT) and (cme <= ELBOW_BENT_ANGLE):
        rob, rot = _robust_cycle_elbows(cbs, cts, bpme, tpme, confirmed_bottom=ccs)
        chi = _evaluate_cycle_form(rob, rot, cmhm, cmf, cmdv,
            ctrs['depth_fail_count'], ctrs['hips_fail_count'], ctrs['lockout_fail_count'],
            ctrs['flare_fail_count'], ctrs['fast_descent_count'],
            ctrs['depth_already_reported'], ctrs['hips_already_reported'],
            ctrs['lockout_already_reported'], ctrs['flare_already_reported'],
            ctrs['tempo_already_reported'], sfe, spt, rc, len(cbs), len(cts), ctrs)
        _count_rep(rr, rc, cme, dsb if dsb is not None else (bsy or 0.0),
                   (bsy + cmd) if bsy is not None else 0.0, asc, chi, rob, rot, cmhm, cmf)
        rc += 1
        if chi: br += 1
        else: gr += 1

    cap.release()

    # Finalize video
    final_path = ""
    if create_video and writer:
        final_path = writer.release()

    # Compute technique score
    if rc == 0:
        ts = 0.0
    else:
        if sfe:
            p = sum(FB_WEIGHTS.get(m, FB_DEFAULT_WEIGHT) for m in sfe)
            p = max(PENALTY_MIN_IF_ANY, p)
        else:
            p = 0.0
        ts = _half_floor10(max(0.0, 10.0 - p))

    if ts == 10.0 and rc > 0: gr = rc; br = 0

    fel = [e for e in FORM_ERROR_PRIORITY if e in sfe]
    ptl = [t for t in PERF_TIP_PRIORITY if t in spt]
    tf = fp + fs; eff = (fs / tf * 100) if tf > 0 else 0

    total_time = time.time() - t_start
    print(f"[PUSHUP] Done: {rc} reps, {fp} proc, {fs} skip ({eff:.0f}%), {total_time:.1f}s",
          file=sys.stderr, flush=True)

    # Write feedback file
    try:
        with open(feedback_path, "w") as f:
            f.write(f"Total Reps: {rc}\nGood: {gr} | Bad: {br}\n")
            f.write(f"Technique Score: {display_half_str(ts)}/10 ({score_label(ts)})\n")
            for fb in fel: f.write(f"- {fb}\n")
    except Exception: pass

    result = {
        "squat_count": int(rc),
        "technique_score": float(ts),
        "technique_score_display": display_half_str(ts),
        "technique_label": score_label(ts),
        "good_reps": int(gr), "bad_reps": int(br),
        "feedback": fel if fel else (["Great form! Keep it up \U0001f4aa"] if ts == 10.0 and rc > 0 else []),
        "tips": ptl, "reps": rr,
        "video_path": final_path if create_video else "",
        "feedback_path": feedback_path,
        "processing_stats": {
            "frames_processed": fp, "frames_skipped": fs,
            "efficiency_percent": round(eff, 1),
            "motion_activations": md.activation_count
        }
    }
    if ptl: result["form_tip"] = ptl[0]; result["primary_perf_tip"] = ptl[0]
    if fel and fel[0] != "Great form! Keep it up \U0001f4aa": result["primary_form_error"] = fel[0]
    return result


def _ret_err(msg, feedback_path):
    try:
        with open(feedback_path, "w") as f: f.write(msg + "\n")
    except Exception: pass
    return {
        "squat_count": 0, "technique_score": 0.0,
        "technique_score_display": display_half_str(0.0),
        "technique_label": score_label(0.0),
        "good_reps": 0, "bad_reps": 0,
        "feedback": [], "tips": [], "reps": [],
        "video_path": "", "feedback_path": feedback_path
    }


def run_analysis(*args, **kwargs):
    return run_pushup_analysis(*args, **kwargs)

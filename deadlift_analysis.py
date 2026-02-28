# -*- coding: utf-8 -*-
# deadlift_analysis.py — Deadlift aligned to squat standard (fixed feedback aggregation)
# - Overlay/encoding/speed identical to good_morning (orig-res output, scaled fonts)
# - Bi-directional donut (0% bottom → 100% upright) with smoothing
# - Robust legs: Kalman + gating + floor anchor (+ optional YOLOv8n-ONNX occlusion detector)
# - Scoring feedback per-rep + one non-scoring tip per rep
# - TOP-LEVEL feedback now aggregates ONLY problems (score < 10). No generic "Analysis complete/Great form".

import os, cv2, math, numpy as np, subprocess
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ===================== Optional ONNX Runtime (YOLO detector) =====================
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except Exception:
    _ORT_AVAILABLE = False

# ===================== STYLE / FONTS =====================
BAR_BG_ALPHA         = 0.55
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)
DEPTH_RING_BG        = (70, 70, 70)

FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"

# Reference sizes (same as good_morning)
_REF_H = 480.0
_REF_REPS_FONT_SIZE        = 28
_REF_FEEDBACK_FONT_SIZE    = 22
_REF_DEPTH_LABEL_FONT_SIZE = 14
_REF_DEPTH_PCT_FONT_SIZE   = 18

def _scaled_font_size(ref_size, frame_h):
    return max(10, int(round(ref_size * (frame_h / _REF_H))))

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

mp_pose = mp.solutions.pose

# ===================== SCORE DISPLAY =====================
def score_label(s: float) -> str:
    s = float(s)
    if s >= 9.5: return "Excellent"
    if s >= 8.5: return "Very good"
    if s >= 7.0: return "Good"
    if s >= 5.5: return "Fair"
    return "Needs work"

def display_half_str(x: float) -> str:
    q = round(float(x) * 2) / 2.0
    return str(int(round(q))) if abs(q - round(q)) < 1e-9 else f"{q:.1f}"

# ===================== BODY-ONLY (hide face) =====================
_FACE = {
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
    mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
    mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
}
_BODY_CONNS = tuple((a, b) for (a, b) in mp_pose.POSE_CONNECTIONS if a not in _FACE and b not in _FACE)
_BODY_POINTS = tuple(sorted({i for (a,b) in _BODY_CONNS for i in (a,b)}))

def _dyn_thickness(h):
    return max(2, int(round(h * 0.003))), max(3, int(round(h * 0.005)))

def draw_body_only_from_dict(frame, pts_norm, color=(255,255,255)):
    h, w = frame.shape[:2]
    line_t, dot_r = _dyn_thickness(h)
    for a, b in _BODY_CONNS:
        if a in pts_norm and b in pts_norm:
            ax, ay = int(pts_norm[a][0] * w), int(pts_norm[a][1] * h)
            bx, by = int(pts_norm[b][0] * w), int(pts_norm[b][1] * h)
            cv2.line(frame, (ax, ay), (bx, by), color, line_t, cv2.LINE_AA)
    for i in _BODY_POINTS:
        if i in pts_norm:
            x, y = int(pts_norm[i][0] * w), int(pts_norm[i][1] * h)
            cv2.circle(frame, (x, y), dot_r, color, -1, cv2.LINE_AA)
    return frame

# ===================== BACK CURVATURE =====================
def analyze_back_curvature(shoulder, hip, head_like, max_angle_deg=20.0, min_head_dist_ratio=0.35):
    torso_vec = shoulder - hip
    head_vec = head_like - shoulder
    torso_nrm = np.linalg.norm(torso_vec) + 1e-9
    head_nrm = np.linalg.norm(head_vec) + 1e-9
    if head_nrm < (min_head_dist_ratio * torso_nrm):
        return 0.0, False
    cosang = float(np.clip(np.dot(torso_vec, head_vec) / (torso_nrm * head_nrm), -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosang))
    return angle_deg, angle_deg > max_angle_deg

# ===================== Deadlift thresholds =====================
HINGE_START_THRESH      = 0.08
STAND_DELTA_TARGET      = 0.025
END_THRESH              = 0.035
MIN_FRAMES_BETWEEN_REPS = 10
PROG_ALPHA              = 0.35
LEG_BACK_MISMATCH_MIN_FRAMES = 6
LEG_BACK_ANGLE_FAST_DEG = 1.0
LEG_BACK_ANGLE_SLOW_DEG = 0.15

TIP_MIN_ECC_S    = 0.35
TIP_MIN_TOP_S    = 0.15
TIP_SMALL_ROM_LO = 0.035
TIP_SMALL_ROM_HI = 0.055

def choose_deadlift_tip(down_s, top_s, rom):
    if down_s is not None and down_s < TIP_MIN_ECC_S:
        return "Slow the lowering to ~2–3s for more hypertrophy"
    if top_s is not None and top_s < TIP_MIN_TOP_S:
        return "Squeeze the glutes for a brief hold at the top"
    if rom is not None and TIP_SMALL_ROM_LO <= rom <= TIP_SMALL_ROM_HI:
        return "Hinge a bit deeper within comfort"
    return "Keep the bar close and move smoothly"

def angle_deg(a, b, c):
    ba = a - b; bc = c - b
    nrm = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = float(np.clip(np.dot(ba, bc) / nrm, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))

# ===================== Kalman Leg Tracker =====================
LEG_VIS_THR    = 0.55
LEG_MAX_MAH    = 6.0
LEG_MISS_FLOOR = 6
LEG_SIDE_XSPAN = 0.25

class _KF:
    def __init__(self, q=1e-3, r=6e-3):
        self.x = np.zeros((4,1), dtype=float)
        self.P = np.eye(4)*1.0
        self.Q = np.eye(4)*q
        self.R = np.eye(2)*r
        self.F = np.eye(4)
        self.H = np.zeros((2,4)); self.H[0,0]=1; self.H[1,1]=1
        self.inited = False
    def predict(self, dt):
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        z = np.array(z, dtype=float).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return float(y.T @ np.linalg.inv(S) @ y)

class KalmanLegTracker:
    def __init__(self, side="right"):
        self.side = side
        PL = mp.solutions.pose.PoseLandmark
        self.hip   = PL.RIGHT_HIP.value   if side=="right" else PL.LEFT_HIP.value
        self.knee  = PL.RIGHT_KNEE.value  if side=="right" else PL.LEFT_KNEE.value
        self.ankle = PL.RIGHT_ANKLE.value if side=="right" else PL.LEFT_ANKLE.value
        self.heel  = PL.RIGHT_HEEL.value  if side=="right" else PL.LEFT_HEEL.value
        self.foot  = PL.RIGHT_FOOT_INDEX.value if side=="right" else PL.LEFT_FOOT_INDEX.value
        self.idxs  = [self.knee, self.ankle, self.heel, self.foot]
        self.kf    = {i: _KF() for i in self.idxs}
        self.miss  = {i: 0 for i in self.idxs}
        self.floor_y_med = None
        self.floor_hist  = []

    @staticmethod
    def _near_hands(lm, pt):
        PL = mp.solutions.pose.PoseLandmark
        for h in (PL.RIGHT_ELBOW.value, PL.RIGHT_WRIST.value, PL.LEFT_ELBOW.value, PL.LEFT_WRIST.value):
            if lm[h].visibility > 0.5:
                if math.hypot(pt[0]-lm[h].x, pt[1]-lm[h].y) < 0.06:
                    return True
        return False

    def _update_floor(self, lm):
        cand = []
        for idx in (self.ankle, self.heel, self.foot):
            if lm[idx].visibility > 0.6:
                cand.append(lm[idx].y)
        if cand:
            y = sorted(cand)[-1]
            self.floor_hist.append(y)
            if len(self.floor_hist) > 25: self.floor_hist.pop(0)
            self.floor_y_med = float(np.median(self.floor_hist))

    def update(self, lm, dt, occlusion_boxes_norm=None):
        self._update_floor(lm)
        out = {}
        hip_x = lm[self.hip].x
        def _inside_any_bbox(p):
            if not occlusion_boxes_norm: return False
            x, y = p
            for (x1,y1,x2,y2) in occlusion_boxes_norm:
                dx=(x2-x1)*0.08; dy=(y2-y1)*0.08
                if (x1-dx)<=x<=(x2+dx) and (y1-dy)<=y<=(y2+dy): return True
            return False
        for idx in self.idxs:
            kf = self.kf[idx]
            if not kf.inited:
                if lm[idx].visibility >= LEG_VIS_THR:
                    kf.x[:2,0] = [lm[idx].x, lm[idx].y]
                    kf.x[2:,0] = [0.0, 0.0]
                    kf.inited = True
                else:
                    continue
            kf.predict(dt)
            meas_ok = False
            if lm[idx].visibility >= LEG_VIS_THR:
                z = (lm[idx].x, lm[idx].y)
                if (not self._near_hands(lm, z)) and (abs(z[0]-hip_x) <= LEG_SIDE_XSPAN) and (not _inside_any_bbox(z)):
                    maha2 = kf.update(z)
                    if maha2 <= (LEG_MAX_MAH**2): meas_ok = True
            if meas_ok:
                self.miss[idx] = 0
            else:
                self.miss[idx] += 1
                if self.miss[idx] >= LEG_MISS_FLOOR and self.floor_y_med is not None:
                    kf.x[1,0] = 0.8*kf.x[1,0] + 0.2*self.floor_y_med
            out[idx] = (float(kf.x[0,0]), float(kf.x[1,0]))
        return out

# ===================== YOLOv8n-ONNX Detector (optional) =====================
class YoloOccluderDetector:
    def __init__(self, onnx_path, providers=None, input_size=640, conf_thres=0.25, iou_thres=0.45, occluder_class_ids=None):
        if not _ORT_AVAILABLE: raise RuntimeError("onnxruntime not available")
        if not os.path.exists(onnx_path): raise FileNotFoundError(onnx_path)
        self.sess = ort.InferenceSession(onnx_path, providers=providers or ort.get_available_providers())
        self.inp_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.imgsz = int(input_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres  = float(iou_thres)
        self.occluder_class_ids = occluder_class_ids

    @staticmethod
    def _letterbox(img, new_shape=640, color=(114,114,114)):
        shape = img.shape[:2]
        if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
        r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
        new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
        dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
        dw, dh = dw//2, dh//2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, dh,dh,dw,dw, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    @staticmethod
    def _nms(boxes, scores, iou_thres=0.45):
        if len(boxes) == 0: return []
        boxes = boxes.astype(np.float32)
        x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2-x1+1)*(y2-y1+1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1=np.maximum(x1[i], x1[order[1:]]); yy1=np.maximum(y1[i], y1[order[1:]])
            xx2=np.minimum(x2[i], x2[order[1:]]); yy2=np.minimum(y2[i], y2[order[1:]])
            w=np.maximum(0, xx2-xx1+1); hh=np.maximum(0, yy2-yy1+1)
            iou = (w*hh) / (areas[i] + areas[order[1:]] - w*hh + 1e-9)
            order = order[np.where(iou <= iou_thres)[0]+1]
        return keep

    def infer(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]
        img, r, (dw, dh) = self._letterbox(frame_bgr, self.imgsz)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img_rgb = np.transpose(img_rgb,(2,0,1))[None]
        out = self.sess.run([self.out_name], {self.inp_name: img_rgb})[0]
        if out.ndim==3 and out.shape[0]==1:
            o=out[0]
            if o.shape[0] in (84,85):
                num=o.shape[1]; xywh=o[0:4,:].T
                if o.shape[0]>=85:
                    obj=o[4,:].reshape(num,1); cls=o[5:,:].T
                    cls_id=np.argmax(cls,axis=1); conf=(obj*cls.max(axis=1)).flatten()
                else:
                    cls=o[4:,:].T; cls_id=np.argmax(cls,axis=1); conf=cls.max(axis=1).flatten()
            else:
                xywh=o[:,0:4]
                if o.shape[1]>=85:
                    obj=o[:,4:5]; cls=o[:,5:]
                    cls_id=np.argmax(cls,axis=1); conf=(obj[:,0]*cls.max(axis=1)).flatten()
                else:
                    cls=o[:,4:]; cls_id=np.argmax(cls,axis=1); conf=cls.max(axis=1).flatten()
        else:
            return []
        m=conf>=self.conf_thres
        if not np.any(m): return []
        xywh=xywh[m]; conf=conf[m]; cls_id=cls_id[m]
        if self.occluder_class_ids is not None:
            sel=np.isin(cls_id,list(self.occluder_class_ids))
            xywh,conf,cls_id=xywh[sel],conf[sel],cls_id[sel]
            if len(xywh)==0: return []
        xyxy=np.zeros_like(xywh)
        xyxy[:,0]=xywh[:,0]-xywh[:,2]/2; xyxy[:,1]=xywh[:,1]-xywh[:,3]/2
        xyxy[:,2]=xywh[:,0]+xywh[:,2]/2; xyxy[:,3]=xywh[:,1]+xywh[:,3]/2
        keep=self._nms(xyxy,conf,self.iou_thres); xyxy=xyxy[keep]
        xyxy[:,[0,2]]-=dw; xyxy[:,[1,3]]-=dh; xyxy/=r
        xyxy[:,0]=np.clip(xyxy[:,0],0,w0-1); xyxy[:,2]=np.clip(xyxy[:,2],0,w0-1)
        xyxy[:,1]=np.clip(xyxy[:,1],0,h0-1); xyxy[:,3]=np.clip(xyxy[:,3],0,h0-1)
        return [(float(x1)/w0,float(y1)/h0,float(x2)/w0,float(y2)/h0) for x1,y1,x2,y2 in xyxy]

# ===================== Overlay (aligned to good_morning standard) =====================
def _wrap2(draw, text, font, maxw):
    words = text.split()
    if not words: return [""]
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
        if len(lines) == 2: break
    if cur and len(lines) < 2: lines.append(cur)
    if len(lines) >= 2 and draw.textlength(lines[-1], font=font) > maxw:
        last = lines[-1] + "…"
        while draw.textlength(last, font=font) > maxw and len(last) > 1:
            last = last[:-2] + "…"
        lines[-1] = last
    return lines

def draw_overlay(frame, reps=0, feedback=None, progress_pct=0.0):
    """
    Draw overlay on frame at its ACTUAL resolution.
    Fonts scale proportionally to frame height — identical to good_morning standard.
    """
    h, w, _ = frame.shape

    # ── Scale fonts to THIS frame's height (same as good_morning) ──
    reps_font_size        = _scaled_font_size(_REF_REPS_FONT_SIZE,        h)
    feedback_font_size    = _scaled_font_size(_REF_FEEDBACK_FONT_SIZE,    h)
    depth_label_font_size = _scaled_font_size(_REF_DEPTH_LABEL_FONT_SIZE, h)
    depth_pct_font_size   = _scaled_font_size(_REF_DEPTH_PCT_FONT_SIZE,   h)

    _REPS_FONT        = _load_font(FONT_PATH, reps_font_size)
    _FEEDBACK_FONT    = _load_font(FONT_PATH, feedback_font_size)
    _DEPTH_LABEL_FONT = _load_font(FONT_PATH, depth_label_font_size)
    _DEPTH_PCT_FONT   = _load_font(FONT_PATH, depth_pct_font_size)

    pct = float(np.clip(progress_pct, 0, 1))
    bg_alpha_val = int(round(255 * BAR_BG_ALPHA))

    ref_h  = max(int(h * 0.06), int(reps_font_size * 1.6))
    radius = int(ref_h * DONUT_RADIUS_SCALE)
    thick  = max(3, int(radius * DONUT_THICKNESS_FRAC))
    margin = 12
    cx = w - margin - radius
    cy = max(ref_h + radius // 8, radius + thick // 2 + 2)

    # Build overlay as RGBA numpy array directly (no HD upscale needed)
    overlay_np = np.zeros((h, w, 4), dtype=np.uint8)

    pad_x, pad_y = 10, 6
    tmp_pil  = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_pil)
    txt = f"Reps: {int(reps)}"
    tw  = tmp_draw.textlength(txt, font=_REPS_FONT)
    thh = _REPS_FONT.size
    cv2.rectangle(overlay_np, (0, 0), (int(tw + 2*pad_x), int(thh + 2*pad_y)), (0,0,0,bg_alpha_val), -1)

    # Donut background ring
    cv2.circle(overlay_np, (cx, cy), radius, (*DEPTH_RING_BG, 255), thick, cv2.LINE_AA)
    # Donut progress arc
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    if end_ang != start_ang:
        cv2.ellipse(overlay_np, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                    (*DEPTH_COLOR, 255), thick, cv2.LINE_AA)

    # Feedback background
    fb_y0 = 0
    fb_lines = []
    fb_pad_x = fb_pad_y = line_gap = line_h = 0
    if feedback:
        safe_margin = max(6, int(h * 0.02))
        fb_pad_x, fb_pad_y, line_gap = 12, 8, 4
        max_text_w = int(w - 2*fb_pad_x - 20)
        fb_lines = _wrap2(tmp_draw, feedback, _FEEDBACK_FONT, max_text_w)
        line_h   = _FEEDBACK_FONT.size + 6
        block_h  = 2*fb_pad_y + len(fb_lines)*line_h + (len(fb_lines)-1)*line_gap
        fb_y0    = max(0, h - safe_margin - block_h)
        y1       = h - safe_margin
        cv2.rectangle(overlay_np, (0, fb_y0), (w, y1), (0,0,0,bg_alpha_val), -1)

    # Composite via PIL for crisp text
    overlay_pil = Image.fromarray(overlay_np, mode="RGBA")
    draw = ImageDraw.Draw(overlay_pil)

    # Reps text
    draw.text((pad_x, pad_y - 1), txt, font=_REPS_FONT, fill=(255,255,255,255))

    # Donut labels
    gap  = max(2, int(radius * 0.10))
    by   = cy - (_DEPTH_LABEL_FONT.size + gap + _DEPTH_PCT_FONT.size) // 2
    label   = "DEPTH"
    pct_txt = f"{int(pct * 100)}%"
    lw = draw.textlength(label,   font=_DEPTH_LABEL_FONT)
    pw = draw.textlength(pct_txt, font=_DEPTH_PCT_FONT)
    draw.text((cx - int(lw//2), by),                              label,   font=_DEPTH_LABEL_FONT, fill=(255,255,255,255))
    draw.text((cx - int(pw//2), by + _DEPTH_LABEL_FONT.size + gap), pct_txt, font=_DEPTH_PCT_FONT,   fill=(255,255,255,255))

    # Feedback text
    if feedback and fb_lines:
        ty = fb_y0 + fb_pad_y
        for ln in fb_lines:
            tw2 = draw.textlength(ln, font=_FEEDBACK_FONT)
            tx  = max(fb_pad_x, (w - int(tw2)) // 2)
            draw.text((tx, ty), ln, font=_FEEDBACK_FONT, fill=(255,255,255,255))
            ty += line_h + line_gap

    # Alpha-blend overlay onto frame
    overlay_rgba  = np.array(overlay_pil)
    alpha         = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr   = overlay_rgba[:, :, [2,1,0]].astype(np.float32)
    out_f         = frame.astype(np.float32) * (1.0 - alpha) + overlay_bgr * alpha
    return out_f.astype(np.uint8)

# ===================== MAIN =====================
def run_deadlift_analysis(video_path,
                          frame_skip=3,
                          scale=0.4,
                          output_path="deadlift_analyzed.mp4",
                          feedback_path="deadlift_feedback.txt",
                          detector_onnx_path="barbell_plates_yolov8n.onnx",
                          detector_input_size=640,
                          detector_conf_thres=0.25,
                          detector_iou_thres=0.45,
                          detector_class_ids=None,
                          return_video=True,
                          fast_mode=None):
    if fast_mode is True:
        return_video = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    detector = None
    if detector_onnx_path and _ORT_AVAILABLE and os.path.exists(detector_onnx_path):
        try:
            detector = YoloOccluderDetector(
                detector_onnx_path,
                input_size=detector_input_size,
                conf_thres=detector_conf_thres,
                iou_thres=detector_iou_thres,
                occluder_class_ids=detector_class_ids
            )
        except Exception:
            detector = None

    PL = mp.solutions.pose.PoseLandmark
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = None
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    frame_dt = 1.0 / float(effective_fps)  # renamed to avoid shadowing loop variable

    # ── Read original dimensions (same pattern as good_morning) ──
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    counter = good_reps = bad_reps = 0
    all_scores, reps_report = [], []

    rep = False
    last_rep_frame = -999
    frame_idx = 0

    top_ref    = STAND_DELTA_TARGET
    bottom_est = None
    prog       = 1.0

    BACK_MIN_FRAMES = max(2, int(0.25 / frame_dt))
    back_frames = leg_back_leg_fast = leg_back_back_fast = 0

    down_frames = up_frames = top_hold_frames = 0
    prev_progress  = None
    prev_knee_angle  = None
    prev_torso_angle = None

    right_leg = KalmanLegTracker("right")
    left_leg  = KalmanLegTracker("left")

    with mp.solutions.pose.Pose(model_complexity=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue

            # Small working frame for pose estimation (fast)
            work = cv2.resize(frame, (0,0), fx=scale, fy=scale) if scale != 1.0 else frame.copy()

            # ── Open VideoWriter at ORIGINAL resolution (key fix) ──
            if return_video and out_vid is None:
                out_vid = cv2.VideoWriter(output_path, fourcc, effective_fps, (orig_w, orig_h))

            occl_boxes = []
            if detector is not None:
                try: occl_boxes = detector.infer(work)
                except Exception: occl_boxes = []

            rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            rt_fb = None
            if not res.pose_landmarks:
                if return_video and out_vid is not None:
                    # Upscale to original res then draw overlay
                    frame_out = cv2.resize(work, (orig_w, orig_h))
                    frame_drawn = draw_overlay(frame_out, reps=counter, feedback=None, progress_pct=prog)
                    out_vid.write(frame_drawn)
                continue

            try:
                lm = res.pose_landmarks.landmark
                shoulder = np.array([lm[PL.RIGHT_SHOULDER.value].x, lm[PL.RIGHT_SHOULDER.value].y], dtype=float)
                hip      = np.array([lm[PL.RIGHT_HIP.value].x,      lm[PL.RIGHT_HIP.value].y],      dtype=float)

                head = None
                for idx in (PL.RIGHT_EAR.value, PL.LEFT_EAR.value, PL.NOSE.value):
                    if lm[idx].visibility > 0.5:
                        head = np.array([lm[idx].x, lm[idx].y], dtype=float); break

                right_leg_pts = right_leg.update(lm, frame_dt, occlusion_boxes_norm=occl_boxes)
                left_leg_pts  = left_leg.update(lm,  frame_dt, occlusion_boxes_norm=occl_boxes)

                delta_x = abs(hip[0] - shoulder[0])
                if delta_x < (STAND_DELTA_TARGET * 1.4):
                    top_ref = 0.9*top_ref + 0.1*delta_x

                if head is not None:
                    mid_spine = (shoulder + hip) * 0.5 * 0.4 + head * 0.6
                    _, rounded = analyze_back_curvature(shoulder, hip, mid_spine)
                    if delta_x < (HINGE_START_THRESH * 1.05):
                        rounded = False
                else:
                    rounded = False
                back_frames = back_frames + 1 if rounded else max(0, back_frames - 1)

                if (not rep) and (delta_x > HINGE_START_THRESH) and (frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS):
                    rep = True
                    bottom_est = delta_x
                    back_frames = 0
                    leg_back_leg_fast = leg_back_back_fast = 0
                    down_frames = up_frames = top_hold_frames = 0
                    prev_progress = prog
                    prev_knee_angle = prev_torso_angle = None

                if rep:
                    bottom_est = max(bottom_est or delta_x, delta_x)
                    denom = max(1e-4, bottom_est - top_ref)
                    pr    = float(np.clip(1.0 - ((delta_x - top_ref) / denom), 0.0, 1.0))
                    prog  = PROG_ALPHA * pr + (1 - PROG_ALPHA) * prog

                    if prev_progress is not None:
                        if prog < prev_progress - 1e-4:   down_frames     += 1
                        elif prog > prev_progress + 1e-4: up_frames       += 1
                    prev_progress = prog

                    if prog >= 0.90: top_hold_frames += 1
                    if rounded:      rt_fb = "Try to keep your back a bit straighter"

                    kv = (lm[PL.RIGHT_KNEE.value].visibility     > 0.5 and
                          lm[PL.RIGHT_ANKLE.value].visibility    > 0.5 and
                          lm[PL.RIGHT_HIP.value].visibility      > 0.5 and
                          lm[PL.RIGHT_SHOULDER.value].visibility > 0.5)
                    if kv:
                        knee  = np.array([lm[PL.RIGHT_KNEE.value].x,  lm[PL.RIGHT_KNEE.value].y],  dtype=float)
                        ankle = np.array([lm[PL.RIGHT_ANKLE.value].x, lm[PL.RIGHT_ANKLE.value].y], dtype=float)
                        k_ang = angle_deg(hip, knee, ankle)
                        t_ang = angle_deg(shoulder, hip, hip + np.array([0.0, -1.0]))
                        if prev_knee_angle is not None and prev_torso_angle is not None and prog > 0.1:
                            dk = k_ang - prev_knee_angle
                            dta = prev_torso_angle - t_ang
                            if dk  > LEG_BACK_ANGLE_FAST_DEG and dta < LEG_BACK_ANGLE_SLOW_DEG: leg_back_leg_fast  += 1
                            elif dta > LEG_BACK_ANGLE_FAST_DEG and dk < LEG_BACK_ANGLE_SLOW_DEG: leg_back_back_fast += 1
                        prev_knee_angle = k_ang
                        prev_torso_angle = t_ang
                else:
                    prog = PROG_ALPHA * 1.0 + (1 - PROG_ALPHA) * prog

                if rep and (delta_x < END_THRESH):
                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        penalty = 0.0; fb = []
                        if delta_x > (top_ref + 0.02):
                            fb.append("Try to finish more upright"); penalty += 1.0
                        if back_frames >= BACK_MIN_FRAMES:
                            fb.append("Try to keep your back a bit straighter"); penalty += 1.5
                        if leg_back_leg_fast >= LEG_BACK_MISMATCH_MIN_FRAMES:
                            fb.append("Drive the back up with the legs (avoid legs shooting up first)"); penalty += 1.0
                        if leg_back_back_fast >= LEG_BACK_MISMATCH_MIN_FRAMES:
                            fb.append("Let the legs push more (avoid back rising faster than the legs)"); penalty += 1.0

                        score = round(max(4, 10 - penalty) * 2) / 2
                        moved_enough = (bottom_est - delta_x) > 0.05
                        if moved_enough:
                            counter += 1
                            if score >= 9.5: good_reps += 1
                            else:            bad_reps  += 1
                            down_s = (down_frames * frame_dt) if down_frames is not None else None
                            top_s  = (top_hold_frames * frame_dt) if top_hold_frames is not None else None
                            rom    = (bottom_est - top_ref) if (bottom_est is not None and top_ref is not None) else None
                            tip_str = choose_deadlift_tip(down_s, top_s, rom)
                            rep_fb = fb if score < 10.0 else []
                            reps_report.append({
                                "rep_index":      counter,
                                "score":          float(score),
                                "score_display":  display_half_str(score),
                                "feedback":       rep_fb,
                                "tip":            tip_str
                            })
                            all_scores.append(score)
                        last_rep_frame = frame_idx

                    rep = False; bottom_est = None; back_frames = 0
                    leg_back_leg_fast = leg_back_back_fast = 0
                    prev_progress = None; down_frames = up_frames = top_hold_frames = 0
                    prev_knee_angle = prev_torso_angle = None

                # ── Draw at ORIGINAL resolution (key fix, same as good_morning) ──
                if return_video:
                    # Upscale work frame to original res first
                    frame_out = cv2.resize(work, (orig_w, orig_h))

                    # Build normalised landmark dict (coords already normalised 0–1)
                    pts_draw = {
                        PL.RIGHT_SHOULDER.value: tuple(shoulder),
                        PL.RIGHT_HIP.value:      tuple(hip),
                    }
                    for d in (right_leg_pts, left_leg_pts):
                        for idx, pt in d.items():
                            if occl_boxes:
                                x, y = pt
                                if any(x1<=x<=x2 and y1<=y<=y2 for (x1,y1,x2,y2) in occl_boxes):
                                    continue
                            pts_draw[idx] = pt

                    frame_out = draw_body_only_from_dict(frame_out, pts_draw)
                    frame_out = draw_overlay(frame_out, reps=counter, feedback=rt_fb, progress_pct=prog)
                    out_vid.write(frame_out)

            except Exception:
                if return_video and out_vid is not None:
                    frame_out = cv2.resize(work, (orig_w, orig_h))
                    out_vid.write(draw_overlay(frame_out, reps=counter, feedback=None, progress_pct=prog))
                continue

    cap.release()
    if return_video and out_vid: out_vid.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)

    seen = set(); feedback_list = []
    for r in reps_report:
        if float(r.get("score") or 0.0) >= 10.0: continue
        for msg in (r.get("feedback") or []):
            if msg and msg not in seen:
                seen.add(msg); feedback_list.append(msg)
    if not feedback_list:
        feedback_list = ["Great form! Keep it up 💪"]

    final_video_path = ""
    if return_video:
        encoded_path = output_path.replace(".mp4", "_encoded.mp4")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", output_path,
                "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                encoded_path
            ], check=False, capture_output=True)
            if os.path.exists(output_path) and os.path.exists(encoded_path):
                os.remove(output_path)
        except Exception:
            pass
        final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")

    return {
        "squat_count": int(counter),
        "technique_score": float(technique_score),
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": int(good_reps),
        "bad_reps": int(bad_reps),
        "feedback": feedback_list,
        "reps": reps_report,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }

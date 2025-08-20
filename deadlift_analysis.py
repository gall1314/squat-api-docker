# -*- coding: utf-8 -*-
# deadlift_analysis.py — v2.1 (robust IRL)
# - Finite-state machine (Top→Down→Up) with hysteresis
# - Dual gating: torso-hinge + bar proxy (wrists / YOLO optional)
# - Kalman tracking for legs AND bar center (vertical)
# - Rejects unrelated/noisy motion; counts only real ROM
# - Back rounding only during active rep
# - Overlay sizes aligned to squat standard

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
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
DEPTH_LABEL_FONT_SIZE = 14
DEPTH_PCT_FONT_SIZE   = 18

def _font(path,size):
    try: return ImageFont.truetype(path,size)
    except: return ImageFont.load_default()

REPS_FONT        = _font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

mp_pose = mp.solutions.pose

# ===================== SCORE LABELS =====================
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

# ===================== BODY ONLY DRAW =====================
_FACE = {
    mp_pose.PoseLandmark.NOSE.value,
    mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
    mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
    mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
    mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value,
}
_BODY_CONNS = tuple((a, b) for (a, b) in mp_pose.POSE_CONNECTIONS if a not in _FACE and b not in _FACE)
_BODY_POINTS = tuple(sorted({i for (a,b) in _BODY_CONNS for i in (a,b)}))

def draw_body_only_from_dict(frame, pts_norm, color=(255,255,255)):
    h, w = frame.shape[:2]
    for a, b in _BODY_CONNS:
        if a in pts_norm and b in pts_norm:
            ax, ay = int(pts_norm[a][0] * w), int(pts_norm[a][1] * h)
            bx, by = int(pts_norm[b][0] * w), int(pts_norm[b][1] * h)
            cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)
    for i in _BODY_POINTS:
        if i in pts_norm:
            x, y = int(pts_norm[i][0] * w), int(pts_norm[i][1] * h)
            cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
    return frame

# ===================== UTILITIES =====================
def analyze_back_curvature(shoulder, hip, head_like, threshold=0.04):
    line_vec = hip - shoulder
    nrm = np.linalg.norm(line_vec) + 1e-9
    u = line_vec / nrm
    proj_len = np.dot(head_like - shoulder, u)
    proj = shoulder + proj_len * u
    off = head_like - proj
    signed = float(np.sign(off[1]) * -1 * np.linalg.norm(off))  # inward negative
    return signed, signed < -threshold

class EMA:
    def __init__(self, alpha=0.25, init=None):
        self.a = float(alpha); self.v = init; self.ready = init is not None
    def update(self, x):
        x = float(x)
        if not self.ready:
            self.v = x; self.ready=True
        else:
            self.v = self.a*x + (1-self.a)*self.v
        return self.v

# ===================== THRESHOLDS =====================
# hinge (torso) via |delta_x(hip-shoulder)|
HINGE_START_THRESH     = 0.08
HINGE_END_THRESH       = 0.035
HINGE_TOP_TRACK_ALPHA  = 0.10  # adapt upright slowly

# bar proxy (wrists)
WRIST_VIS_THR          = 0.55
GRIP_WIDTH_MIN         = 0.10   # normalized fraction of image width
GRIP_WIDTH_MAX         = 0.55
GRIP_WIDTH_DRIFT_MAX   = 0.12   # max relative drift of grip width within rep

# rep FSM
MIN_FRAMES_DOWN        = 4
MIN_FRAMES_UP          = 4
MIN_FRAMES_TOTAL       = 12
MIN_FRAMES_BW_REPS     = 10
ROM_MIN_HINGE          = 0.055   # minimal hinge ROM
ROM_MIN_BAR            = 0.055   # minimal bar vertical ROM (if bar proxy valid)
PROG_ALPHA             = 0.35    # donut smoothing

# back rounding inside rep
BACK_MIN_TIME_S        = 0.25

# tips thresholds
TIP_MIN_ECC_S          = 0.35
TIP_MIN_TOP_S          = 0.15
TIP_SMALL_ROM_LO       = 0.035
TIP_SMALL_ROM_HI       = 0.055

def choose_deadlift_tip(down_s, top_s, rom):
    if down_s is not None and down_s < TIP_MIN_ECC_S:
        return "Slow the lowering to ~2–3s for more hypertrophy"
    if top_s is not None and top_s < TIP_MIN_TOP_S:
        return "Squeeze the glutes briefly at the top"
    if rom is not None and TIP_SMALL_ROM_LO <= rom <= TIP_SMALL_ROM_HI:
        return "Hinge a bit deeper within comfort"
    return "Keep the bar close and move smoothly"

# ===================== KALMAN (legs + bar-y) =====================
class _KF:
    def __init__(self, q=1e-3, r=6e-3):
        self.x = np.zeros((4,1), dtype=float)  # [x, y, vx, vy]
        self.P = np.eye(4)*1.0
        self.Q = np.eye(4)*q
        self.R = np.eye(2)*r
        self.F = np.eye(4); self.H = np.zeros((2,4)); self.H[0,0]=1; self.H[1,1]=1
        self.inited=False
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
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        maha2 = float(y.T @ np.linalg.inv(S) @ y)
        return maha2

class KalmanLegTracker:
    LEG_VIS_THR    = 0.55
    LEG_MAX_MAH    = 6.0
    LEG_MISS_FLOOR = 6
    LEG_SIDE_XSPAN = 0.25
    def __init__(self, side="right"):
        PL = mp.solutions.pose.PoseLandmark
        self.side = side
        self.hip  = PL.RIGHT_HIP.value   if side=="right" else PL.LEFT_HIP.value
        self.knee = PL.RIGHT_KNEE.value  if side=="right" else PL.LEFT_KNEE.value
        self.ankle= PL.RIGHT_ANKLE.value if side=="right" else PL.LEFT_ANKLE.value
        self.heel = PL.RIGHT_HEEL.value  if side=="right" else PL.LEFT_HEEL.value
        self.foot = PL.RIGHT_FOOT_INDEX.value if side=="right" else PL.LEFT_FOOT_INDEX.value
        self.idxs = [self.knee, self.ankle, self.heel, self.foot]
        self.kf   = {i:_KF() for i in self.idxs}
        self.miss = {i:0 for i in self.idxs}
        self.floor_y_med = None
        self.floor_hist = []
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
            for (x1, y1, x2, y2) in occlusion_boxes_norm:
                dx = (x2 - x1) * 0.08; dy = (y2 - y1) * 0.08
                if (x1-dx) <= x <= (x2+dx) and (y1-dy) <= y <= (y2+dy):
                    return True
            return False
        for idx in self.idxs:
            kf = self.kf[idx]
            if not kf.inited:
                if lm[idx].visibility >= self.LEG_VIS_THR:
                    kf.x[:2,0] = [lm[idx].x, lm[idx].y]
                    kf.x[2:,0] = [0.0, 0.0]
                    kf.inited=True
                else:
                    continue
            kf.predict(dt)
            meas_ok=False
            if lm[idx].visibility >= self.LEG_VIS_THR:
                z = (lm[idx].x, lm[idx].y)
                if (not self._near_hands(lm, z)) and (abs(z[0]-hip_x) <= self.LEG_SIDE_XSPAN) and (not _inside_any_bbox(z)):
                    maha2 = kf.update(z)
                    if maha2 <= (self.LEG_MAX_MAH**2): meas_ok=True
            if meas_ok:
                self.miss[idx]=0
            else:
                self.miss[idx]+=1
                if self.miss[idx] >= self.LEG_MISS_FLOOR and self.floor_y_med is not None:
                    kf.x[1,0] = 0.8*kf.x[1,0] + 0.2*self.floor_y_med
            out[idx] = (float(kf.x[0,0]), float(kf.x[1,0]))
        return out

class BarYTracker:
    """Tracks bar vertical position via wrists proxy (avg of wrists)."""
    def __init__(self):
        self.kf = _KF(q=8e-4, r=9e-4)
        self.ema_y = EMA(0.2)
        self.valid = False
        self.top_ref_ema = EMA(0.05)   # learns 'lockout' bar height
        self.width0 = None
        self.width_max_drift = GRIP_WIDTH_DRIFT_MAX
    def reset_rep(self):
        self.width0 = None
    def update(self, lm):
        PL = mp_pose.PoseLandmark
        lw, rw = PL.LEFT_WRIST.value, PL.RIGHT_WRIST.value
        if lm[lw].visibility < WRIST_VIS_THR or lm[rw].visibility < WRIST_VIS_THR:
            self.valid = False; return None, False
        xL, yL = lm[lw].x, lm[lw].y
        xR, yR = lm[rw].x, lm[rw].y
        grip_w = abs(xR - xL)
        if not (GRIP_WIDTH_MIN <= grip_w <= GRIP_WIDTH_MAX):
            self.valid = False; return None, False
        # lock drift within rep
        if self.width0 is None: self.width0 = grip_w
        else:
            if abs(grip_w - self.width0) > self.width_max_drift * max(1e-3, self.width0):
                self.valid = False; return None, False
        y = 0.5*(yL + yR)
        # Kalman on (x: dummy mid-x, y: vertical)
        cx = 0.5*(xL + xR)
        if not self.kf.inited:
            self.kf.x[:2,0] = [cx, y]; self.kf.x[2:,0] = [0.0, 0.0]; self.kf.inited=True
        else:
            self.kf.predict(1/25.0)
            self.kf.update([cx, y])
        y_f = self.ema_y.update(float(self.kf.x[1,0]))
        self.top_ref_ema.update(y_f)  # learns over time
        self.valid = True
        return y_f, True
    def top_ref(self):
        return self.top_ref_ema.v if self.top_ref_ema.ready else None

# ===================== YOLO OCCLUDER (OPTIONAL) =====================
class YoloOccluderDetector:
    def __init__(self, onnx_path, providers=None, input_size=640, conf_thres=0.25, iou_thres=0.45, occluder_class_ids=None):
        if not _ORT_AVAILABLE: raise RuntimeError("onnxruntime not available")
        if not os.path.exists(onnx_path): raise FileNotFoundError(onnx_path)
        self.sess = ort.InferenceSession(onnx_path, providers=providers or ort.get_available_providers())
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        self.imgsz = int(input_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
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
        img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
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
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[i if order.size==1 else order[1:]])
            yy2 = np.minimum(y2[i], y2[i if order.size==1 else order[1:]])
            w = np.maximum(0, xx2-xx1+1); h = np.maximum(0, yy2-yy1+1)
            inter = w*h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds+1]
        return keep
    def infer(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]
        img, r, (dw, dh) = self._letterbox(frame_bgr, self.imgsz)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        rgb = np.transpose(rgb, (2,0,1))[None]
        out = self.sess.run([self.out], {self.inp: rgb})[0]
        if out.ndim != 3 or out.shape[0] != 1: return []
        o = out[0]
        if o.shape[0] in (84,85):
            num = o.shape[1]
            xywh = o[0:4,:].T
            if o.shape[0] >= 85:
                obj = o[4,:].reshape(num,1)
                cls = o[5:,:].T
                cls_id = np.argmax(cls, axis=1)
                conf = (obj * cls.max(axis=1)).flatten()
            else:
                cls = o[4:,:].T
                cls_id = np.argmax(cls, axis=1)
                conf = cls.max(axis=1).flatten()
        else:
            xywh = o[:,0:4]
            if o.shape[1] >= 85:
                obj = o[:,4:5]
                cls = o[:,5:]
                cls_id = np.argmax(cls, axis=1)
                conf = (obj[:,0]*cls.max(axis=1)).flatten()
            else:
                cls = o[:,4:]
                cls_id = np.argmax(cls, axis=1)
                conf = cls.max(axis=1).flatten()
        m = conf >= self.conf_thres
        if not np.any(m): return []
        xywh = xywh[m]; conf = conf[m]; cls_id = cls_id[m]
        if self.occluder_class_ids is not None:
            sel = np.isin(cls_id, list(self.occluder_class_ids))
            xywh, conf, cls_id = xywh[sel], conf[sel], cls_id[sel]
            if len(xywh) == 0: return []
        xyxy = np.zeros_like(xywh)
        xyxy[:,0] = xywh[:,0] - xywh[:,2]/2
        xyxy[:,1] = xywh[:,1] - xywh[:,3]/2
        xyxy[:,2] = xywh[:,0] + xywh[:,2]/2
        xyxy[:,3] = xywh[:,1] + xywh[:,3]/2
        keep = self._nms(xyxy, conf, self.iou_thres)
        xyxy = xyxy[keep]
        xyxy[:,[0,2]] -= dw; xyxy[:,[1,3]] -= dh
        xyxy /= r
        xyxy[:,0] = np.clip(xyxy[:,0], 0, w0-1)
        xyxy[:,2] = np.clip(xyxy[:,2], 0, w0-1)
        xyxy[:,1] = np.clip(xyxy[:,1], 0, h0-1)
        xyxy[:,3] = np.clip(xyxy[:,3], 0, h0-1)
        boxes_norm=[]
        for x1,y1,x2,y2 in xyxy:
            boxes_norm.append((float(x1)/w0, float(y1)/h0, float(x2)/w0, float(y2)/h0))
        return boxes_norm

# ===================== OVERLAY =====================
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
    leftover = len(words) - sum(len(l.split()) for l in lines)
    if leftover > 0 and len(lines) >= 2:
        last = lines[-1] + "…"
        while draw.textlength(last, font=font) > maxw and len(last) > 1:
            last = last[:-2] + "…"
        lines[-1] = last
    return lines

def _donut(frame, c, r, t, p):
    p = float(np.clip(p, 0, 1))
    cx, cy = int(c[0]), int(c[1]); r = int(r); t = int(t)
    cv2.circle(frame, (cx,cy), r, DEPTH_RING_BG, t, cv2.LINE_AA)
    cv2.ellipse(frame, (cx,cy), (r,r), 0, -90, -90 + int(360*p), DEPTH_COLOR, t, cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, progress_pct=0.0, dbg_text=None):
    h, w, _ = frame.shape
    pil = Image.fromarray(frame); d = ImageDraw.Draw(pil)
    txt = f"Reps: {reps}"; padx, pady = 10, 6
    tw = d.textlength(txt, font=REPS_FONT); th = REPS_FONT.size
    x0, y0 = 0, 0; x1 = int(tw + 2*padx); y1 = int(th + 2*pady)
    top = frame.copy(); cv2.rectangle(top, (x0,y0), (x1,y1), (0,0,0), -1)
    frame = cv2.addWeighted(top, BAR_BG_ALPHA, frame, 1-BAR_BG_ALPHA, 0)
    pil = Image.fromarray(frame)
    ImageDraw.Draw(pil).text((x0+padx, y0+pady-1), txt, font=REPS_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # Donut (top-right)
    ref_h = max(int(h*0.06), int(REPS_FONT_SIZE*1.6))
    r = int(ref_h * DONUT_RADIUS_SCALE)
    thick = max(3, int(r * DONUT_THICKNESS_FRAC))
    m = 12; cx = w - m - r; cy = max(ref_h + r//8, r + thick//2 + 2)
    frame = _donut(frame, (cx,cy), r, thick, float(np.clip(progress_pct,0,1)))

    pil = Image.fromarray(frame); d = ImageDraw.Draw(pil)
    label = "DEPTH"; pct = f"{int(float(np.clip(progress_pct,0,1))*100)}%"
    lw = d.textlength(label, font=DEPTH_LABEL_FONT); pw = d.textlength(pct, font=DEPTH_PCT_FONT)
    gap = max(2, int(r*0.10)); base = cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size)//2
    d.text((cx-int(lw//2), base), label, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    d.text((cx-int(pw//2), base+DEPTH_LABEL_FONT.size+gap), pct, font=DEPTH_PCT_FONT, fill=(255,255,255))

    # bottom feedback
    if feedback:
        pil_fb = Image.fromarray(frame); dfb = ImageDraw.Draw(pil_fb)
        safe = max(6, int(h*0.02)); pad_x, pad_y, lg = 12, 8, 4
        maxw = int(w - 2*pad_x - 20)
        lines = _wrap2(dfb, feedback, FEEDBACK_FONT, maxw)
        lh = FEEDBACK_FONT.size + 6
        block = (2*pad_y) + len(lines)*lh + (len(lines)-1)*lg
        y0 = max(0, h - safe - block); y1 = h - safe
        over = frame.copy(); cv2.rectangle(over, (0,y0), (w,y1), (0,0,0), -1)
        frame = cv2.addWeighted(over, BAR_BG_ALPHA, frame, 1-BAR_BG_ALPHA, 0)
        pil_fb = Image.fromarray(frame); dfb = ImageDraw.Draw(pil_fb); ty = y0 + pad_y
        for ln in lines:
            t_w = dfb.textlength(ln, font=FEEDBACK_FONT); tx = max(pad_x, (w - int(t_w)) // 2)
            dfb.text((tx,ty), ln, font=FEEDBACK_FONT, fill=(255,255,255)); ty += lh + lg
        frame = np.array(pil_fb)

    # optional tiny debug (disabled by default)
    if dbg_text:
        cv2.putText(frame, dbg_text, (8, int(h*0.12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return frame

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
                          detector_class_ids=None,   # e.g., {0,1}; None = treat all as occluders
                          debug=False
                          ):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "squat_count": 0, "technique_score": 0.0,
            "technique_score_display": display_half_str(0.0),
            "technique_label": score_label(0.0),
            "good_reps": 0, "bad_reps": 0, "feedback": ["Could not open video"],
            "reps": [], "video_path": "", "feedback_path": feedback_path
        }

    # Optional YOLO occluder detector (רק לסינון נק’ מוסתרות; לא חייב)
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    # counters
    counter = good_reps = bad_reps = 0
    all_scores, reps_report, overall_feedback = [], [], []

    # FSM
    STATE_TOP, STATE_DOWN, STATE_UP = 0, 1, 2
    state = STATE_TOP
    last_rep_frame = -999
    frame_idx = 0

    # hinge calibration
    top_ref = EMA(alpha=HINGE_TOP_TRACK_ALPHA)
    bottom_est = None
    prog = 1.0

    # Bar tracker
    bar = BarYTracker()

    # back curvature window
    back_frames = 0
    BACK_MIN_FRAMES = max(2, int(BACK_MIN_TIME_S / dt))

    # tempo per rep
    down_frames = up_frames = top_hold_frames = 0

    # ROM bookkeeping
    hinge_min = hinge_max = None
    bar_min = bar_max = None

    # choose leg side automatically by visible ankle depth
    def auto_side(lm):
        left_vis  = lm[PL.LEFT_ANKLE.value].visibility
        right_vis = lm[PL.RIGHT_ANKLE.value].visibility
        return "left" if (left_vis > right_vis) else "right"

    right_leg = None
    left_leg  = None

    with mp.solutions.pose.Pose(model_complexity=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            if frame_idx % frame_skip != 0: continue
            if scale != 1.0: frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            h, w = frame.shape[:2]
            if out is None:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

            occl_boxes = []
            if detector is not None:
                try: occl_boxes = detector.infer(frame)
                except Exception: occl_boxes = []

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            rt_fb = None
            dbg = None

            if not res.pose_landmarks:
                frame = draw_overlay(frame, reps=counter, feedback=None, progress_pct=prog, dbg_text=dbg if debug else None)
                out.write(frame); continue

            try:
                lm = res.pose_landmarks.landmark

                # init legs on first frame with auto side
                if right_leg is None or left_leg is None:
                    side = auto_side(lm)
                    right_leg = KalmanLegTracker("right" if side=="right" else "left")
                    left_leg  = KalmanLegTracker("left"  if side=="right" else "right")

                # core joints
                shoulder = np.array([lm[PL.RIGHT_SHOULDER.value].x, lm[PL.RIGHT_SHOULDER.value].y], dtype=float)
                hip      = np.array([lm[PL.RIGHT_HIP.value].x,      lm[PL.RIGHT_HIP.value].y],      dtype=float)

                # head-like
                head = None
                for idx in (PL.RIGHT_EAR.value, PL.LEFT_EAR.value, PL.NOSE.value):
                    if lm[idx].visibility > 0.5:
                        head = np.array([lm[idx].x, lm[idx].y], dtype=float); break
                if head is None:
                    frame = draw_overlay(frame, reps=counter, feedback=None, progress_pct=prog, dbg_text=dbg if debug else None)
                    out.write(frame); continue

                # legs update (floor anchor)
                right_pts = right_leg.update(lm, dt, occlusion_boxes_norm=occl_boxes)
                left_pts  = left_leg.update(lm, dt, occlusion_boxes_norm=occl_boxes)
                floor_y = None
                for d in (right_leg, left_leg):
                    if d.floor_y_med is not None:
                        floor_y = d.floor_y_med if floor_y is None else max(floor_y, d.floor_y_med)

                # hinge metric (torסo)
                delta_x = abs(hip[0] - shoulder[0])
                if not top_ref.ready:
                    top_ref.update(delta_x)  # initialize
                else:
                    if delta_x < (top_ref.v * 1.4):
                        top_ref.update(delta_x)  # adapt slowly only when near upright

                hinge = delta_x
                # bar y proxy
                bar_y, bar_ok = bar.update(lm)

                # curvature only inside rep
                mid_spine = (shoulder + hip) * 0.5 * 0.4 + head * 0.6
                _, rounded = analyze_back_curvature(shoulder, hip, mid_spine)
                if state != STATE_TOP:
                    back_frames = back_frames + 1 if rounded else max(0, back_frames - 1)
                else:
                    back_frames = 0

                # === FSM ===
                if state == STATE_TOP:
                    prog = PROG_ALPHA*1.0 + (1-PROG_ALPHA)*prog  # glide to 100%
                    # wait for descent: hinge rises OR bar_y moves down sufficiently for few frames
                    start_cond = (hinge - top_ref.v) > HINGE_START_THRESH
                    if not start_cond and bar_ok and bar.top_ref() is not None:
                        start_cond = (bar_y - bar.top_ref()) > (ROM_MIN_BAR*0.5)
                    if start_cond and (frame_idx - last_rep_frame > MIN_FRAMES_BW_REPS):
                        state = STATE_DOWN
                        bottom_est = hinge
                        hinge_min = hinge_max = hinge
                        bar_min = bar_max = bar_y if bar_ok else None
                        down_frames = up_frames = top_hold_frames = 0
                        bar.reset_rep()
                elif state == STATE_DOWN:
                    bottom_est = max(bottom_est or hinge, hinge)
                    hinge_min = min(hinge_min, hinge) if hinge_min is not None else hinge
                    hinge_max = max(hinge_max, hinge) if hinge_max is not None else hinge
                    if bar_ok:
                        bar_min = min(bar_min, bar_y) if bar_min is not None else bar_y
                        bar_max = max(bar_max, bar_y) if bar_max is not None else bar_y

                    # progress for donut (1=top, 0=bottom)
                    denom = max(1e-4, (bottom_est - top_ref.v))
                    pr = 1.0 - ((hinge - top_ref.v) / denom)
                    pr = float(np.clip(pr, 0.0, 1.0))
                    prog = PROG_ALPHA*pr + (1-PROG_ALPHA)*prog
                    down_frames += 1

                    # reversal to UP: hinge stops increasing AND starts decreasing (or bar starts up)
                    rev_by_hinge = down_frames >= MIN_FRAMES_DOWN and (hinge_max - hinge) > 0.01
                    rev_by_bar   = bar_ok and down_frames >= MIN_FRAMES_DOWN and (bar_min is not None) and ((bar_y - bar_min) < -0.005)
                    if rev_by_hinge or rev_by_bar:
                        state = STATE_UP
                elif state == STATE_UP:
                    hinge_min = min(hinge_min, hinge) if hinge_min is not None else hinge
                    hinge_max = max(hinge_max, hinge) if hinge_max is not None else hinge
                    if bar_ok:
                        bar_min = min(bar_min, bar_y) if bar_min is not None else bar_y
                        bar_max = max(bar_max, bar_y) if bar_max is not None else bar_y

                    denom = max(1e-4, (bottom_est - top_ref.v))
                    pr = 1.0 - ((hinge - top_ref.v) / denom)
                    pr = float(np.clip(pr, 0.0, 1.0))
                    prog = PROG_ALPHA*pr + (1-PROG_ALPHA)*prog
                    up_frames += 1
                    if prog >= 0.90: top_hold_frames += 1

                    # end near upright
                    end_cond_hinge = (hinge - top_ref.v) < HINGE_END_THRESH
                    end_cond_bar   = (bar_ok and bar.top_ref() is not None and (abs(bar_y - bar.top_ref()) < (ROM_MIN_BAR*0.5)))
                    if (end_cond_hinge or end_cond_bar):
                        # check ROM and duration => reject unrelated/noisy motion
                        rom_hinge = (hinge_max - hinge_min) if (hinge_max is not None and hinge_min is not None) else 0.0
                        rom_bar   = (bar_max - bar_min) if (bar_max is not None and bar_min is not None) else 0.0
                        enough_rom = (rom_hinge >= ROM_MIN_HINGE) or (rom_bar >= ROM_MIN_BAR and bar_ok)
                        long_enough = (down_frames + up_frames) >= MIN_FRAMES_TOTAL

                        if enough_rom and long_enough and (frame_idx - last_rep_frame > MIN_FRAMES_BW_REPS):
                            # scoring (penalties)
                            penalty = 0.0
                            fb = []
                            if rounded and back_frames >= BACK_MIN_FRAMES:
                                fb.append("Try to keep your back a bit straighter"); penalty += 1.5
                            if not end_cond_hinge and not end_cond_bar:
                                fb.append("Try to finish more upright"); penalty += 1.0

                            score = round(max(4.0, 10.0 - penalty) * 2) / 2

                            counter += 1
                            (good_reps, bad_reps) = (good_reps+1, bad_reps) if score >= 9.5 else (good_reps, bad_reps+1)
                            all_scores.append(score)

                            down_s = (down_frames * dt)
                            top_s  = (top_hold_frames * dt)
                            # ROM chosen by stronger signal
                            rom_norm = None
                            if rom_bar and rom_bar >= ROM_MIN_BAR:
                                # normalize by (bar_max - top_ref_bar) if יש
                                top_b = bar.top_ref()
                                if top_b is not None:
                                    rom_norm = (bar_max - top_b)
                                else:
                                    rom_norm = rom_bar
                            else:
                                rom_norm = rom_hinge
                            tip_str = choose_deadlift_tip(down_s, top_s, rom_norm)

                            reps_report.append({
                                "rep_index": counter,
                                "score": float(score),
                                "score_display": display_half_str(score),
                                "feedback": fb,
                                "tip": tip_str
                            })
                            last_rep_frame = frame_idx

                        # reset to TOP
                        state = STATE_TOP
                        bottom_est = None
                        back_frames = 0
                        down_frames = up_frames = top_hold_frames = 0
                        hinge_min = hinge_max = None
                        bar_min = bar_max = None
                        bar.reset_rep()

                # live feedback (non-scoring) בתוך חזרה
                if state != STATE_TOP and rounded:
                    rt_fb = "Try to keep your back a bit straighter"

                # ציור שלד וחפיפות
                pts_draw = {PL.RIGHT_SHOULDER.value: tuple(shoulder),
                            PL.RIGHT_HIP.value: tuple(hip)}
                for d in (right_pts, left_pts):
                    for idx, pt in d.items():
                        if occl_boxes:
                            x,y = pt; skip=False
                            for (x1,y1,x2,y2) in occl_boxes:
                                if x1<=x<=x2 and y1<=y<=y2: skip=True; break
                            if skip: continue
                        pts_draw[idx] = pt

                # (אופציונלי) נקודה קטנה למיקום ה"בר"
                if debug and bar_ok and bar_y is not None:
                    by = int((bar_y)*h); bx = int(w*0.08)
                    cv2.circle(frame, (bx, by), 5, (60,220,255), -1, cv2.LINE_AA)
                    dbg = f"state={state} hinge={hinge:.3f} top={top_ref.v:.3f} bar_ok={int(bar_ok)}"

                frame = draw_body_only_from_dict(frame, pts_draw)
                frame = draw_overlay(frame, reps=counter, feedback=rt_fb, progress_pct=prog, dbg_text=dbg if debug else None)
                out.write(frame)

            except Exception:
                frame = draw_overlay(frame, reps=counter, feedback=None, progress_pct=prog)
                if out is not None: out.write(frame)
                continue

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)
    feedback_list = overall_feedback[:] if overall_feedback else ["Great form! Keep your spine neutral and hinge smoothly. 💪"]

    try:
        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(f"Total Reps: {counter}\n")
            f.write(f"Technique Score: {technique_score}/10\n")
            if feedback_list:
                f.write("Feedback:\n")
                for fb in feedback_list:
                    f.write(f"- {fb}\n")
    except Exception:
        pass

    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg","-y","-i", output_path,
            "-c:v","libx264","-preset","fast","-movflags","+faststart","-pix_fmt","yuv420p",
            encoded_path
        ], check=False)
        if os.path.exists(output_path) and os.path.exists(encoded_path):
            os.remove(output_path)
    except Exception:
        pass
    final_video_path = encoded_path if os.path.exists(encoded_path) else (output_path if os.path.exists(output_path) else "")

    return {
        "squat_count": counter,
        "technique_score": technique_score,
        "technique_score_display": display_half_str(technique_score),
        "technique_label": score_label(technique_score),
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "feedback": feedback_list,
        "reps": reps_report,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }

        "reps": reps_report,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }


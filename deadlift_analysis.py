# -*- coding: utf-8 -*-
# deadlift_analysis.py — Deadlift aligned to squat standard (fixed feedback aggregation)
# - Overlay/encoding/speed identical to squat
# - Bi-directional donut (0% bottom → 100% upright) with smoothing
# - Robust legs: Kalman + gating + floor anchor (+ optional YOLOv8n-ONNX occlusion detector)
# - Scoring feedback per-rep + one non-scoring tip per rep
# - TOP-LEVEL feedback now aggregates ONLY problems (score < 10). No generic “Analysis complete/Great form”.

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

# ===================== BACK CURVATURE =====================
def analyze_back_curvature(shoulder, hip, head_like, max_angle_deg=20.0, min_head_dist_ratio=0.35):
    torso_vec = shoulder - hip  # hip -> shoulder
    head_vec = head_like - shoulder
    torso_nrm = np.linalg.norm(torso_vec) + 1e-9
    head_nrm = np.linalg.norm(head_vec) + 1e-9
    if head_nrm < (min_head_dist_ratio * torso_nrm):
        return 0.0, False
    cosang = float(np.dot(torso_vec, head_vec) / (torso_nrm * head_nrm))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cosang))
    return angle_deg, angle_deg > max_angle_deg

# ===================== Deadlift thresholds =====================
HINGE_START_THRESH    = 0.08
STAND_DELTA_TARGET   = 0.025
END_THRESH           = 0.035
MIN_FRAMES_BETWEEN_REPS = 10
PROG_ALPHA           = 0.35  # smoothing for donut only

# Tips (non-scoring) thresholds (no bottom-pause tip)
TIP_MIN_ECC_S     = 0.35
TIP_MIN_TOP_S     = 0.15
TIP_SMALL_ROM_LO  = 0.035
TIP_SMALL_ROM_HI  = 0.055

def choose_deadlift_tip(down_s, top_s, rom):
    if down_s is not None and down_s < TIP_MIN_ECC_S:
        return "Slow the lowering to ~2–3s for more hypertrophy"
    if top_s is not None and top_s < TIP_MIN_TOP_S:
        return "Squeeze the glutes for a brief hold at the top"
    if rom is not None and TIP_SMALL_ROM_LO <= rom <= TIP_SMALL_ROM_HI:
        return "Hinge a bit deeper within comfort"
    return "Keep the bar close and move smoothly"

# ===================== Kalman Leg Tracker (robust to occlusion) =====================
LEG_VIS_THR    = 0.55
LEG_MAX_MAH    = 6.0
LEG_MISS_FLOOR = 6
LEG_SIDE_XSPAN = 0.25

class _KF:
    def __init__(self, q=1e-3, r=6e-3):
        self.x = np.zeros((4,1), dtype=float)  # [x, y, vx, vy]
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
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        maha2 = float(y.T @ np.linalg.inv(S) @ y)
        return maha2

class KalmanLegTracker:
    def __init__(self, side="right"):
        self.side = side
        PL = mp.solutions.pose.PoseLandmark
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
            kf=self.kf[idx]
            if not kf.inited:
                if lm[idx].visibility >= LEG_VIS_THR:
                    kf.x[:2,0] = [lm[idx].x, lm[idx].y]
                    kf.x[2:,0] = [0.0, 0.0]
                    kf.inited = True
                else:
                    continue
            kf.predict(dt)
            meas_ok=False
            if lm[idx].visibility >= LEG_VIS_THR:
                z=(lm[idx].x, lm[idx].y)
                if (not self._near_hands(lm, z)) and (abs(z[0]-hip_x) <= LEG_SIDE_XSPAN) and (not _inside_any_bbox(z)):
                    maha2 = kf.update(z)
                    if maha2 <= (LEG_MAX_MAH**2): meas_ok=True
            if meas_ok:
                self.miss[idx]=0
            else:
                self.miss[idx]+=1
                if self.miss[idx] >= LEG_MISS_FLOOR and self.floor_y_med is not None:
                    kf.x[1,0] = 0.8*kf.x[1,0] + 0.2*self.floor_y_med
            out[idx]=(float(kf.x[0,0]), float(kf.x[1,0]))
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
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2-xx1+1); h = np.maximum(0, yy2-yy1+1)
            inter = w*h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds+1]
        return keep
    def infer(self, frame_bgr):
        h0, w0 = frame_bgr.shape[:2]
        img, r, (dw, dh) = self._letterbox(frame_bgr, self.imgsz)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = np.transpose(img_rgb, (2,0,1))[None]
        out = self.sess.run([self.out_name], {self.inp_name: img_rgb})[0]
        if out.ndim == 3 and out.shape[0] == 1:
            o = out[0]
            if o.shape[0] in (84,85):
                num = o.shape[1]
                xywh = o[0:4, :].T
                if o.shape[0] >= 85:
                    obj = o[4, :].reshape(num,1)
                    cls = o[5:, :].T
                    cls_id = np.argmax(cls, axis=1)
                    conf = (obj * cls.max(axis=1)).flatten()
                else:
                    cls = o[4:, :].T
                    cls_id = np.argmax(cls, axis=1)
                    conf = cls.max(axis=1).flatten()
            else:
                xywh = o[:, 0:4]
                if o.shape[1] >= 85:
                    obj = o[:, 4:5]
                    cls = o[:, 5:]
                    cls_id = np.argmax(cls, axis=1)
                    conf = (obj[:,0] * cls.max(axis=1)).flatten()
                else:
                    cls = o[:, 4:]
                    cls_id = np.argmax(cls, axis=1)
                    conf = cls.max(axis=1).flatten()
        else:
            return []
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
        boxes_norm = []
        for x1,y1,x2,y2 in xyxy:
            boxes_norm.append((float(x1)/w0, float(y1)/h0, float(x2)/w0, float(y2)/h0))
        return boxes_norm

# ===================== Overlay =====================
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

def draw_overlay(frame, reps=0, feedback=None, progress_pct=0.0):
    h, w, _ = frame.shape
    # Reps box
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
    frame = np.array(pil)

    # Feedback (bottom)
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
                          detector_class_ids=None,
                          # NEW: API-compat flags
                          return_video=True,
                          fast_mode=None  # alias: if True -> no video
                          ):
    """
    API-compat:
      - return_video=False  => מסלול מהיר בלי החזרת וידאו (ניתוח בלבד)
      - return_video=True   => מסלול עם וידאו באיכות מקורית + אוברליי
      - fast_mode=True      => זהה ל-return_video=False (כינוי רך לא לשבור קריאות)
    """
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

    # Optional YOLO occluder detector (נריץ גם במסלול מהיר, אבל זה לא חובה; אפשר לכבות אם תרצה)
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
            detector = None  # fallback silently

    PL = mp.solutions.pose.PoseLandmark
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))
    dt = 1.0 / float(effective_fps)

    counter = good_reps = bad_reps = 0
    all_scores, reps_report = [], []

    rep = False
    last_rep_frame = -999
    frame_idx = 0

    # donut progress state
    top_ref = STAND_DELTA_TARGET
    bottom_est = None
    prog = 1.0

    # back curvature frames accumulation
    BACK_MIN_FRAMES = max(2, int(0.25 / dt))
    back_frames = 0

    # tempo counters per rep
    down_frames = up_frames = top_hold_frames = 0
    prev_progress = None

    # trackers
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

            # מסלול מהיר עובד על scale (כמו קודם); במסלול וידאו זה גם הפריים שנכתוב
            work = cv2.resize(frame, (0,0), fx=scale, fy=scale) if scale != 1.0 else frame

            # פותחים VideoWriter רק אם רוצים וידאו חזרה
            if return_video and out is None:
                h0, w0 = work.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w0, h0))

            occl_boxes = []
            if detector is not None:
                try:
                    occl_boxes = detector.infer(work)
                except Exception:
                    occl_boxes = []

            rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            rt_fb = None
            if not res.pose_landmarks:
                if return_video:
                    frame_drawn = draw_overlay(work.copy(), reps=counter, feedback=None, progress_pct=prog)
                    out.write(frame_drawn)
                continue

            try:
                lm = res.pose_landmarks.landmark
                shoulder = np.array([lm[PL.RIGHT_SHOULDER.value].x, lm[PL.RIGHT_SHOULDER.value].y], dtype=float)
                hip      = np.array([lm[PL.RIGHT_HIP.value].x,      lm[PL.RIGHT_HIP.value].y],      dtype=float)

                # head-like point (fallback: לא נעצור ניתוח אם אין)
                head = None
                for idx in (PL.RIGHT_EAR.value, PL.LEFT_EAR.value, PL.NOSE.value):
                    if lm[idx].visibility > 0.5:
                        head = np.array([lm[idx].x, lm[idx].y], dtype=float); break

                # legs trackers
                right_leg_pts = right_leg.update(lm, dt, occlusion_boxes_norm=occl_boxes)
                left_leg_pts  = left_leg.update(lm, dt, occlusion_boxes_norm=occl_boxes)

                # hinge proxy: shoulder-hip horizontal delta
                delta_x = abs(hip[0] - shoulder[0])
                if delta_x < (STAND_DELTA_TARGET * 1.4):
                    top_ref = 0.9*top_ref + 0.1*delta_x

                # back curvature (אם אין ראש — לא מפילים חזרה)
                if head is not None:
                    mid_spine = (shoulder + hip) * 0.5 * 0.4 + head * 0.6
                    _, rounded = analyze_back_curvature(shoulder, hip, mid_spine)
                    if delta_x < (HINGE_START_THRESH * 1.05):
                        rounded = False
                else:
                    rounded = False
                back_frames = back_frames + 1 if rounded else max(0, back_frames - 1)

                # start rep
                if (not rep) and (delta_x > HINGE_START_THRESH) and (frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS):
                    rep = True
                    bottom_est = delta_x
                    back_frames = 0
                    down_frames = up_frames = top_hold_frames = 0
                    prev_progress = prog

                # progress (bi-directional proximity to upright)
                if rep:
                    bottom_est = max(bottom_est or delta_x, delta_x)
                    denom = max(1e-4, (bottom_est - top_ref))
                    pr = 1.0 - ((delta_x - top_ref) / denom)     # 1=upright, 0=bottom
                    pr = float(np.clip(pr, 0.0, 1.0))
                    prog = PROG_ALPHA * pr + (1-PROG_ALPHA) * prog

                    if prev_progress is not None:
                        if prog < prev_progress - 1e-4:   down_frames += 1
                        elif prog > prev_progress + 1e-4: up_frames += 1
                    prev_progress = prog

                    if prog >= 0.90: top_hold_frames += 1
                    if rounded: rt_fb = "Try to keep your back a bit straighter"
                else:
                    prog = PROG_ALPHA*1.0 + (1-PROG_ALPHA)*prog

                # end rep near upright
                if rep and (delta_x < END_THRESH):
                    if frame_idx - last_rep_frame > MIN_FRAMES_BETWEEN_REPS:
                        penalty = 0.0
                        fb = []
                        if delta_x > (top_ref + 0.02):
                            fb.append("Try to finish more upright"); penalty += 1.0
                        if back_frames >= BACK_MIN_FRAMES:
                            fb.append("Try to keep your back a bit straighter"); penalty += 1.5

                        score = round(max(4, 10 - penalty) * 2) / 2
                        moved_enough = (bottom_est - delta_x) > 0.05
                        if moved_enough:
                            counter += 1
                            if score >= 9.5: good_reps += 1
                            else: bad_reps += 1
                            down_s = (down_frames * dt) if down_frames is not None else None
                            top_s  = (top_hold_frames * dt) if top_hold_frames is not None else None
                            rom    = (bottom_est - top_ref) if (bottom_est is not None and top_ref is not None) else None
                            tip_str = choose_deadlift_tip(down_s, top_s, rom)
                            # שמים feedback רק כשיש הורדת ציון
                            rep_fb = fb if score < 10.0 else []
                            reps_report.append({
                                "rep_index": counter,
                                "score": float(score),
                                "score_display": display_half_str(score),
                                "feedback": rep_fb,
                                "tip": tip_str
                            })
                            all_scores.append(score)
                        last_rep_frame = frame_idx

                    # reset
                    rep = False
                    bottom_est = None
                    back_frames = 0
                    prev_progress = None
                    down_frames = up_frames = top_hold_frames = 0

                # ציור רק אם return_video=True
                if return_video:
                    pts_draw = {
                        PL.RIGHT_SHOULDER.value: tuple(shoulder),
                        PL.RIGHT_HIP.value: tuple(hip),
                    }
                    for d in (right_leg_pts, left_leg_pts):
                        for idx, pt in d.items():
                            if occl_boxes:
                                x,y = pt
                                if any(x1<=x<=x2 and y1<=y<=y2 for (x1,y1,x2,y2) in occl_boxes):
                                    continue
                            pts_draw[idx] = pt
                    work_drawn = draw_body_only_from_dict(work.copy(), pts_draw)
                    work_drawn = draw_overlay(work_drawn, reps=counter, feedback=rt_fb, progress_pct=prog)
                    out.write(work_drawn)

            except Exception:
                if return_video and out is not None:
                    out.write(draw_overlay(work.copy(), reps=counter, feedback=None, progress_pct=prog))
                continue

    cap.release()
    if return_video and out: out.release()
    cv2.destroyAllWindows()

    avg = np.mean(all_scores) if all_scores else 0.0
    technique_score = round(round(avg * 2) / 2, 2)

    # Aggregate ONLY issues from reps (score<10), unique+ordered
    seen = set(); feedback_list = []
    for r in reps_report:
        if float(r.get("score") or 0.0) >= 10.0: continue
        for msg in (r.get("feedback") or []):
            if msg and msg not in seen:
                seen.add(msg); feedback_list.append(msg)

    final_video_path = ""
    if return_video:
        # encode H.264 faststart
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
        "feedback": feedback_list,   # רק בעיות; אם מושלם – []
        "reps": reps_report,
        "video_path": final_video_path,
        "feedback_path": feedback_path
    }


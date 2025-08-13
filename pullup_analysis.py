# -*- coding: utf-8 -*-
# pullup_analysis.py — Lean & Fast version
# - Counting: simple & robust (elbow closing + head rises -> UP; top -> REP; bottom -> reset)
# - Performance: model_complexity=0, pure-OpenCV overlay, minimal logs
# - Matches squat API/fields; global name: "squat_count"

import os, cv2, math, time, json, numpy as np, subprocess

# ===================== Config =====================
# Counting thresholds
ANGLE_DELTA_START = 4.0     # deg (elbow closing over last processed frame)
HEAD_DELTA_START  = 0.003   # head_y decreases by at least this (rise)
ELBOW_TOP_THRESHOLD    = 65.0
ELBOW_BOTTOM_THRESHOLD = 158.0
MIN_ASCENT_FROM_BASE   = 0.06
BOTTOM_HEAD_TOLERANCE  = 0.02  # how close head returns to base
REFRACTORY_FRAMES      = 3     # prevent double count after a rep
# Performance
FPS_FALLBACK = 25.0
MODEL_COMPLEXITY = 0        # 0 for speed
DETECT_CONF = 0.4
TRACK_CONF  = 0.4
# Overlay
SHOW_DONUT = False          # donut off for speed; set True to draw
BAR_BG_ALPHA = 0.55

# ===================== Helpers =====================
def _angle(a,b,c):
    try:
        ba=(a[0]-b[0],a[1]-b[1]); bc=(c[0]-b[0],c[1]-b[1])
        den=((ba[0]**2+ba[1]**2)**0.5*(bc[0]**2+bc[1]**2)**0.5) + 1e-9
        cosang=(ba[0]*bc[0]+ba[1]*bc[1])/den
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))
    except Exception:
        return 180.0

def _round_score_half(x): return round(x*2)/2.0
def display_half_str(x):
    q=round(float(x)*2)/2.0
    return str(int(round(q))) if abs(q-round(q))<1e-9 else f"{q:.1f}"

def score_label(s):
    s=float(s)
    if s>=9.5:return "Excellent"
    if s>=8.5:return "Very good"
    if s>=7.0:return "Good"
    if s>=5.5:return "Fair"
    return "Needs work"

# ===================== MediaPipe =====================
try:
    import mediapipe as mp
    MP_AVAILABLE=True
except Exception:
    MP_AVAILABLE=False

if MP_AVAILABLE:
    mp_pose=mp.solutions.pose
    _FACE_LMS={mp_pose.PoseLandmark.NOSE.value,
               mp_pose.PoseLandmark.LEFT_EYE_INNER.value, mp_pose.PoseLandmark.LEFT_EYE.value, mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
               mp_pose.PoseLandmark.RIGHT_EYE_INNER.value, mp_pose.PoseLandmark.RIGHT_EYE.value, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
               mp_pose.PoseLandmark.LEFT_EAR.value, mp_pose.PoseLandmark.RIGHT_EAR.value,
               mp_pose.PoseLandmark.MOUTH_LEFT.value, mp_pose.PoseLandmark.MOUTH_RIGHT.value}
    _BODY_CONNECTIONS=tuple((a,b) for (a,b) in mp_pose.POSE_CONNECTIONS if a not in _FACE_LMS and b not in _FACE_LMS)

POSE_IDXS={"nose":0,"left_shoulder":11,"right_shoulder":12,"left_elbow":13,"right_elbow":14,"left_wrist":15,"right_wrist":16}

def _pick_side(lms):
    rs=lms[POSE_IDXS["right_shoulder"]].visibility; ls=lms[POSE_IDXS["left_shoulder"]].visibility
    re=lms[POSE_IDXS["right_elbow"]].visibility; le=lms[POSE_IDXS["left_elbow"]].visibility
    return "right" if (rs>0.4)+(re>0.4) >= (ls>0.4)+(le>0.4) else "left"

def _get_xy(landmarks, idx, w, h):
    lm=landmarks[idx]; return (lm.x*w, lm.y*h, lm.visibility)

def _safe_vis(*vals, min_v=0.4): return all(v>=min_v for v in vals)

def draw_overlay_cv(frame, reps):
    # Minimal overlay like squat (left-top reps, with bg)
    (tw, th), _ = cv2.getTextSize(f"Reps: {int(reps)}", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    pad=8
    bg=frame.copy()
    cv2.rectangle(bg, (0,0), (tw+2*pad, th + int(1.8*pad)), (0,0,0), -1)
    frame = cv2.addWeighted(bg, BAR_BG_ALPHA, frame, 1.0 - BAR_BG_ALPHA, 0)
    cv2.putText(frame, f"Reps: {int(reps)}", (pad, th+pad), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return frame

# ===================== Analyzer =====================
class PullUpAnalyzer:
    def __init__(self): self.reset()
    def reset(self):
        self.rep_count=0
        self.state="idle"         # idle -> up -> wait_bottom
        self.side_locked=None
        self.head_base=None
        self.min_head_y=None
        self.last_head_y=None
        self.last_elbow=None
        self.refractory=0
        self.reps_meta=[]
        self.all_scores=[]

    def process(self, input_path, output_path=None, frame_skip=3, scale=0.4, overlay_enabled=True):
        if not MP_AVAILABLE: raise RuntimeError("mediapipe not available")
        try: cv2.setUseOptimized(True); cv2.setNumThreads(1)
        except Exception: pass

        cap=cv2.VideoCapture(input_path)
        if not cap.isOpened(): return self._empty_result("Could not open video", output_path)

        fps=cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        if fps<=1: fps=FPS_FALLBACK
        effective_fps=max(1.0, fps/max(1,frame_skip))

        ok, first=cap.read()
        if not ok:
            cap.release(); return self._empty_result("Empty video", output_path)
        h0,w0=first.shape[:2]
        out_w=int(w0*(scale if scale>0 else 1.0)); out_h=int(h0*(scale if scale>0 else 1.0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        out=None
        if output_path:
            fourcc=cv2.VideoWriter_fourcc(*'mp4v')
            out=cv2.VideoWriter(output_path,fourcc,effective_fps,(out_w,out_h))

        print(f"[PULLUP] open | fps={fps:.2f} | skip={frame_skip} | eff_fps={effective_fps:.2f}")
        with mp_pose.Pose(model_complexity=MODEL_COMPLEXITY,
                          min_detection_confidence=DETECT_CONF,
                          min_tracking_confidence=TRACK_CONF) as pose:
            frame_idx=0
            while True:
                ret,frame0=cap.read()
                if not ret: break
                frame_idx+=1
                if frame_idx % max(1,frame_skip) != 0: continue

                frame=cv2.resize(frame0,(out_w,out_h),interpolation=cv2.INTER_LINEAR)

                # Pose
                image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=pose.process(image)
                elbow_angle=None; head_y=None

                if results.pose_landmarks:
                    lms=results.pose_landmarks.landmark
                    if self.side_locked is None:
                        self.side_locked=_pick_side(lms)
                        print(f"[PULLUP] side locked = {self.side_locked}")
                    side=self.side_locked
                    sh_idx=POSE_IDXS["right_shoulder"] if side=="right" else POSE_IDXS["left_shoulder"]
                    el_idx=POSE_IDXS["right_elbow"]    if side=="right" else POSE_IDXS["left_elbow"]
                    wr_idx=POSE_IDXS["right_wrist"]    if side=="right" else POSE_IDXS["left_wrist"]

                    nose=_get_xy(lms,POSE_IDXS["nose"],out_w,out_h)
                    shld=_get_xy(lms,sh_idx,out_w,out_h)
                    elbw=_get_xy(lms,el_idx,out_w,out_h)
                    wrst=_get_xy(lms,wr_idx,out_w,out_h)

                    if _safe_vis(nose[2],shld[2],elbw[2],wrst[2],min_v=0.4):
                        head_y = nose[1]/out_h
                        elbow_angle = _angle(shld[:2], elbw[:2], wrst[:2])
                        if self.head_base is None: self.head_base = head_y

                # ===== Counting (simple) =====
                if elbow_angle is not None and head_y is not None:
                    d_el = 0.0 if self.last_elbow is None else (elbow_angle - self.last_elbow)
                    d_hd = 0.0 if self.last_head_y is None else (head_y - self.last_head_y)

                    if self.state == "idle":
                        # start of upward movement
                        if (d_el <= -ANGLE_DELTA_START) and (d_hd <= -HEAD_DELTA_START) and self.refractory==0:
                            self.state = "up"
                            self.min_head_y = head_y
                    elif self.state == "up":
                        if self.min_head_y is None or head_y < self.min_head_y:
                            self.min_head_y = head_y
                        ascent = (self.head_base - (self.min_head_y if self.min_head_y is not None else head_y)) if self.head_base is not None else 0.0
                        if (elbow_angle <= ELBOW_TOP_THRESHOLD) or (ascent >= MIN_ASCENT_FROM_BASE):
                            self.rep_count += 1
                            self.reps_meta.append({"rep_index": self.rep_count})
                            print(f"[PULLUP] REP {self.rep_count}")
                            self.state = "wait_bottom"
                            self.refractory = REFRACTORY_FRAMES
                    elif self.state == "wait_bottom":
                        # require clear bottom to reset
                        if (elbow_angle >= ELBOW_BOTTOM_THRESHOLD) and (self.head_base is not None and head_y >= self.head_base - BOTTOM_HEAD_TOLERANCE):
                            self.state = "idle"
                            self.min_head_y = None
                            self.head_base = head_y  # gentle drift

                # Overlay (fast)
                if overlay_enabled:
                    frame = draw_overlay_cv(frame, self.rep_count)
                if out is not None: out.write(frame)

                # Update deltas & refractory
                if elbow_angle is not None: self.last_elbow = elbow_angle
                if head_y is not None:     self.last_head_y = head_y
                if self.refractory>0: self.refractory -= 1

            # end while

        cap.release()
        if out is not None: out.release()

        # Faststart encode (post)
        final_video_path = ""
        if output_path:
            encoded=output_path.replace(".mp4","_encoded.mp4")
            try:
                subprocess.run(["ffmpeg","-y","-i",output_path,"-c:v","libx264","-preset","fast","-movflags","+faststart","-pix_fmt","yuv420p",encoded],
                               check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(output_path) and os.path.exists(encoded): os.remove(output_path)
                final_video_path = encoded if os.path.exists(encoded) else (output_path if os.path.exists(output_path) else "")
            except Exception:
                final_video_path = output_path if os.path.exists(output_path) else ""

        technique_score = float(_round_score_half(10.0 if self.rep_count>0 else 0.0))
        return {"squat_count": int(self.rep_count),
                "technique_score": technique_score,
                "technique_score_display": display_half_str(technique_score),
                "technique_label": score_label(technique_score),
                "good_reps": int(self.rep_count),
                "bad_reps": 0,
                "feedback": ["Nice pull!"] if self.rep_count>0 else ["No valid reps detected."],
                "tips": ["Keep core tight; minimize swing"],
                "reps": self.reps_meta,
                "video_path": final_video_path,
                "feedback_path": "pullup_feedback.txt"}

    def _empty_result(self, msg, output_path):
        return {"squat_count": 0,
                "technique_score": 0.0,
                "technique_score_display": display_half_str(0.0),
                "technique_label": score_label(0.0),
                "good_reps": 0, "bad_reps": 0,
                "feedback": [msg], "tips": [], "reps": [],
                "video_path": output_path or "", "feedback_path": "pullup_feedback.txt"}

# ===================== API =====================
def run_pullup_analysis(input_path, frame_skip=3, scale=0.4, output_path=None, overlay_enabled=True):
    try: cv2.setNumThreads(1)
    except Exception: pass
    t0=time.time()
    analyzer=PullUpAnalyzer()
    res=analyzer.process(input_path=input_path, output_path=output_path,
                         frame_skip=frame_skip, scale=scale, overlay_enabled=overlay_enabled)
    res["elapsed_sec"]=round(time.time()-t0,3)
    if output_path and "video_path" not in res: res["video_path"]=output_path
    print(f"[PULLUP] done | reps={res['squat_count']} | elapsed={res['elapsed_sec']}s")
    return res

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--input","-i",required=True)
    ap.add_argument("--output","-o",default="")
    ap.add_argument("--scale",type=float,default=0.4)
    ap.add_argument("--skip",type=int,default=3)
    ap.add_argument("--no-overlay",action="store_true")
    args=ap.parse_args()
    result=run_pullup_analysis(input_path=args.input, frame_skip=args.skip, scale=args.scale,
                               output_path=(args.output if args.output else None), overlay_enabled=(not args.no_overlay))
    print(json.dumps(result, ensure_ascii=False, indent=2))

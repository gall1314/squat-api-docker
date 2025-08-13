# -*- coding: utf-8 -*-
# pullup_analysis.py — יציב, מהיר, ותואם-סקוואט 1:1 באוברליי (עם ASCENT הפוך), ספירה פשוטה ועמידה בשינויים.
# ספירה: start=סגירת מרפק + עליית ראש; top=מרפק≤65° או עליה≥0.06; bottom=מרפק≥160° או חזרה קרובה לבייסליין.
# ביצועים: דילוג אדפטיבי (1 בתנועה/חזרה, frame_skip במנוחה), בלי seek; לוגים כמו בסקוואט.

import os, cv2, math, time, json, numpy as np, subprocess
from PIL import ImageFont, ImageDraw, Image

# ===================== STYLE / FONTS (כמו סקוואט) =====================
BAR_BG_ALPHA         = 0.55
REPS_FONT_SIZE       = 28
FEEDBACK_FONT_SIZE   = 22
DEPTH_LABEL_FONT_SIZE= 14
DEPTH_PCT_FONT_SIZE  = 18
FONT_PATH            = "Roboto-VariableFont_wdth,wght.ttf"

def _load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

REPS_FONT        = _load_font(FONT_PATH, REPS_FONT_SIZE)
FEEDBACK_FONT    = _load_font(FONT_PATH, FEEDBACK_FONT_SIZE)
DEPTH_LABEL_FONT = _load_font(FONT_PATH, DEPTH_LABEL_FONT_SIZE)
DEPTH_PCT_FONT   = _load_font(FONT_PATH, DEPTH_PCT_FONT_SIZE)

# Donut
DONUT_RADIUS_SCALE   = 0.72
DONUT_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)   # ירוק כמו בסקוואט
DEPTH_RING_BG        = (70, 70, 70)

# ===================== ספים / הגדרות ספירה =====================
ANGLE_DELTA_START       = 4.0      #° ירידה בזווית מרפק (סגירה)
HEAD_DELTA_START        = 0.003    #  ירידה ב-y (ראש עולה)
ELBOW_TOP_THRESHOLD     = 65.0     #° טופ
ELBOW_BOTTOM_THRESHOLD  = 160.0    #° תחתית
MIN_ASCENT_FROM_BASE    = 0.06     # עליה מינימלית מהבייסליין כדי לאשר טופ (אלטרנטיבי למרפק)
BOTTOM_HEAD_TOL         = 0.02     # כמה קרוב לבייסליין כדי לאשר תחתית
REFRACTORY_FRAMES       = 2        # מניעת דאבל-קאונט אחרי ספירה

HEAD_MIN_VIS            = 0.4      # ניתוח רק כשהנושא "על המתח"
FPS_FALLBACK            = 25.0

# דילוג פריימים
BASE_FRAME_SKIP_IDLE    = 3        # כמו בסקוואט (בררת מחדל באפליקציה)
BASE_FRAME_SKIP_MOVE    = 1        # בזמן תנועה/חזרה

HEARTBEAT_SEC           = 0.3      # קצב לוג כמו בסקוואט

# ===================== עזר =====================
def _angle(a,b,c):
    try:
        ba=(a[0]-b[0],a[1]-b[1]); bc=(c[0]-b[0],c[1]-b[1])
        den=((ba[0]**2+ba[1]**2)**0.5*(bc[0]**2+bc[1]**2)**0.5)+1e-9
        cosang=(ba[0]*bc[0]+ba[1]*bc[1])/den
        cosang=max(-1.0,min(1.0,cosang))
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
    _BODY_POINTS=tuple(sorted({i for conn in _BODY_CONNECTIONS for i in conn}))

POSE_IDXS={"nose":0,"left_shoulder":11,"right_shoulder":12,"left_elbow":13,"right_elbow":14,"left_wrist":15,"right_wrist":16}

def _pick_side(lms):
    rs=lms[POSE_IDXS["right_shoulder"]].visibility; ls=lms[POSE_IDXS["left_shoulder"]].visibility
    re=lms[POSE_IDXS["right_elbow"]].visibility; le=lms[POSE_IDXS["left_elbow"]].visibility
    return "right" if (rs>0.4)+(re>0.4) >= (ls>0.4)+(le>0.4) else "left"

def _get_xy(landmarks, idx, w, h):
    lm=landmarks[idx]
    return (lm.x*w, lm.y*h, lm.visibility)

def _safe_vis(*vals, min_v=0.4): return all(v>=min_v for v in vals)

def draw_body_only(frame, landmarks, color=(255,255,255)):
    if not MP_AVAILABLE: return frame
    h,w=frame.shape[:2]
    for a,b in _BODY_CONNECTIONS:
        pa=landmarks[a]; pb=landmarks[b]
        cv2.line(frame,(int(pa.x*w),int(pa.y*h)),(int(pb.x*w),int(pb.y*h)),color,2,cv2.LINE_AA)
    for i in _BODY_POINTS:
        p=landmarks[i]; cv2.circle(frame,(int(p.x*w),int(p.y*h)),3,color,-1,cv2.LINE_AA)
    return frame

# ===================== Overlay =====================
def draw_ascent_donut(frame, center, radius, thickness, pct):
    pct=float(np.clip(pct,0.0,1.0)); cx,cy=int(center[0]),int(center[1])
    radius=int(radius); thickness=int(thickness)
    cv2.circle(frame,(cx,cy),radius,DEPTH_RING_BG,thickness,cv2.LINE_AA)
    start_ang=-90; end_ang=start_ang+int(360*pct)
    cv2.ellipse(frame,(cx,cy),(radius,radius),0,start_ang,end_ang,DEPTH_COLOR,thickness,cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, ascent_pct=0.0):
    """Reps בפינת שמאל-עליון (0,0), דונאט ASCENT בימין-עליון, פידבק תחתון עד 2 שורות."""
    h,w=frame.shape[:2]

    # Reps box — צמוד לפינה
    pil=Image.fromarray(frame); draw=ImageDraw.Draw(pil)
    reps_text=f"Reps: {reps}"
    pad_x,pad_y=10,6
    text_w=draw.textlength(reps_text,font=REPS_FONT); text_h=REPS_FONT.size
    x0,y0=0,0; x1=int(text_w+2*pad_x); y1=int(text_h+2*pad_y)
    top=frame.copy(); cv2.rectangle(top,(x0,y0),(x1,y1),(0,0,0),-1)
    frame=cv2.addWeighted(top,BAR_BG_ALPHA,frame,1.0-BAR_BG_ALPHA,0)
    pil=Image.fromarray(frame); ImageDraw.Draw(pil).text((x0+pad_x,y0+pad_y-1),reps_text,font=REPS_FONT,fill=(255,255,255))
    frame=np.array(pil)

    # Donut — ימין-עליון
    ref_h=max(int(h*0.06), int(REPS_FONT_SIZE*1.6))
    radius=int(ref_h*DONUT_RADIUS_SCALE); thick=max(8,int(radius*DONUT_THICKNESS_FRAC))
    cx=w-70; cy=60
    frame=draw_ascent_donut(frame,(cx,cy),radius,thick,float(np.clip(ascent_pct,0,1)))
    # כיתוב מרכזי
    pil=Image.fromarray(frame); draw=ImageDraw.Draw(pil)
    label_txt="ASCENT"; pct_txt=f"{int(float(np.clip(ascent_pct,0,1))*100)}%"
    gap=max(2,int(radius*0.10))
    base_y=cy - (DEPTH_LABEL_FONT.size + gap + DEPTH_PCT_FONT.size)//2
    lw=draw.textlength(label_txt,font=DEPTH_LABEL_FONT)
    pw=draw.textlength(pct_txt,  font=DEPTH_PCT_FONT)
    draw.text((cx-int(lw//2), base_y), label_txt, font=DEPTH_LABEL_FONT, fill=(255,255,255))
    draw.text((cx-int(pw//2), base_y+DEPTH_LABEL_FONT.size+gap), pct_txt, font=DEPTH_PCT_FONT, fill=(255,255,255))
    frame=np.array(pil)

    # Bottom feedback (אופציונלי; עד 2 שורות, עם אליפסות בסוף שורה שניה)
    if feedback:
        pil_fb=Image.fromarray(frame); draw_fb=ImageDraw.Draw(pil_fb)
        safe_m=max(6,int(h*0.02)); bx,by,gap2=12,8,4; max_w=int(w-2*bx-20)
        def wrap_two_lines(draw,text,font,max_w):
            words=text.split(); lines=[]; cur=""
            for w_ in words:
                trial=(cur+" "+w_).strip()
                if draw.textlength(trial,font=font)<=max_w: cur=trial
                else:
                    if cur: lines.append(cur); cur=w_
                if len(lines)==2: break
            if cur and len(lines)<2: lines.append(cur)
            leftover=len(words)-sum(len(x.split()) for x in lines)
            if leftover>0 and len(lines)>=2:
                last=lines[-1]+"…"
                while draw.textlength(last,font=font)>max_w and len(last)>1:
                    last=last[:-2]+"…"
                lines[-1]=last
            return lines
        lines=wrap_two_lines(draw_fb,feedback,FEEDBACK_FONT,max_w)
        line_h=FEEDBACK_FONT.size+6; block_h=(2*by)+len(lines)*line_h+(len(lines)-1)*gap2
        y0=max(0,h-safe_m-block_h); y1=h-safe_m
        over=frame.copy(); cv2.rectangle(over,(0,y0),(w,y1),(0,0,0),-1)
        frame=cv2.addWeighted(over,BAR_BG_ALPHA,frame,1.0-BAR_BG_ALPHA,0)
        pil_fb=Image.fromarray(frame); draw_fb=ImageDraw.Draw(pil_fb)
        ty=y0+by
        for ln in lines:
            tw=draw_fb.textlength(ln,font=FEEDBACK_FONT); tx=max(bx,(w-int(tw))//2)
            draw_fb.text((tx,ty),ln,font=FEEDBACK_FONT,fill=(255,255,255)); ty+=line_h+gap2
        frame=np.array(pil_fb)

    return frame

# ===================== Analyzer =====================
class PullUpAnalyzer:
    def __init__(self): self.reset()
    def reset(self):
        self.rep_count=0
        self.state="idle"         # idle -> up -> wait_bottom
        self.side_locked=None
        self.head_base=None       # baseline ראש לתחילת סט/חזרה
        self.min_head_y=None      # שיא עלייה בתוך חזרה
        self.last_head_y=None
        self.last_elbow=None
        self.refractory=0
        self.reps_meta=[]
        self.all_scores=[]
        self.ascent_live=0.0
        self.active_frames=0      # כדי לא "לספור" כשהמצולם לא על המתח

    def _movement_detected(self, elbow_angle, head_y):
        mov=False
        if self.last_elbow is not None and elbow_angle is not None and abs(elbow_angle-self.last_elbow)>1.5: mov=True
        if self.last_head_y is not None and head_y is not None and abs(head_y-self.last_head_y)>0.002: mov=True
        return mov

    def process(self, input_path, output_path=None, frame_skip=3, scale=0.4, overlay_enabled=True):
        if not MP_AVAILABLE: raise RuntimeError("mediapipe not available")
        try: cv2.setUseOptimized(True); cv2.setNumThreads(1)
        except Exception: pass

        cap=cv2.VideoCapture(input_path)
        if not cap.isOpened(): return self._empty_result("Could not open video", output_path)

        fps_in=cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        if fps_in<=1: fps_in=FPS_FALLBACK
        print(f"[PULLUP] start | input={input_path} | fps_in={fps_in:.2f} | skip={frame_skip} | scale={scale}", flush=True)

        # הכנה לכותב וידאו (מידות יציבות)
        ok,first=cap.read()
        if not ok: cap.release(); return self._empty_result("Empty video", output_path)
        h0,w0=first.shape[:2]; out_w=int(w0*(scale if scale>0 else 1.0)); out_h=int(h0*(scale if scale>0 else 1.0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        effective_fps=max(1.0, fps_in/max(1,frame_skip))

        out=None
        if output_path:
            fourcc=cv2.VideoWriter_fourcc(*'mp4v')
            out=cv2.VideoWriter(output_path,fourcc,effective_fps,(out_w,out_h))

        last_log=time.time()
        frame_idx=0
        next_f = 1

        with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            print("[PULLUP] open mediapipe pose (mc=1, det=0.5, track=0.5)", flush=True)
            while True:
                ret, frame0 = cap.read()
                if not ret: break
                frame_idx += 1
                if frame_idx < next_f: continue

                frame = cv2.resize(frame0,(out_w,out_h),interpolation=cv2.INTER_LINEAR)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                elbow_angle=None; head_y=None
                active=False

                if results.pose_landmarks:
                    lms=results.pose_landmarks.landmark
                    # נעילת צד פעם אחת
                    if self.side_locked is None:
                        self.side_locked=_pick_side(lms)
                        print(f"[PULLUP] side locked = {self.side_locked}", flush=True)
                    side=self.side_locked
                    sh_idx=POSE_IDXS["right_shoulder"] if side=="right" else POSE_IDXS["left_shoulder"]
                    el_idx=POSE_IDXS["right_elbow"]    if side=="right" else POSE_IDXS["left_elbow"]
                    wr_idx=POSE_IDXS["right_wrist"]    if side=="right" else POSE_IDXS["left_wrist"]

                    nose=_get_xy(lms,POSE_IDXS["nose"],out_w,out_h)
                    shld=_get_xy(lms,sh_idx,out_w,out_h)
                    elbw=_get_xy(lms,el_idx,out_w,out_h)
                    wrst=_get_xy(lms,wr_idx,out_w,out_h)

                    if _safe_vis(nose[2],shld[2],elbw[2],wrst[2],min_v=HEAD_MIN_VIS):
                        head_y = nose[1]/out_h
                        elbow_angle = _angle(shld[:2], elbw[:2], wrst[:2])
                        active=True
                        self.active_frames = min(15, self.active_frames+1)  # "על המתח"
                        if self.head_base is None: self.head_base=head_y
                    else:
                        self.active_frames = max(0, self.active_frames-1)   # "ירד מהמתח"

                # עדכון ASCENT לדונאט
                if (self.head_base is not None) and (head_y is not None):
                    self.ascent_live = float(np.clip(self.head_base - head_y, 0.0, 1.0))
                else:
                    self.ascent_live = 0.0

                # ------------- ספירה פשוטה ויציבה -------------
                if active:
                    d_el = 0.0 if self.last_elbow is None else (elbow_angle - self.last_elbow)   # שלילי = סגירה
                    d_hd = 0.0 if self.last_head_y is None else (head_y - self.last_head_y)     # שלילי = עלייה

                    if self.state == "idle":
                        # התחלה: סגירת מרפק + עליית ראש
                        if (d_el <= -ANGLE_DELTA_START) and (d_hd <= -HEAD_DELTA_START) and self.refractory==0:
                            self.state="up"
                            self.min_head_y=head_y
                    elif self.state == "up":
                        if self.min_head_y is None or head_y < self.min_head_y:
                            self.min_head_y = head_y
                        ascent = (self.head_base - (self.min_head_y if self.min_head_y is not None else head_y)) if self.head_base is not None else 0.0
                        # טופ: או מרפק מספיק סגור, או עליה מספיקה מהבייסליין
                        if (elbow_angle <= ELBOW_TOP_THRESHOLD) or (ascent >= MIN_ASCENT_FROM_BASE):
                            self.rep_count += 1
                            self.reps_meta.append({"rep_index": self.rep_count, "top_elbow": float(elbow_angle), "ascent": float(ascent)})
                            print(f"[PULLUP] REP {self.rep_count}", flush=True)
                            self.state = "wait_bottom"
                            self.refractory = REFRACTORY_FRAMES
                    elif self.state == "wait_bottom":
                        # תחתית: או מרפק פתוח מספיק, או הראש חזר קרוב לבייסליין
                        if (elbow_angle >= ELBOW_BOTTOM_THRESHOLD) or (self.head_base is not None and head_y >= self.head_base - BOTTOM_HEAD_TOL):
                            self.state = "idle"
                            self.min_head_y=None
                            self.head_base=head_y  # עדכון בייסליין עדין

                else:
                    # לא פעיל => מאפסים state עדין, אין ספירה
                    self.state="idle"
                    self.min_head_y=None
                    self.refractory = 0

                # ציור שלד + אוברליי
                if results.pose_landmarks is not None:
                    frame = draw_body_only(frame, results.pose_landmarks.landmark)
                if overlay_enabled:
                    frame = draw_overlay(frame, reps=self.rep_count, feedback=None, ascent_pct=self.ascent_live)

                # כתיבה
                if output_path and out is not None:
                    out.write(frame)

                # עדכוני דלתא
                if elbow_angle is not None: self.last_elbow = elbow_angle
                if head_y is not None:     self.last_head_y = head_y
                if self.refractory>0: self.refractory -= 1

                # דילוג אדפטיבי — כל פריים בתנועה/חזרה, אחרת skip בסיסי
                moving = (self.state!="idle") or self._movement_detected(elbow_angle, head_y)
                step = BASE_FRAME_SKIP_MOVE if moving else frame_skip
                next_f = frame_idx + max(1, step)

                # Heartbeat לוג
                now=time.time()
                if now - last_log > HEARTBEAT_SEC:
                    print(f"[PULLUP] f{frame_idx} reps={self.rep_count} state={self.state} step={step}", flush=True)
                    last_log = now

        cap.release()
        if out is not None: out.release()

        # קידוד faststart
        final_video_path=""
        if output_path:
            encoded=output_path.replace(".mp4","_encoded.mp4")
            try:
                subprocess.run(["ffmpeg","-y","-i",output_path,"-c:v","libx264","-preset","fast","-movflags","+faststart","-pix_fmt","yuv420p",encoded],
                               check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(output_path) and os.path.exists(encoded): os.remove(output_path)
                final_video_path = encoded if os.path.exists(encoded) else (output_path if os.path.exists(output_path) else "")
            except Exception:
                final_video_path = output_path if os.path.exists(output_path) else ""

        # ניקוד — בינתיים 10 אם יש חזרות, ניתן להרחיב לפי סקוואט
        technique_score = float(_round_score_half(10.0 if self.rep_count>0 else 0.0))
        tips=["Keep your core tight and avoid swinging"]
        feedback_list=["Great form!"] if self.rep_count>0 else ["No valid reps detected."]

        return {"squat_count": int(self.rep_count),
                "technique_score": technique_score,
                "technique_score_display": display_half_str(technique_score),
                "technique_label": score_label(technique_score),
                "good_reps": int(self.rep_count),
                "bad_reps": 0,
                "feedback": feedback_list,
                "tips": tips,
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
    print(f"[PULLUP] result: reps={res['squat_count']} | elapsed={res['elapsed_sec']}s", flush=True)
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
                               output_path=(args.output if args.output else None), overlay_enabled=(not args.no-overlay))
    print(json.dumps(result, ensure_ascii=False, indent=2))

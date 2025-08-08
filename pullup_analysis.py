# pullup_analysis.py
import os, time, subprocess
import cv2
import numpy as np
import mediapipe as mp

__all__ = ["run_pullup_analysis", "render_pullup_video", "PullUpAnalyzer"]

mp_pose = mp.solutions.pose

# =========================
# Pull-up Analyzer (UNCHANGED LOGIC)
# =========================

class PullUpAnalyzer:
    """
    מזהה ריפים במתח, בודק טכניקה, ומחשב ROM-Up% לכל פריים.
    מחזיר ציון טכניקה מעוגל ל-.0/.5 עם מינימום 4.0, וסט פידבקים/טיפים באנגלית.
    """

    def __init__(
        self,
        angle_drop_threshold: float = 18.0,  # כמו הגרסה שעבדה
        min_separation: int = 5,
        nose_rise_thresh: float = 0.001
    ):
        self.angle_drop_threshold = angle_drop_threshold
        self.min_separation = min_separation
        self.nose_rise_thresh = nose_rise_thresh

    # ---------- Public: analyze all ----------
    def analyze_all_reps(self, frames):
        rep_ranges = self.segment_reps(frames)

        rep_reports = []
        all_errors, all_tips = [], []

        # רצף ROM לכל הווידאו (לטובת הרינדור)
        full_rom = [0.0] * len(frames)

        for (start, end) in rep_ranges:
            rep_frames = frames[start:end]
            result = self.analyze_rep(rep_frames)
            rep_reports.append(result)
            all_errors.extend(result["errors"])
            all_tips.extend(result["tips"])

            # פיזור ה‑ROM חזרה למיקומים המלאים
            for i, v in enumerate(result.get("rom_series", []), start=start):
                if 0 <= i < len(full_rom):
                    full_rom[i] = max(full_rom[i], v)

        if rep_reports:
            raw_scores = [r["technique_score"] for r in rep_reports]
            avg = float(np.mean(raw_scores))
            rounded_score = float(np.clip(round(avg * 2) / 2, 4.0, 10.0))
        else:
            rounded_score = 0.0

        feedback = list(set(all_errors)) if all_errors else (["Great form! Keep it up"] if rep_reports else ["No pull-ups detected"])
        tips = list(set(all_tips))

        return {
            "rep_count": len(rep_reports),
            "squat_count": len(rep_reports),  # legacy
            "technique_score": rounded_score,
            "good_reps": sum(1 for r in rep_reports if r["technique_score"] >= 8),
            "bad_reps": sum(1 for r in rep_reports if r["technique_score"] < 8),
            "feedback": feedback,
            "tips": tips,
            "reps": rep_reports,
            "rom_full": full_rom,
            "rep_ranges": rep_ranges,
        }

    # ---------- Per-rep analysis ----------
    def analyze_rep(self, frames):
        errors, tips = [], []
        elbow_angles, knee_angles = [], []
        nose_ys, wrist_ys = [], []

        for f in frames:
            l = self.elbow_angle(f, "LEFT")
            r = self.elbow_angle(f, "RIGHT")
            valid_elbows = [a for a in (l, r) if a is not None]
            if valid_elbows:
                elbow_angles.append(float(np.mean(valid_elbows)))

            kl = self.knee_angle(f, "LEFT")
            kr = self.knee_angle(f, "RIGHT")
            valid_knees = [a for a in (kl, kr) if a is not None]
            if valid_knees:
                knee_angles.append(float(np.mean(valid_knees)))

            if "NOSE" in f:
                nose_ys.append(f["NOSE"][1])
            if "LEFT_WRIST" in f and "RIGHT_WRIST" in f:
                wrist_y = (f["LEFT_WRIST"][1] + f["RIGHT_WRIST"][1]) / 2
                wrist_ys.append(wrist_y)

        # בדיקות טכניקה (כמו במקור)
        if not any(n < w - 0.005 for n, w in zip(nose_ys, wrist_ys)):
            errors.append("Try to pull a bit higher – chin past the bar")

        if max(elbow_angles, default=0) < 155:
            errors.append("Start each rep from straight arms for full range")

        if knee_angles and (max(knee_angles) - min(knee_angles)) > 40:
            errors.append("Keep legs steadier for more control")

        if len(elbow_angles) >= 6:
            descent_len = self.detect_descent_duration(elbow_angles)
            if descent_len < 2:
                tips.append("Control the lowering phase for better muscle growth")

        # ROM‑Up% (עלייה למעלה) + החלקה קלה
        rom_series = self.compute_rom_series(nose_ys, wrist_ys)
        rom_series = self.ema_smooth(rom_series, alpha=0.25)

        score_map = {0: 10, 1: 8, 2: 6, 3: 5}
        technique_score = score_map.get(len(errors), 4)

        return {
            "technique_score": technique_score,
            "errors": errors,
            "tips": tips,
            "rom_series": rom_series,
        }

    # ---------- Helpers ----------
    def detect_descent_duration(self, angles):
        if not angles:
            return 0
        min_idx = int(np.argmin(angles))
        for i in range(min_idx + 1, len(angles)):
            if angles[i] > angles[min_idx] + 5:
                return i - min_idx
        return len(angles) - min_idx

    def compute_rom_series(self, nose_ys, wrist_ys):
        if not nose_ys:
            return []
        if not wrist_ys:
            return [0.0] * len(nose_ys)
        bar_y = min(wrist_ys)       # גובה המוט (בערך)
        bottom_y = max(nose_ys)     # הכי נמוך של הסנטר
        denom = max(bottom_y - bar_y, 1e-6)
        out = []
        for ny in nose_ys:
            if ny < bar_y:
                rom = 1.0
            else:
                raw = (bottom_y - ny) / denom
                rom = float(np.clip(raw, 0.0, 1.0))
            out.append(rom)
        return out

    @staticmethod
    def ema_smooth(series, alpha=0.25):
        if not series:
            return []
        out, prev = [], 0.0
        for v in series:
            sm = alpha * v + (1 - alpha) * prev
            out.append(sm)
            prev = sm
        return out

    @staticmethod
    def elbow_angle(f, side):
        keys = [f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"]
        if all(k in f for k in keys):
            a, b, c = np.array(f[keys[0]]), np.array(f[keys[1]]), np.array(f[keys[2]])
            ba, bc = a - b, c - b
            denom = np.linalg.norm(ba) * np.linalg.norm(bc)
            if denom == 0:
                return None
            cosv = np.dot(ba, bc) / denom
            return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))
        return None

    @staticmethod
    def knee_angle(f, side):
        keys = [f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE"]
        if all(k in f for k in keys):
            a, b, c = np.array(f[keys[0]]), np.array(f[keys[1]]), np.array(f[keys[2]])
            ba, bc = a - b, c - b
            denom = np.linalg.norm(ba) * np.linalg.norm(bc)
            if denom == 0:
                return None
            cosv = np.dot(ba, bc) / denom
            return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))
        return None

    # ---------- Rep segmentation ----------
    def segment_reps(self, frames):
        elbow_angles, noses_y = [], []
        for f in frames:
            l = self.elbow_angle(f, "LEFT")
            r = self.elbow_angle(f, "RIGHT")
            valid = [a for a in (l, r) if a is not None]
            elbow_angles.append(float(np.mean(valid)) if valid else None)
            noses_y.append(f["NOSE"][1] if "NOSE" in f else None)

        reps = []
        last_rep_index = -self.min_separation
        state = "waiting"
        start_idx, start_angle = None, None

        for i in range(1, len(elbow_angles)):
            curr_angle = elbow_angles[i]
            prev_angle = elbow_angles[i - 1]
            curr_nose = noses_y[i]
            prev_nose = noses_y[i - 1]
            if curr_angle is None or prev_angle is None or curr_nose is None or prev_nose is None:
                continue

            angle_drop = prev_angle - curr_angle
            nose_rising = curr_nose < prev_nose - self.nose_rise_thresh

            if (
                state == "waiting"
                and angle_drop >= self.angle_drop_threshold
                and nose_rising
                and (i - last_rep_index) >= self.min_separation
            ):
                state = "in_rep"
                start_idx = i - 1
                start_angle = curr_angle

            elif state == "in_rep" and curr_angle > start_angle:
                reps.append((start_idx, i))
                last_rep_index = i
                state = "waiting"

        return reps


# =================
# Overlay helpers (UI only)
# =================

_DRAW_LMS = [
    "LEFT_SHOULDER","RIGHT_SHOULDER",
    "LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST",
    "LEFT_HIP","RIGHT_HIP",
    "LEFT_KNEE","RIGHT_KNEE",
    "LEFT_ANKLE","RIGHT_ANKLE",
]
_SKELETON_EDGES = [
    ("LEFT_SHOULDER","LEFT_ELBOW"), ("LEFT_ELBOW","LEFT_WRIST"),
    ("RIGHT_SHOULDER","RIGHT_ELBOW"), ("RIGHT_ELBOW","RIGHT_WRIST"),
    ("LEFT_SHOULDER","RIGHT_SHOULDER"),
    ("LEFT_HIP","RIGHT_HIP"),
    ("LEFT_SHOULDER","LEFT_HIP"), ("RIGHT_SHOULDER","RIGHT_HIP"),
    ("LEFT_HIP","LEFT_KNEE"), ("LEFT_KNEE","LEFT_ANKLE"),
    ("RIGHT_HIP","RIGHT_KNEE"), ("RIGHT_KNEE","RIGHT_ANKLE"),
]

def draw_skeleton(frame, lms):
    h, w = frame.shape[:2]
    def pt(name):
        if name not in lms: return None
        x, y, _ = lms[name]
        return int(x*w), int(y*h)
    for a, b in _SKELETON_EDGES:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, (255,255,255), 2)
    for name in _DRAW_LMS:
        p = pt(name)
        if p:
            cv2.circle(frame, p, 3, (255,255,255), -1)

def draw_reps_counter(frame, reps_so_far: int):
    # "Reps: N" שמאל למעלה (דינמי, מתעדכן תוך כדי)
    cv2.putText(frame, f"Reps: {reps_so_far}", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

def draw_rom_donut(frame, rom, center=(110,110), radius=60):
    # דונאט שמאל־למעלה כמו בבולגרי, בלי שום ציון נוסף
    pct = int(round(float(rom)*100))
    pct = max(0, min(100, pct))
    cv2.ellipse(frame, center, (radius, radius), 0, 0, 360, (50,50,50), 10, cv2.LINE_AA)
    ang = 360 * (pct / 100.0)
    cv2.ellipse(frame, center, (radius, radius), 0, -90, -90 + ang, (80,220,120), 10, cv2.LINE_AA)
    cv2.putText(frame, f"{pct}%", (center[0]-28, center[1]+10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "ROM-Up", (center[0]-40, center[1]-radius-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)

def draw_feedback_banner(frame, text):
    if not text: return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-60), (w, h), (0,0,0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    cv2.putText(frame, text, (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

def _safe_output_path(prefix="pullup"):
    return f"/tmp/{prefix}_{int(time.time())}.mp4"


# =================
# Video rendering (dynamic reps, donut top-left, no score)
# =================
def render_pullup_video(input_path, output_path, analysis_result,
                        frame_skip=3, scale=0.30, draw_skeleton_flag=True):
    cap = cv2.VideoCapture(input_path)

    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / max(frame_skip, 1)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w0, h0))

    rom_full     = analysis_result.get("rom_full", [])
    frames_lms   = analysis_result.get("frames_landmarks", [])
    rep_ranges   = analysis_result.get("rep_ranges", [])  # [(start,end), ...]
    feedback_txt = (analysis_result.get("feedback") or [""])[0]

    # למונה דינמי: נתקדם לפי סוף הריפ הבא
    next_rep_idx = 0
    reps_so_far  = 0

    idx = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % max(frame_skip, 1) != 0:
            frame_id += 1
            continue

        frame = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_AREA)

        # שלד — משתמשים בלנדמרקס שחושבו בשלב הניתוח (אין ריצת Pose נוספת)
        if draw_skeleton_flag and idx < len(frames_lms):
            lms = frames_lms[idx]
            if lms:
                draw_skeleton(frame, lms)

        # עדכון דינמי של מונה החזרות לפי פריים נוכחי
        while next_rep_idx < len(rep_ranges) and rep_ranges[next_rep_idx][1] <= idx:
            reps_so_far += 1
            next_rep_idx += 1
        draw_reps_counter(frame, reps_so_far)

        # דונאט ROM‑Up (שמאל‑למעלה)
        rom = rom_full[idx] if idx < len(rom_full) else 0.0
        draw_rom_donut(frame, rom)

        # באנר פידבק בתחתית
        draw_feedback_banner(frame, feedback_txt)

        out.write(frame)
        idx += 1
        frame_id += 1

    cap.release()
    out.release()

    # בדיקה קצרה (לא חובה)
    try:
        size = os.path.getsize(output_path)
        print(f"[pullup] wrote RAW: {output_path} | frames={idx} | size={size} bytes")
    except Exception as e:
        print("[pullup] raw render check failed:", e)


# =================
# Orchestrator (UNCHANGED)
# =================
def run_pullup_analysis(
    video_path,
    output_path,            # ← חובה, כמו בסקוואט/בולגרי
    frame_skip=3,           # כמו שעבד
    scale=0.30,
    model_complexity=1,
    verbose=False,
):
    """
    מריץ ניתוח + רינדור, ואז מקודד ל-H.264 (ffmpeg) ומחזיר video_path של המקודד.
    """
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=model_complexity)

    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % max(frame_skip, 1) != 0:
            frame_count += 1
            continue

        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            frame_landmarks = {}
            for i, lm in enumerate(res.pose_landmarks.landmark):
                frame_landmarks[mp_pose.PoseLandmark(i).name] = (lm.x, lm.y, lm.z)
            landmarks_list.append(frame_landmarks)
        else:
            landmarks_list.append({})

        frame_count += 1
        if verbose and frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    pose.close()

    analyzer = PullUpAnalyzer()
    analysis = analyzer.analyze_all_reps(landmarks_list)
    analysis["frames_landmarks"] = landmarks_list  # שימוש ברינדור — אין ריצת Pose שנייה

    # ודא שיש נתיב קובץ
    if not output_path:
        output_path = _safe_output_path("pullup")

    # שלב 1: רינדור MP4 גולמי (mp4v)
    render_pullup_video(
        input_path=video_path,
        output_path=output_path,
        analysis_result=analysis,
        frame_skip=frame_skip,
        scale=scale,
        draw_skeleton_flag=True,
    )

    # שלב 2: קידוד H.264 (libx264 + faststart + yuv420p)
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            encoded_path
        ], check=False)
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"[pullup] encoded: {encoded_path}")
    except Exception as e:
        print("[pullup] ffmpeg encode failed:", e)
        encoded_path = output_path

    analysis["video_path"] = encoded_path
    return analysis


# =================
# Example (CLI)
# =================
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        in_path, out_path = sys.argv[1], sys.argv[2]
        res = run_pullup_analysis(in_path, out_path, frame_skip=3, scale=0.30, model_complexity=1, verbose=True)
        print("Done:", {k: v for k, v in res.items() if k not in ("rom_full","reps","rep_ranges","frames_landmarks")})
    else:
        print("Usage: python pullup_analysis.py <input.mp4> <output.mp4>")

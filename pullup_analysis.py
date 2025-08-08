import cv2
import numpy as np
import mediapipe as mp

# =========================
# Pull-up Analyzer + Render
# =========================

class PullUpAnalyzer:
    """
    מזהה ריפים במתח, בודק טכניקה, ומחשב ROM-Up% לכל פריים.
    מחזיר ציון טכניקה מעוגל ל-.0/.5 עם מינימום 4.0, וסט פידבקים/טיפים באנגלית.
    """

    def __init__(
        self,
        angle_drop_threshold: float = 18.0,  # כמה המרפק "נסגר" בתחילת ריפ
        min_separation: int = 5,            # מינימום פריימים בין ריפים
        nose_rise_thresh: float = 0.001     # כמה ה-NOSE צריך "לעלות" (y קטן) בתחילת ריפ
    ):
        self.angle_drop_threshold = angle_drop_threshold
        self.min_separation = min_separation
        self.nose_rise_thresh = nose_rise_thresh

    # ---------- Public: analyze all ----------
    def analyze_all_reps(self, frames):
        """
        frames: list[dict], בכל איבר מיפוי שם לנקודת גוף: {'NOSE': (x,y,z), 'LEFT_WRIST':(...), ...}
        """
        rep_ranges = self.segment_reps(frames)

        rep_reports = []
        all_errors, all_tips = [], []

        # רצף ROM לכל הווידאו (לפי frames) - יתמלא בהמשך
        full_rom = [0.0] * len(frames)

        for (start, end) in rep_ranges:
            rep_frames = frames[start:end]
            result = self.analyze_rep(rep_frames)

            rep_reports.append(result)
            all_errors.extend(result["errors"])
            all_tips.extend(result["tips"])

            # "מריחת" ה-ROM של הריפ הזה חזרה למיקום המקורי
            rom_series = result.get("rom_series", [])
            for i, v in enumerate(rom_series, start=start):
                if 0 <= i < len(full_rom):
                    # אם יש חפיפה בין ריפים, נשמור את המקסימום
                    full_rom[i] = max(full_rom[i], v)

        if rep_reports:
            raw_scores = [r["technique_score"] for r in rep_reports]
            avg_score = np.mean(raw_scores)
            rounded_score = float(np.clip(round(avg_score * 2) / 2, 4.0, 10.0))
        else:
            rounded_score = 0.0

        feedback = list(set(all_errors)) if all_errors else (["Great form! Keep it up"] if rep_reports else ["No pull-ups detected"])
        tips = list(set(all_tips))

        return {
            "rep_count": len(rep_reports),
            "squat_count": len(rep_reports),  # legacy key שהקוד הקיים אולי מצפה לו
            "technique_score": rounded_score,
            "good_reps": sum(1 for r in rep_reports if r["technique_score"] >= 8),
            "bad_reps": sum(1 for r in rep_reports if r["technique_score"] < 8),
            "feedback": feedback,
            "tips": tips,
            "reps": rep_reports,
            "rom_full": full_rom,     # <— חדש: ROM לכל פריים
            "rep_ranges": rep_ranges, # שימושי לאוברליי/דיבוג
        }

    # ---------- Per-rep analysis ----------
    def analyze_rep(self, frames):
        errors, tips = [], []

        elbow_angles = []
        knee_angles = []
        nose_ys = []
        wrist_ys = []

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

        # ===== בדיקות טכניקה =====
        # Full range up: הסנטר אמור לעבור את קו המוט (בערך גובה שורש כף היד)
        if not any(n < w - 0.005 for n, w in zip(nose_ys, wrist_ys)):
            errors.append("Try to pull a bit higher – chin past the bar")

        # Full extension bottom – המרפקים צריכים להיות ישרים בתחילת הריפ (זווית מקסימלית גבוהה דיה)
        if max(elbow_angles, default=0) < 155:
            errors.append("Start each rep from straight arms for full range")

        # Kipping (רגליים זזות חזק)
        if knee_angles and (max(knee_angles) - min(knee_angles)) > 40:
            errors.append("Keep legs steadier for more control")

        # טיפ: שליטה בירידה (אקסצנטרי)
        if len(elbow_angles) >= 6:
            descent_len = self.detect_descent_duration(elbow_angles)
            if descent_len < 2:
                tips.append("Control the lowering phase for better muscle growth")

        # ===== ROM-Up% =====
        rom_series = self.compute_rom_series(nose_ys, wrist_ys)
        rom_series = self.ema_smooth(rom_series, alpha=0.25)

        # ===== ציון טכניקה =====
        score_map = {0: 10, 1: 8, 2: 6, 3: 5}
        technique_score = score_map.get(len(errors), 4)

        return {
            "technique_score": technique_score,
            "errors": errors,
            "tips": tips,
            "rom_series": rom_series,  # <— נשתמש ברינדור
        }

    # ---------- Helpers ----------
    def detect_descent_duration(self, angles):
        """
        החזרת "אורך ירידה" גס: כמה פריימים אחרי המינימום עד ששוב עולים בזווית.
        """
        if not angles:
            return 0
        min_idx = int(np.argmin(angles))
        for i in range(min_idx + 1, len(angles)):
            if angles[i] > angles[min_idx] + 5:
                return i - min_idx
        return len(angles) - min_idx

    def compute_rom_series(self, nose_ys, wrist_ys):
        """
        ROM-Up%: 0% בתחתית (nose_y הגדול ביותר), 100% כש‑NOSE מעל קו המוט (מוערך ע"י min(wrist_y)).
        """
        if not nose_ys:
            return []
        if not wrist_ys:
            # אין שורש כף יד — נחזיר אפס (לא נשבור רינדור)
            return [0.0] * len(nose_ys)

        bar_y = min(wrist_ys)       # בערך גובה המוט
        bottom_y = max(nose_ys)     # הכי נמוך של הסנטר (y גדול)
        denom = max(bottom_y - bar_y, 1e-6)

        rom_series = []
        for ny in nose_ys:
            if ny < bar_y:
                rom = 1.0
            else:
                raw = (bottom_y - ny) / denom
                rom = float(np.clip(raw, 0.0, 1.0))
            rom_series.append(rom)
        return rom_series

    @staticmethod
    def ema_smooth(series, alpha=0.25):
        if not series:
            return []
        out = []
        prev = 0.0
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
            ba = a - b
            bc = c - b
            denom = np.linalg.norm(ba) * np.linalg.norm(bc)
            if denom == 0:
                return None
            cos = np.dot(ba, bc) / denom
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return None

    @staticmethod
    def knee_angle(f, side):
        keys = [f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE"]
        if all(k in f for k in keys):
            a, b, c = np.array(f[keys[0]]), np.array(f[keys[1]]), np.array(f[keys[2]])
            ba = a - b
            bc = c - b
            denom = np.linalg.norm(ba) * np.linalg.norm(bc)
            if denom == 0:
                return None
            cos = np.dot(ba, bc) / denom
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return None

    # ---------- Rep segmentation ----------
    def segment_reps(self, frames):
        """
        הפרדה פשוטה לריפים ע"פ:
        - ירידה חדה בזווית המרפק (מרפק נסגר)
        - וה‑NOSE עולה (y קטן) מעבר לסף קטן
        - ושומרים על מרחק מינימלי בין ריפים
        """
        elbow_angles = []
        noses_y = []

        for f in frames:
            l = self.elbow_angle(f, "LEFT")
            r = self.elbow_angle(f, "RIGHT")
            valid = [a for a in (l, r) if a is not None]
            avg = float(np.mean(valid)) if valid else None
            elbow_angles.append(avg)
            noses_y.append(f["NOSE"][1] if "NOSE" in f else None)

        reps = []
        last_rep_index = -self.min_separation
        state = "waiting"
        start_idx = None
        start_angle = None

        for i in range(1, len(elbow_angles)):
            curr_angle = elbow_angles[i]
            prev_angle = elbow_angles[i - 1]
            curr_nose = noses_y[i]
            prev_nose = noses_y[i - 1]

            if None in (curr_angle, prev_angle, curr_nose, prev_nose):
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
# Video processing
# =================

# שלד גוף (בלי פנים)
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
        if name not in lms:
            return None
        x, y, _ = lms[name]
        return int(x * w), int(y * h)

    # קווים
    for a, b in _SKELETON_EDGES:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, (255, 255, 255), 2)

    # נקודות
    for name in _DRAW_LMS:
        p = pt(name)
        if p:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)

def draw_rom_donut(frame, rom, center=(110, 110), radius=60):
    pct = int(round(float(rom) * 100))
    pct = max(0, min(100, pct))

    # טבעת רקע
    cv2.circle(frame, center, radius, (50, 50, 50), 10)

    # קשת ROM-Up (ירקרק עדין)
    angle = int(360 * (pct / 100.0))
    for a in range(0, angle):
        theta = np.deg2rad(a - 90)
        x = int(center[0] + radius * np.cos(theta))
        y = int(center[1] + radius * np.sin(theta))
        cv2.circle(frame, (x, y), 5, (80, 220, 120), -1)

    # טקסטים
    cv2.putText(frame, f"{pct}%", (center[0] - 28, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "ROM-Up", (center[0] - 40, center[1] - radius - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

def run_pullup_analysis(video_path, frame_skip=3, scale=0.3, verbose=True):
    """
    מריץ Mediapipe Pose על הווידאו ומחזיר תוצאת ניתוח מה-Analyzer.
    חשוב: כדי שהרינדור יתאם לאורך ה-ROM, הרץ את render_pullup_video עם אותם frame_skip ו-scale.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

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
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            frame_landmarks = {}
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[mp_pose.PoseLandmark(i).name] = (lm.x, lm.y, lm.z)
            landmarks_list.append(frame_landmarks)
        else:
            # גם אם אין לנדמרקס בפריים הזה (צילומי צד/יציאה מהפריים) נשמור "ריק"
            landmarks_list.append({})

        frame_count += 1
        if verbose and frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    pose.close()

    analyzer = PullUpAnalyzer()
    return analyzer.analyze_all_reps(landmarks_list)

def render_pullup_video(input_path, output_path, analysis_result, frame_skip=3, scale=0.3):
    """
    מרנדר וידאו: שלד גוף + דונאט ROM‑Up.
    השתמש באותם frame_skip ו-scale כמו בפונקציית הניתוח כדי שתווי ה-ROM יתאימו לאורך.
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) / max(frame_skip, 1)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w0, h0))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    rom_full = analysis_result.get("rom_full", [])
    idx = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % max(frame_skip, 1) != 0:
            frame_id += 1
            continue

        frame = cv2.resize(frame, (w0, h0))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        # שלד גוף בלבד
        if res.pose_landmarks:
            lms = {}
            for i, lm in enumerate(res.pose_landmarks.landmark):
                lms[mp_pose.PoseLandmark(i).name] = (lm.x, lm.y, lm.z)
            draw_skeleton(frame, lms)

        # דונאט ROM-Up
        rom = rom_full[idx] if idx < len(rom_full) else 0.0
        draw_rom_donut(frame, rom)

        out.write(frame)
        idx += 1
        frame_id += 1

    cap.release()
    out.release()
    pose.close()

# =================
# Example (CLI)
# =================
if __name__ == "__main__":
    """
    שימוש לדוגמה:
    python pullup_analysis.py input.mp4 output.mp4

    (ודא שהתקנת mediapipe + opencv-python)
    pip install mediapipe opencv-python numpy
    """
    import sys
    if len(sys.argv) >= 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        analysis = run_pullup_analysis(in_path, frame_skip=3, scale=0.3, verbose=True)
        render_pullup_video(in_path, out_path, analysis, frame_skip=3, scale=0.3)
        print("Done:", analysis)
    else:
        print("Usage: python pullup_analysis.py <input.mp4> <output.mp4>")

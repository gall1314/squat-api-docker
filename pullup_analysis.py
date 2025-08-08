# pullup_analysis.py
import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
from PIL import ImageFont, ImageDraw, Image

mp_pose = mp.solutions.pose

# ===================== קבועים =====================
# ספירה (היסטרזיס)
ROM_UP_THRESH   = 0.80   # התחלת חזרה כשעוברים מלמטה ל-80%
ROM_DOWN_THRESH = 0.20   # סיום חזרה/נכונות לריפ הבא כשיורדים מתחת 20%
REP_DEBOUNCE_FRAMES = 6  # מניעת ספירה כפולה

# החלקה ויציבות
EMA_ALPHA = 0.5          # החלקת ROM
MISS_TOL  = 6            # כמה פריימים אפשר "לפספס" לנדמרקס לפני איפוס של ROM לרגע

# UI (תואם בולגרי)
FONT_PATH = "Roboto-VariableFont_wdth,wght.ttf"
REPS_FONT_SIZE = 28
FEEDBACK_FONT_SIZE = 22
ROM_LABEL_FONT_SIZE = 14
ROM_PCT_FONT_SIZE   = 18

DEPTH_RADIUS_SCALE   = 0.70   # יחס לגובה הפס העליון
DEPTH_THICKNESS_FRAC = 0.28
DEPTH_COLOR          = (40, 200, 80)   # BGR ירוק
DEPTH_RING_BG        = (70, 70, 70)

try:
    REPS_FONT = ImageFont.truetype(FONT_PATH, REPS_FONT_SIZE)
    FEEDBACK_FONT = ImageFont.truetype(FONT_PATH, FEEDBACK_FONT_SIZE)
    ROM_LABEL_FONT = ImageFont.truetype(FONT_PATH, ROM_LABEL_FONT_SIZE)
    ROM_PCT_FONT   = ImageFont.truetype(FONT_PATH, ROM_PCT_FONT_SIZE)
except Exception:
    REPS_FONT = ImageFont.load_default()
    FEEDBACK_FONT = ImageFont.load_default()
    ROM_LABEL_FONT = ImageFont.load_default()
    ROM_PCT_FONT = ImageFont.load_default()

# ===================== עזרי ציור (תואם בולגרי) =====================
def draw_plain_text(pil_img, xy, text, font, color=(255,255,255)):
    ImageDraw.Draw(pil_img).text((int(xy[0]), int(xy[1])), text, font=font, fill=color)
    return np.array(pil_img)

def draw_donut(frame, center, radius, thickness, pct):
    """
    pct ∈ [0..1], 0=ריק, 1=מלא. מתחיל מ-12 (למעלה) עם כיוון השעון.
    """
    pct = float(np.clip(pct, 0.0, 1.0))
    cx, cy = int(center[0]), int(center[1])
    radius   = int(radius)
    thickness = int(thickness)
    cv2.circle(frame, (cx, cy), radius, DEPTH_RING_BG, thickness, lineType=cv2.LINE_AA)
    start_ang = -90
    end_ang   = start_ang + int(360 * pct)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_ang, end_ang,
                DEPTH_COLOR, thickness, lineType=cv2.LINE_AA)
    return frame

def draw_overlay(frame, reps=0, feedback=None, rom_up_pct=0.0):
    """
    פס עליון: Reps + ROM-Up donut קטן; פס תחתון: פידבק.
    """
    h, w, _ = frame.shape
    bar_h = int(h * 0.065)

    # פס עליון שקוף
    top = frame.copy()
    cv2.rectangle(top, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(top, 0.55, frame, 0.45, 0)

    # Reps
    pil = Image.fromarray(frame)
    frame = draw_plain_text(pil, (16, int(bar_h*0.22)), f"Reps: {reps}", REPS_FONT)

    # ROM-Up donut בפינה ימין-עליון
    margin = 12
    radius = int(bar_h * DEPTH_RADIUS_SCALE)
    thick  = max(3, int(radius * DEPTH_THICKNESS_FRAC))
    cx = w - margin - radius
    cy = int(bar_h * 0.52)
    frame = draw_donut(frame, (cx, cy), radius, thick, rom_up_pct)

    pil  = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)
    label_txt = "ROM-Up"
    pct_txt   = f"{int(rom_up_pct*100)}%"

    label_w = draw.textlength(label_txt, font=ROM_LABEL_FONT)
    pct_w   = draw.textlength(pct_txt,   font=ROM_PCT_FONT)

    gap    = max(2, int(radius * 0.10))
    block_h = ROM_LABEL_FONT_SIZE + gap + ROM_PCT_FONT_SIZE
    base_y  = cy - block_h // 2

    lx = cx - int(label_w // 2)
    ly = base_y
    draw.text((lx, ly), label_txt, font=ROM_LABEL_FONT, fill=(255,255,255))

    px = cx - int(pct_w // 2)
    py = ly + ROM_LABEL_FONT_SIZE + gap
    draw.text((px, py), pct_txt, font=ROM_PCT_FONT, fill=(255,255,255))
    frame = np.array(pil)

    # פס תחתון לפידבק
    if feedback:
        bottom_h = int(h * 0.07)
        over = frame.copy()
        cv2.rectangle(over, (0, h-bottom_h), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(over, 0.55, frame, 0.45, 0)

        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        tw = draw.textlength(feedback, font=FEEDBACK_FONT)
        tx = (w - int(tw)) // 2
        ty = h - bottom_h + 6
        draw.text((tx, ty), feedback, font=FEEDBACK_FONT, fill=(255,255,255))
        frame = np.array(pil)

    return frame

# ===================== ROM / ספירה =====================
class EMA:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = float(alpha)
        self.v = None
    def update(self, x):
        x = float(x)
        if self.v is None:
            self.v = x
        else:
            a = self.alpha
            self.v = a * x + (1 - a) * self.v
        return self.v

def rom_up_from_landmarks(lms):
    """
    ROM-Up%: כמה כף היד (ממוצע שמאל+ימין) גבוהה מהכתף (y קטן = גבוה).
    מנורמל יחסית לכתף; מוגבל ל[0..1]. יציב ובלתי תלוי במוט.
    """
    try:
        lw = lms[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        rw = lms[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    except Exception:
        return 0.0
    wrist_y = (lw + rw) / 2.0
    shoulder_y = (ls + rs) / 2.0
    # אם כף היד מתחת לכתפיים (wrist_y >= shoulder_y) → 0
    # אם כף היד עולה מעלה (wrist_y << shoulder_y) → עד 1.0
    # סקייל אמפירי: shoulder_y - wrist_y יכול להיות קטן, נכפיל בקבוע
    raw = (shoulder_y - wrist_y) * 3.0   # 3.0 ~ 300% ביחידות normalized
    return float(np.clip(raw, 0.0, 1.0))

def count_reps_from_series(series, up=ROM_UP_THRESH, down=ROM_DOWN_THRESH):
    rep = 0
    dir_flag = 0   # 0=מוכן להתחיל, 1=באפ-פאזה
    last_idx_up = -999
    for i, v in enumerate(series):
        if v > up and dir_flag == 0 and (i - last_idx_up) > REP_DEBOUNCE_FRAMES:
            rep += 1
            dir_flag = 1
            last_idx_up = i
        elif v < down and dir_flag == 1:
            dir_flag = 0
    return rep

# ===================== ריצה ראשית (תואם API) =====================
def run_pullup_analysis(video_path, frame_skip=3, scale=0.4,
                        output_path="pullup_analyzed.mp4",
                        feedback_path="feedback_summary.txt"):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    effective_fps = max(1.0, fps_in / max(1, frame_skip))

    out = None
    frame_no = 0
    ema = EMA(alpha=EMA_ALPHA)
    rom_series = []
    miss = 0

    # מונה חזרות בזמן אמת (לא חובה, אבל יפה ל-UI למעלה)
    live_reps = 0
    dir_flag = 0
    last_up_i = -999

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_skip > 1 and (frame_no % frame_skip) != 0:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        h, w = frame.shape[:2]
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (w, h))

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # שלד
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            lms = results.pose_landmarks.landmark
            miss = 0
            rom = rom_up_from_landmarks(lms)
        else:
            miss += 1
            # כשאין זיהוי – החזר זמנית ל-0, אבל אל "תשבור" את כל הסדרה מיד
            rom = 0.0 if miss <= MISS_TOL else 0.0

        rom_smooth = ema.update(rom)
        rom_series.append(rom_smooth)

        # ספירה חיה (אותה היסטרזיס)
        i = len(rom_series) - 1
        if rom_smooth > ROM_UP_THRESH and dir_flag == 0 and (i - last_up_i) > REP_DEBOUNCE_FRAMES:
            live_reps += 1
            dir_flag = 1
            last_up_i = i
        elif rom_smooth < ROM_DOWN_THRESH and dir_flag == 1:
            dir_flag = 0

        # פידבק קצר בזמן אמת (אופציונלי)
        feedback_live = ""
        if rom_smooth > 0.9:
            feedback_live = "Try to pull a bit higher – chin past the bar"  # יוצג רגעית
        # ציור UI
        frame = draw_overlay(frame, reps=live_reps, feedback=feedback_live, rom_up_pct=rom_smooth)

        out.write(frame)

    pose.close()
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    # ספירה סופית מהסדרה המוחלקת
    rep_count = int(count_reps_from_series(rom_series))
    technique_score = 10.0 if rep_count > 0 else 0.0
    final_feedback = ["Great form! Keep it up 💪"] if rep_count > 0 else ["No pull-ups detected"]

    # קובץ תקציר (כמו בבולגרי)
    with open(feedback_path, "w", encoding="utf-8") as f:
        f.write(f"Total Reps: {rep_count}\n")
        f.write(f"Technique Score: {technique_score}/10\n")
        f.write("ROM-Up = relative raising of wrists vs. shoulders (0–100%)\n")
        for fb in final_feedback:
            f.write(f"- {fb}\n")

    # קידוד יצוא (ffmpeg) – אותו דבר כמו בבולגרי
    encoded_path = output_path.replace(".mp4", "_encoded.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        encoded_path
    ])
    if os.path.exists(output_path):
        os.remove(output_path)

    return {
        "squat_count": rep_count,           # לשמירה על תאימות מפתחות קיימים
        "rep_count": rep_count,
        "technique_score": technique_score,
        "good_reps": rep_count,            # אין חלוקה כרגע – אפשר לשפר בהמשך
        "bad_reps": 0,
        "feedback": final_feedback,
        "reps": [],
        "video_path": encoded_path,
        "feedback_path": feedback_path
    }

# -------------
# CLI (optional)
# -------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        src, dst = sys.argv[1], sys.argv[2]
        res = run_pullup_analysis(src, frame_skip=3, scale=0.4, output_path=dst)
        print(res)
    else:
        print("Usage: python pullup_analysis.py <input.mp4> <output.mp4>")


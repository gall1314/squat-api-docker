import cv2
import mediapipe as mp
import numpy as np

def run_deadlift_analysis(video_path):
    # הגדרת המודל של Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # פתיחת הווידאו לניתוח פריים-אחר-פריים
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    
    # איפוס משתני ספירה וניטור
    total_reps = 0
    proper_reps = 0
    minor_error_count = 0
    severe_error_count = 0

    # ספי זויות (ניתנים לכיוונון)
    minor_threshold = 170.0  # זווית מתחת לזה -> חריגה קלה
    severe_threshold = 160.0  # זווית מתחת לזה -> חריגה חמורה
    threshold_up = 160.0      # זווית בפשיטת ירך שמעליה נספור כחזרה (קרוב לעמידה זקופה)
    threshold_down = 110.0    # זווית בכיפוף ירך שמתחתיה נחשב שירד מספיק לתחילת חזרה חדשה

    # למעקב אחר חזרה נוכחית
    stage = None        # 'down' או 'up' לפי שלב התנועה (ירידה/עלייה)
    min_angle_current_rep = None  # זווית מינימלית בגב במהלך החזרה הנוכחית

    # נחוש את הצד הפונה למצלמה ע"פ נראות האוזניים במסגרת הפריים הראשון שנזהה בו גוף
    head_landmark_idx = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # סוף הווידאו
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if not results.pose_landmarks:
            continue  # דלג על פריים ללא זיהוי תנוחה

        # ראשית, אם לא נקבע עדיין, בחירת נקודת הראש המתאימה (אוזן שמאל/ימין או אף) לפרופיל
        if head_landmark_idx is None:
            # נראות (visibility) של האוזניים: ערך בין 0 ל-1 המעיד עד כמה הנקודה נצפתה בבירור
            left_ear_vis = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].visibility
            right_ear_vis = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].visibility
            if left_ear_vis + right_ear_vis < 0.2:
                # אם שתי האוזניים כמעט לא נראות (אולי פנים למצלמה) – נשתמש באף כנקודת ראש
                head_landmark_idx = mp_pose.PoseLandmark.NOSE
            else:
                # אחרת נשתמש באוזן הפונה למצלמה (הנראית יותר טוב)
                head_landmark_idx = (mp_pose.PoseLandmark.LEFT_EAR 
                                     if left_ear_vis >= right_ear_vis 
                                     else mp_pose.PoseLandmark.RIGHT_EAR)
            # קביעת שלב התנועה ההתחלתי לפי הזווית בהיפ (למקרה שהתרגיל מתחיל מהרמה מהרצפה או מעמידה)
            # חישוב זווית הירך הראשונית כדי לקבוע אם המתרגל בתחילת התנועה למטה או למעלה:
            left_sh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_sh = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            # נקודות אמצע לכתפיים, ירכיים וברכיים עבור חישובי הזוויות
            image_h, image_w = frame.shape[0], frame.shape[1]
            shoulder_mid_x = (left_sh.x + right_sh.x) / 2 * image_w
            shoulder_mid_y = (left_sh.y + right_sh.y) / 2 * image_h
            hip_mid_x = (left_hip.x + right_hip.x) / 2 * image_w
            hip_mid_y = (left_hip.y + right_hip.y) / 2 * image_h
            knee_mid_x = (left_knee.x + right_knee.x) / 2 * image_w
            knee_mid_y = (left_knee.y + right_knee.y) / 2 * image_h
            # חישוב זווית הירך הראשונית (בין כתף-ירך לברך-ירך) כדי לקבוע את מצב ההתחלה
            hip_vector1 = np.array([shoulder_mid_x - hip_mid_x, shoulder_mid_y - hip_mid_y])
            hip_vector2 = np.array([knee_mid_x - hip_mid_x, knee_mid_y - hip_mid_y])
            norm1 = np.linalg.norm(hip_vector1)
            norm2 = np.linalg.norm(hip_vector2)
            initial_hip_angle = 180.0
            if norm1 > 1e-5 and norm2 > 1e-5:  # אם החישוב תקף
                cos_hip = np.dot(hip_vector1, hip_vector2) / (norm1 * norm2)
                cos_hip = np.clip(cos_hip, -1.0, 1.0)
                initial_hip_angle = np.degrees(np.arccos(cos_hip))
            # קביעת השלב ההתחלתי: אם הירך די ישרה (מעל 150°) נניח שהוא מתחיל מעמידה ("up"), אחרת מהרצפה ("down")
            stage = "up" if initial_hip_angle > 150.0 else "down"
            min_angle_current_rep = None  # נאתחל מעקב זווית חזרה בהמשך בהתאם לתנועה

        # שליפת הקואורדינטות הדרושות (בפיקסלים) עבור הראש, כתפיים, ירכיים וברכיים
        landmarks = results.pose_landmarks.landmark
        image_h, image_w = frame.shape[0], frame.shape[1]
        # נקודת ראש (אוזן שמאל/ימין או אף לפי הבחירה שעשינו)
        head_lm = landmarks[head_landmark_idx.value]  # landmark האובייקט
        head_x = head_lm.x * image_w
        head_y = head_lm.y * image_h
        # נקודות אמצע עמוד השדרה העליון והתחתון
        left_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        shoulder_mid_x = (left_sh.x + right_sh.x) / 2 * image_w
        shoulder_mid_y = (left_sh.y + right_sh.y) / 2 * image_h
        hip_mid_x = (left_hip.x + right_hip.x) / 2 * image_w
        hip_mid_y = (left_hip.y + right_hip.y) / 2 * image_h
        # חישוב הזווית בגב (בכתפיים) בין הראש, כתף-אמצע, אגן-אמצע
        v_shoulder_head = np.array([head_x - shoulder_mid_x, head_y - shoulder_mid_y])
        v_shoulder_hip = np.array([hip_mid_x - shoulder_mid_x, hip_mid_y - shoulder_mid_y])
        norm_sh_head = np.linalg.norm(v_shoulder_head)
        norm_sh_hip = np.linalg.norm(v_shoulder_hip)
        angle_back = 180.0
        if norm_sh_head > 1e-5 and norm_sh_hip > 1e-5:
            cos_angle = np.dot(v_shoulder_head, v_shoulder_hip) / (norm_sh_head * norm_sh_hip)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_back = np.degrees(np.arccos(cos_angle))
        # מעקב אחר הזווית המינימלית בחזרה הנוכחית
        if min_angle_current_rep is None:
            # אתחול הערך ההתחלתי במידת הצורך
            min_angle_current_rep = angle_back
        else:
            # עדכון הזווית המינימלית אם הנוכחית קטנה יותר
            if angle_back < min_angle_current_rep:
                min_angle_current_rep = angle_back

        # חישוב זווית הירך (במפרק הירך) בין הכתף לאגן ולברך, לצורך ספירת חזרות
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        knee_mid_x = (left_knee.x + right_knee.x) / 2 * image_w
        knee_mid_y = (left_knee.y + right_knee.y) / 2 * image_h
        v_hip_shoulder = np.array([shoulder_mid_x - hip_mid_x, shoulder_mid_y - hip_mid_y])
        v_hip_knee   = np.array([knee_mid_x - hip_mid_x, knee_mid_y - hip_mid_y])
        norm_hip_sh = np.linalg.norm(v_hip_shoulder)
        norm_hip_kn = np.linalg.norm(v_hip_knee)
        angle_hip = 180.0
        if norm_hip_sh > 1e-5 and norm_hip_kn > 1e-5:
            cos_hip_angle = np.dot(v_hip_shoulder, v_hip_knee) / (norm_hip_sh * norm_hip_kn)
            cos_hip_angle = np.clip(cos_hip_angle, -1.0, 1.0)
            angle_hip = np.degrees(np.arccos(cos_hip_angle))

        # זיהוי שלב התנועה וספירת חזרות בהתאם
        if stage == "down":
            # אם היינו במצב ירידה/התחלה וכעת הירך התיישרה מספיק (מעל threshold_up) – חזרה הושלמה
            if angle_hip > threshold_up:
                total_reps += 1
                # סיווג החזרה לפי מינימום הזווית שנצפה בגב במהלכה
                if min_angle_current_rep < severe_threshold:
                    severe_error_count += 1  # שגיאה חמורה בגב בחזרה זו
                elif min_angle_current_rep < minor_threshold:
                    minor_error_count += 1   # שגיאה קלה (גב מעוגל קלות)
                else:
                    proper_reps += 1         # חזרה תקינה (גב ישר לאורך כל התנועה)
                # איפוס למעקב חזרה הבאה
                stage = "up"
                min_angle_current_rep = None
        elif stage == "up":
            # אם היינו במצב עלייה/סיום חזרה, נחכה שירד מטה מספיק כדי להתחיל חזרה חדשה
            if angle_hip < threshold_down:
                stage = "down"
                min_angle_current_rep = None  # אתחל מעקב זווית לחזרה החדשה (יופעל מחדש בלולאה הבאה)
                # נשמור את הזווית הנוכחית כנקודת התחלה לחזרה החדשה
                min_angle_current_rep = angle_back

    # שחרור המשאבים
    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    # חישוב ציון הטכניקה הכללי מתוך 10
    total_deviations = minor_error_count * 1 + severe_error_count * 2  # חישוב נקודות לגריעה
    score = 10 - total_deviations
    if score < 0:
        score = 0  # הציון המינימלי הוא 0, לא שלילי

    # קביעת משוב מילולי לפי רמת החריגה החמורה ביותר
    feedback = ""
    if severe_error_count > 0:
        feedback = "סכנת פציעה – גב עקום מאוד! יש ליישר את הגב באופן דחוף."
    elif minor_error_count > 0:
        feedback = "שמור על גב ישר לאורך כל התנועה. ישנה עקמומיות קלה שיש לתקן."
    else:
        feedback = "מצוין! הגב נשמר ישר לאורך כל החזרות."

    # הדפסת תקציר (ניתן להתאים לפורמט הממשק באפליקציה)
    print(f"Total Reps: {total_reps}, Proper Reps: {proper_reps}, Reps to Improve: {total_reps - proper_reps}")
    print(f"Overall Technique Score: {score}/10")
    print("Feedback:", feedback)

    # החזרת התוצאות לשימוש באפליקציה, אם נדרש
    return {
        "total_reps": total_reps,
        "proper_reps": proper_reps,
        "reps_to_improve": total_reps - proper_reps,
        "score": score,
        "feedback": feedback
    }



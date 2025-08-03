from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os
import uuid

# ייבוא פונקציות ניתוח
from squat_analysis import run_analysis
from deadlift_analysis import run_deadlift_analysis
from bulgarian_split_squat_analysis import run_bulgarian_analysis
from pullup_analysis import run_pullup_analysis
from barbell_bicep_curl import run_barbell_bicep_curl_analysis  # ✅ חדש

app = Flask(__name__)
CORS(app)

# שימוש בנתיב מוחלט עבור media
MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)

def compress_video(input_path, output_path, scale=0.4):
    import cv2
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        out.write(frame)
    cap.release()
    out.release()
    return True

EXERCISE_MAP = {
    "barbell squat": "squat",
    "barbell back squat": "squat",
    "squat": "squat",
    "deadlift": "deadlift",
    "bulgarian split squat": "bulgarian",
    "split squat": "bulgarian",
    "pull-up": "pullup",
    "pull up": "pullup",
    "barbell bicep curl": "bicep_curl",
}

@app.route('/analyze', methods=['POST'])
def analyze():
    print("==== POST RECEIVED ====")
    print("FORM KEYS:", list(request.form.keys()))
    print("FILES KEYS:", list(request.files.keys()))

    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400

    exercise_type = request.form.get('exercise_type')
    if not exercise_type:
        return jsonify({"error": "Missing exercise_type"}), 400
    exercise_type = exercise_type.lower().strip()
    resolved_type = EXERCISE_MAP.get(exercise_type)

    if not resolved_type:
        return jsonify({"error": f"Unsupported exercise type: {exercise_type}"}), 400

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp.name)

    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"{resolved_type}_{unique_id}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)

    success = compress_video(temp.name, output_path, scale=0.4)
    os.remove(temp.name)
    if not success:
        return jsonify({"error": "Video compression failed"}), 500

    # הרצת הניתוח בהתאם לסוג התרגיל
    if resolved_type == 'squat':
        result = run_analysis(output_path, frame_skip=3, scale=0.4)
    elif resolved_type == 'deadlift':
        result = run_deadlift_analysis(output_path, frame_skip=3, scale=0.4)
    elif resolved_type == 'bulgarian':
        result, _, _ = run_bulgarian_analysis(output_path, frame_skip=3, scale=0.4)
    elif resolved_type == 'pullup':
        result = run_pullup_analysis(output_path, frame_skip=3, scale=0.4)
    elif resolved_type == 'bicep_curl':
        result = run_barbell_bicep_curl_analysis(output_path, frame_skip=3, scale=0.4)
    else:
        return jsonify({"error": "Unsupported exercise type"}), 400

    return jsonify({
        "result": result,
        "video_url": f"/media/{output_filename}"
    })

# ✅ תיקון נתיב לוידאו: תמיכה ב־<path:filename>
@app.route('/media/<path:filename>')
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os
import uuid
import shutil

# ייבוא פונקציות ניתוח
from squat_analysis import run_analysis
from deadlift_analysis import run_deadlift_analysis
from bulgarian_split_squat_analysis import run_bulgarian_analysis
from pullup_analysis import run_pullup_analysis
from barbell_bicep_curl import run_barbell_bicep_curl_analysis

app = Flask(__name__)
CORS(app)

MEDIA_DIR = "media"
os.makedirs(MEDIA_DIR, exist_ok=True)

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

    # שמור קובץ זמני
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp.name)

    unique_id = str(uuid.uuid4())[:8]
    input_filename = f"{resolved_type}_{unique_id}.mp4"
    input_path = os.path.join(MEDIA_DIR, input_filename)

    shutil.copyfile(temp.name, input_path)
    os.remove(temp.name)

    analyzed_filename = f"{resolved_type}_{unique_id}_analyzed.mp4"
    analyzed_path = os.path.join(MEDIA_DIR, analyzed_filename)

    # ניתוח בהתאם לסוג
    if resolved_type == 'squat':
        result = run_analysis(input_path, frame_skip=3, scale=0.4)
    elif resolved_type == 'deadlift':
        result = run_deadlift_analysis(input_path, frame_skip=3, scale=0.4)
    elif resolved_type == 'bulgarian':
        result, _, _ = run_bulgarian_analysis(
            input_path, frame_skip=3, scale=0.4, output_path=analyzed_path
        )
        input_path = analyzed_path  # נעדכן לוידאו המנותח
    elif resolved_type == 'pullup':
        result = run_pullup_analysis(input_path, frame_skip=3, scale=0.4)
    elif resolved_type == 'bicep_curl':
        result = run_barbell_bicep_curl_analysis(input_path, frame_skip=3, scale=0.4)

    # החזרת תשובה עם לינק לוידאו
    response = {
        "result": result,
        "video_url": f"/media/{os.path.basename(input_path)}"
    }
    return jsonify(response)


@app.route('/media/<filename>')
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


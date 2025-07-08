from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import os
import uuid

# ייבוא ניתוחים לפי תרגילים
from squat_analysis import run_analysis
from deadlift_analysis import run_deadlift_analysis

app = Flask(__name__)
CORS(app)

MEDIA_DIR = "media"
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

@app.route('/analyze', methods=['POST'])
def analyze():
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video uploaded"}), 400

    # סוג התרגיל (squat, deadlift וכו׳)
    exercise_type = request.form.get('exercise_type', 'squat')

    # שמירה זמנית
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_file.save(temp.name)

    # יצירת שם קובץ ייחודי
    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"{exercise_type}_{unique_id}.mp4"
    output_path = os.path.join(MEDIA_DIR, output_filename)

    success = compress_video(temp.name, output_path, scale=0.4)
    os.remove(temp.name)
    if not success:
        return jsonify({"error": "Video compression failed"}), 500

    # ניתוב לפונקציה הנכונה
    if exercise_type == 'squat':
        result = run_analysis(output_path, frame_skip=3, scale=0.4)
    elif exercise_type == 'deadlift':
        result = run_deadlift_analysis(output_path, frame_skip=3, scale=0.4)
    else:
        return jsonify({"error": "Unsupported exercise type"}), 400

    result["video_url"] = f"/media/{output_filename}"
    return jsonify(result)

@app.route('/media/<filename>')
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

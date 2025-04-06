from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import os
import threading  
from overlayFunc import process_video
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'uploads/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_video_in_background(input_path, output_path):
    try:
        process_video(input_path, output_path)
    except Exception as e:
        print(f"Processing error: {e}")

@app.template_global()
def timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and file.filename.endswith('.mp4'):
        input_path = os.path.join(UPLOAD_FOLDER, "currentVid.mp4")
        output_path = os.path.join(PROCESSED_FOLDER, "currentVidProc.mp4")

        file.save(input_path)

        threading.Thread(target=process_video_in_background, args=(input_path, output_path)).start()

        return render_template("loading.html")

    return 'Invalid file', 400

@app.route('/results')
def results():
    video_path = os.path.join(PROCESSED_FOLDER, 'currentVidProc.mp4')
    if not os.path.exists(video_path):
        return render_template('error.html')
    return render_template('processing.html')

@app.route('/uploads/processed/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
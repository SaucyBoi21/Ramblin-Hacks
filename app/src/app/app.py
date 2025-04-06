from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import os
from overlayFunc import process_video

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'uploads/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

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

        try:
            process_video(input_path, output_path)
        except Exception as e:
            return f"Processing error: {e}", 500

        return render_template("processing.html")

    return 'Invalid file', 400

@app.route('/uploads/processed/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads/processed', filename)



if __name__ == '__main__':
    app.run(debug=True)
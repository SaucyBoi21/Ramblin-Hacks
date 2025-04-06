from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import os
import threading  
from overlayFunc import process_video
from datetime import datetime
from overlayFuncMap import overlay_prediction
import numpy as np
import cv2

from tensorflow.keras.models import load_model # type: ignore

CNN = load_model("../app/weights/full_CNN.keras")
CNN.load_weights("../app/weights/CNN.weights.h5")

LSTM = load_model("../app/weights/full_LSTM.keras")
LSTM.load_weights("../app/weights/LSTM.weights.h5")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'uploads/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def process_video_in_background(input_path, output_path, output_path_BW):
    try:
        process_video(input_path, output_path, output_path_BW)
        npArray = extract_and_resize_frames(output_path_BW, target_size=(224,224), max_frames=120)
        print(npArray.shape)
        cnnOut = np.expand_dims(CNN.predict(npArray), axis=0)
        print(cnnOut.shape)
        lstmOut = LSTM.predict(cnnOut)
        #print(lstmOut)
        print(lstmOut[0])
        print(lstmOut[0][0][0], lstmOut [0][0][1])
        output_path_pred = os.path.join(PROCESSED_FOLDER, "currentVidPred.mp4")
        overlay_prediction(input_path, output_path_pred, lstmOut)
    except Exception as e:
        print(f"Processing error: {e}")

def extract_and_resize_frames(video_path, target_size=(224,224), max_frames=120):
     cap = cv2.VideoCapture(video_path)
     frames = []
     
     while len(frames) < max_frames:
         ret, frame = cap.read()
         if not ret:
             break
     
         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         frame_resized = cv2.resize(frame, target_size)
         frame_resized = frame_resized / 255.0
         frames.append(frame_resized)
     
     #while len(frames) < max_frames:
     #    frames.append(np.zeros((640, 360)))
 
     frames = padding(frames, 120)
     
     cap.release()
     return np.array(frames)
 
def padding(frames, max_frames):
     new_frames = frames
     while len(new_frames) < max_frames:
         zero_array = np.zeros((224,224, 3))
         new_frames.append(zero_array)
     
     return new_frames
 

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
    print("TEST")
    file = request.files['file']
    print(file.filename)
    print('HEYEYEYEYEYEYEY')
    if file and file.filename.endswith('.mp4'):
        input_path = os.path.join(UPLOAD_FOLDER, "currentVid.mp4")
        output_path = os.path.join(PROCESSED_FOLDER, "currentVidProc.mp4")
        output_path_BW = os.path.join(PROCESSED_FOLDER, "currentVidProcBW.mp4")

        file.save(input_path)
        print('saved' + input_path)
        threading.Thread(target=process_video_in_background, args=(input_path, output_path, output_path_BW)).start()

        return render_template("loading.html")

    return 'Invalid file', 400

@app.route('/results')
def results():
    video_path = os.path.join(PROCESSED_FOLDER, 'currentVidPred.mp4')
    if not os.path.exists(video_path):
        return render_template('error.html')
    return render_template('processing.html')

@app.route('/uploads/processed/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
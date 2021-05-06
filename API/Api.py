import threading
from multiprocessing import Process
from flask import request, flash, url_for
from flask import Response
import gsocketpool.pool
from deepface import DeepFace
from mprpc import RPCPoolClient
from flask import Flask
import os
import cv2
import json
import subprocess
from werkzeug.utils import secure_filename, redirect

app = Flask(__name__)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "dummy"

# init ipc
client_pool = gsocketpool.pool.Pool(RPCPoolClient, dict(host='127.0.0.1', port=6000))

is_false = False


# -----Settings


@app.route('/setting', methods=['POST'])
def setting():
    pass


#   ------Control----------------

@app.route('/stream', methods=['POST'])
def start_stream():
    buffer = []
    data = request.get_json()
    settings = {
        "db_path": request.form['db_path'],
        "source": request.form['source']
    }
    with open("settings.json", 'w') as f:
        json.dump(settings, f)

    try:
        if cv2.VideoCapture(settings["source"]):
            pass
    except:
        return {"message": "can't connect"}, 404

    def Start(*args):
        with client_pool.connection() as client:
            Response(client.call('stream_process'))

    # try:
    with client_pool.connection() as client:
        Response(client.call('stream_process2'))
    # except:
        # return {"message": "can't connect"}, 404

    return {"message": "Accepted"}, 202


@app.route('/restart', methods=['POST'])
def restart():
    # pass
    return DeepFace.stream("./database", source="rtsp://192.168.1.3:5554/playlist.m3u", enable_face_analysis=False)


@app.route('/upload', methods=['POST'])
def upload():
    img = request.files['image']
    name = request.form['name']
    db = os.listdir('./database')
    path = "./database/" + str(name)
    if img.filename == '':
        flash('No selected file')
    if img:
        filename = secure_filename(img.filename)
        os.mkdir(path)
        img.save(path + '/' + filename)
    return "Success!"


# -----Require Data-----


@app.route('/active', methods=['GET'])
def active_count():
    return {"message": threading.active_count()}


@app.route('/getlog', methods=['GET'])
def getLog():
    pass


# -----Register------


@app.route('/register', methods=['POST'])
def register():
    img = request.files['image']
    name = request.form['name']
    db = os.listdir('./database')
    path = ".\database\\" + str(name)
    if name not in db:
        print("hello")
        command = "mkdir " + path
        os.mkdir(path)
    if img.filename == '':
        flash('No selected file')
    if img:
        filename = secure_filename(img.filename)
        img.save(path + '\\' + filename)
    # img = Image.open(request.files['file'])
    return "success"


app.run(host="127.0.0.1", port=5000, debug=True)

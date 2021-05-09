import os
import face_recognition
import cv2
from gevent import monkey
from datetime import datetime
import numpy as np
import threading
import subprocess
import shlex
from subprocess import Popen
from multiprocessing import Process
import time
import json
from gevent.server import StreamServer
from mprpc.server import RPCServer


def SerializeJson(list_logs):
    t = datetime.now().strftime('%d-%m-%Y---%H-%M-%S')
    jsonfile = open(".\\logs\\" + str(t) + '.json', 'w')
    for i in list_logs:
        jsonfile.write(i)
        jsonfile.write("\n")
    jsonfile.close()
    list_logs.clear()


class Core:

    @staticmethod
    def stream(ip, buffer):
        # print(threading.currentThread().getName(), 'Starting')
        if int(ip) == 0:
            capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture("rtsp://"+ip)
        # capture.set(cv2.CAP_PROP_BUFFERSIZE,2)
        while capture.isOpened():
            ret, frame = capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            buffer.append((ret, small_frame))
            face_locations = face_recognition.face_locations(small_frame)
            try:
                (top, right, bottom, left) = face_locations[0]
                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
            except:
                print("No face")
            cv2.imshow(str(threading.currentThread().native_id), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(1 / 24)
        cv2.destroyAllWindows()
        # print(threading.currentThread().getName(), 'Exit')

    @staticmethod
    def processing(db_path, ip, buffer):
        # print(threading.currentThread().getName(), 'Starting')
        path_folder = os.listdir(str(db_path))
        image = []
        path = str(db_path)
        previous_name = ''
        timeApperance = 0
        for i in path_folder:
            image.append(os.listdir(path + i))
        know_face_encodings = []
        know_face_names = []
        for i, j in zip(path_folder, image):
            strings = str(j).strip("'[]'")
            images = face_recognition.load_image_file(path + "/" + str(i) + "/" + strings)
            face_encoding = face_recognition.face_encodings(images)[0]
            know_face_names.append(i)
            know_face_encodings.append(face_encoding)
        list_logs = []
        while True:
            print("processing")
            for i, (ret, frame) in enumerate(buffer):
                # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                face_locations = face_recognition.face_locations(frame)
                if face_locations is None:
                    print("None")
                    continue
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(know_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(know_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = know_face_names[best_match_index]

                    face_names.append(name)

                if len(face_names) >= 1 and previous_name == '':
                    previous_name = face_names[0]
                if len(face_names) >= 1 and previous_name == face_names[0]:
                    timeApperance += 1

                if timeApperance == 3:
                    timeApperance = 0
                    previous_name = ''
                    (top, right, bottom, left), name = face_locations[0], face_names[0]
                    dict = {
                        "time": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                        "region": (top * 4, right * 4, bottom * 4, left * 4),
                        "name": name,
                        "camera-ip": ip
                    }
                    data_json = json.dumps(dict)
                    list_logs.append(data_json)
                # for (top, right, bottom, left), name in zip(face_locations, face_names):
                #     dict = {
                #         "time": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                #         "region": (top * 4, right * 4, bottom * 4, left * 4),
                #         "name": name,
                #         "camera-ip": ip
                #     }
                #     data_json = json.dumps(dict)
                #     list_logs.append(data_json)
                print(len(list_logs), len(buffer))
                if len(list_logs) >= 3:
                    print("write logs")
                    SerializeJson(list_logs)
                    list_logs.clear()
                if len(buffer) > 10:
                    buffer.clear()
            time.sleep(1)
        # print(threading.currentThread().getName(), 'Exit')

    @staticmethod
    # tao image folder cho nhan vien moi
    def register_db_path(db_path, name):
        path = db_path + name
        if os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    # luu hinh anh nhan vien
    def registeremployee(db_path_name):
        return 0


monkey.patch_all()


class DeepFaceServer(RPCServer):

    @staticmethod
    def stream_process():
        print("stream and process")
        with open("../../API/settings.json") as f:
            settings = json.loads(f.read())
        buffer = []
        thread1 = threading.Thread(target=Core.stream, args=(settings['source'], buffer))
        thread2 = threading.Thread(target=Core.processing, args=(settings['db_path'], settings['source'], buffer))
        thread1.start()
        thread2.start()
        # thread1.join()
        # thread2.join()

    @staticmethod
    def stream_process2():
        print("open process")
        args = shlex.join(["python", "Core/Face-Recogition/script.py"])
        p = subprocess.Popen(args=args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.pid

    @staticmethod
    def kill_process(pid):
        return os.kill(pid)


print("--running--")
server = StreamServer(('127.0.0.1', 6000), DeepFaceServer())
server.serve_forever()

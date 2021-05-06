import os
import json
from gevent.server import StreamServer
from gevent import monkey
from mprpc import RPCServer
import threading
import os
import face_recognition
import cv2
from datetime import datetime
import numpy as np
import json
import mprpc


# serializeJson
def SerializeJson(list_logs):
    time = datetime.now().strftime('%d-%m-%Y---%H-%M-%S')
    jsonfile = open("E:\\PycharmProjects\\IPCtest\\logs\\" + str(time) + '.json', 'w')
    for i in list_logs:
        jsonfile.write(i)
        jsonfile.write('\n')
    jsonfile.close()
    list_logs.clear()



class Core:
    @staticmethod
    def detection(ip, db_path):
        capture = cv2.VideoCapture(0)
        path_folder = os.listdir(str(db_path))
        image = []
        path = str(db_path)
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

        process_this_frame = True
        list_logs = []

        previous_name = None
        appearance_times = 0

        while (capture.isOpened()):
            print("capturing")
            ret, frame = capture.read()

            rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if face_locations is None:
                    continue
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(know_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(know_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = know_face_names[best_match_index]

                    face_names.append(name)

            # face_locations = []
            # face_names = []
            # if len(face_names) != 0:
            #     if previous_name is None:
            #         previous_name = face_names[-1]
            #     else:
            #         if face_names[-1] == previous_name:
            #             appearance_times += 1
            #         else:
            #             appearance_times = 0

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                dict = {"time": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),"region": (top, right, bottom, left),"name": name,"camera-ip": ip}
                data_json = json.dumps(dict)
                list_logs.append(data_json)
                cv2.rectangle(frame, (left*4, top*4), (right*4, bottom*4), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if (len(list_logs) >= 5):
                SerializeJson(list_logs)
        capture.release()
        cv2.destroyAllWindows()

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


def detection_(ip, db_path):
    capture = cv2.VideoCapture('rtsp://' + str(ip))
    path_folder = os.listdir(str(db_path))
    image = []
    path = str(db_path)
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

    process_this_frame = True
    list_logs = []
    while (capture.isOpened()):
        ret, frame = capture.read()

        rgb_small_frame = frame

        if process_this_frame:
            face_locations = face_recognition.batch_face_locations(rgb_small_frame)
            if face_locations is None:
                continue
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(know_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(know_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = know_face_names[best_match_index]

                face_names.append(name)
        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            dict = {
                "time": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                "region": (top, right, bottom, left),
                "name": name,
                "camera-ip": ip
            }
            data_json = json.dumps(dict, indent=4)
            list_logs.append(data_json)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if (len(list_logs) >= 500):
            SerializeJson(list_logs)
    capture.release()
    cv2.destroyAllWindows()

monkey.patch_all()


class DeepFaceServer(RPCServer):

    @staticmethod
    def stream():
        print("stream")
        with open("E:/PycharmProjects/Facereg/settings.json") as f:
            settings = json.loads(f.read())
        Core.detection(settings['source'], settings['db_path'])


# print("--running--")
# server = StreamServer(('127.0.0.1', 6000), DeepFaceServer())
# server.serve_forever()

Core.detection("192.168.1.2:5554/playlist.m3u", "E:/PycharmProjects/Facereg/database/")
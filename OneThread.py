import time
import cv2
from threading import Thread
import pickle
from collections import deque
import face_recognition
from datetime import datetime
import json
import numpy as np
import os


class Camera:
    def __init__(self, url, deque_size,camera_id,db_path):
        self.url = url
        self.deque_ = deque(maxlen=deque_size)
        self.camera_id = camera_id
        self.list_log = []
        self.know_face_names =[]
        self.db_path=db_path
        self.name()
        self.camera_thread = Thread(target=self.open_camera, args=())
        #self.camera_thread.daemon=True
        self.process_thread = Thread(target=self.update_frame,args=())
        self.process_thread.daemon=True
        self.svm = pickle.load(open('svm.pkl','rb'))
    def open_camera(self):
        capture = cv2.VideoCapture(self.url)
        i=0
        while True:
            _, frame = capture.read()
            i+=1
            frame_resize = cv2.resize(frame, (640,480))
            cv2.imshow('Video', frame_resize)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cvtColor
            frame_resize = frame_resize[:, :, ::-1]
            if i==5:
                self.deque_.append(frame_resize)
                i=0
        capture.release()
    def update_frame(self):
        while True:
            frame = self.deque_[-1]
            face_names=[]
            face_location = face_recognition.face_locations(frame)
            face_enc = face_recognition.face_encodings(frame, face_location)
            for face_encoding in face_enc:
                proba = self.svm.predict_proba(face_encoding.reshape(1, -1))
                best_class_indices = np.argmax(proba, axis=1)
                best_class_proba = proba[np.arange(len(best_class_indices)), best_class_indices]
                print(best_class_proba)
                if best_class_proba >= 0.8:
                    best_name = self.know_face_names[best_class_indices[0]]
                    face_names.append(best_name)
                    print(best_name)
                else:
                    print('unknown')
            for (top, right, bottom, left), name in zip(face_location, face_names):
                dict = {
                    "time": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                    "region": str((top, right, bottom, left)),
                    "name": name,
                    "camera-ip":self.camera_id
                }

                data_json = json.dumps(dict)
                self.list_log.append((data_json))
            # if (len(self.list_log)>=50):
            #     self.SerializeJson()
    def SerializeJson(self):
        time = datetime.now().strftime('%d-%m-%Y---%H-%M-%S')
        jsonfile = open('D:/pythonProjects/Logs/' + str(time) + '.json', 'w')
        for i in self.list_log:
            jsonfile.write(i+'\n')
        jsonfile.close()
        self.list_log.clear()

    def start(self):
        self.camera_thread.start()
        time.sleep(2)
        print("lenght deque global: "+str(len(self.deque_)))
        self.process_thread.start()
    def name(self):
        path_folder = os.listdir(str(self.db_path))
        for i in path_folder:
            strings = str(i).strip("'[]'")
            self.know_face_names.append(strings)
if __name__ == '__main__':
    task = Camera(0,30,1,db_path = 'D:/pythonProjects/dataset/')
    task.start()



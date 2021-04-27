import time
import cv2
from threading import Thread
import pickle
from collections import deque
import face_recognition
from datetime import datetime

pickle.load(open('svm.pkl','rb'))

class Camera:
    def __init__(self, url, deque_size,camera_id):
        self.url = url
        self.deque_ = deque(maxlen=deque_size)
        self.camera_id = camera_id
        self.camera_thread = Thread(target=self.open_camera, args=())
        #self.camera_thread.daemon=True
        self.process_thread = Thread(target=self.update_frame,args=())
        self.process_thread.daemon=True
        self.model = pickle.load(open('svm.pkl','rb'))
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
            if i==15:
                self.deque_.append(frame_resize)
                i=0
        capture.release()
    def update_frame(self):
        while self.deque_:
            frame = self.deque_[-1]
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                name = str(self.model.predict(face_encoding.reshape(1, -1)))
                face_names.append(name)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                dict = {
                    "time": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                    "region": str((top, right, bottom, left)),
                    "name": name,
                    "camera-ip":self.camera_id
                }
                print(dict)

    def start(self):
        self.camera_thread.start()
        time.sleep(2)
        print("lenght deque global: "+str(len(self.deque_)))
        self.process_thread.start()
        #self.print1()
if __name__ == '__main__':
    task = Camera(0,30,1)
    task.start()



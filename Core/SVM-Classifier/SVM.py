import face_recognition
from sklearn import svm
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN

detector = MTCNN()

encodings = []
names =[]
path_dataset= 'D:/pythonProjects/Face-Recognition-Core/dataset/'
train_dir = os.listdir(path_dataset)
print(train_dir)
for person in train_dir:
    pix =os.listdir(path_dataset+person)
    for person_img in pix:
        face = face_recognition.load_image_file(path_dataset+'/'+person+'/'+person_img)
        face_bbox = detector.detect_faces(face)
        if len(face_bbox)==1:
            x1, y1, width, height = face_bbox[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face_location = [(y1,x2,y2,x1)]
            face_enc = face_recognition.face_encodings(face,face_location)
            face_enc = np.array(face_enc).reshape(-1)
            encodings.append(face_enc)
            names.append(person)
            print(person + '/' + person_img + " was used for training")
        else:
            print(person+'/'+person_img+" was skipped and can't be used for training")
clf = svm.SVC(kernel='linear',probability=True)
clf.fit(encodings,names)
pickle.dump(clf,open('D:/pythonProjects/Face-Recognition-Core/svm.pkl','wb'))


import face_recognition
from sklearn import svm
import numpy as np
import os
import pickle

encodings = []
names =[]
path_dataset= 'D:/pythonProjects/dataset/'
train_dir = os.listdir(path_dataset)
print(train_dir)
for person in train_dir:
    pix =os.listdir(path_dataset+person)
    for person_img in pix:
        face = face_recognition.load_image_file(path_dataset+'/'+person+'/'+person_img)
        face_bbox = face_recognition.face_locations(face)
        if len(face_bbox)==1:
            face_enc = face_recognition.face_encodings(face)[0]
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person+'/'+person_img+"was skipped and can't be used for training")
clf = svm.SVC(gamma='scale',kernel='rbf',probability=True)
clf.fit(encodings,names)
pickle.dump(clf,open('svm.pkl','wb'))


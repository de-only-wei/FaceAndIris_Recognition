import numpy
import os
from sklearn import neighbors
import pickle

# %reload_ext autoreload
# %autoreload 2
import knn_utilities

features_f = numpy.load(os.path.join('Face_Output', 'features.npy'))
labels_f = numpy.load(os.path.join('Face_Output', 'feature_labels.npy'))

features_iL = numpy.load(os.path.join('Iris_Output', 'features_L.npy'))
labels_iL = numpy.load(os.path.join('Iris_Output', 'feature_labels_L.npy'))

features_iR = numpy.load(os.path.join('Iris_Output', 'features_R.npy'))
labels_iR = numpy.load(os.path.join('Iris_Output', 'feature_labels_R.npy'))

print(type(features_f))

face_model_path = 'face_knn_model.clf'
iris_model_L_path = 'iris_knn_model_L.clf'
iris_model_R_path = 'iris_knn_model_R.clf'

face_model: neighbors.KNeighborsClassifier = pickle.load(
    open(os.path.join(face_model_path), 'rb'))
iris_model_L: neighbors.KNeighborsClassifier = pickle.load(
    open(os.path.join(iris_model_L_path), 'rb'))
iris_model_R: neighbors.KNeighborsClassifier = pickle.load(
    open(os.path.join(iris_model_R_path), 'rb'))

new_face_test, new_iris_L_test, new_iris_R_test = knn_utilities.fuse(
    zip(features_f, labels_f), zip(features_iL, labels_iL), zip(features_iR, labels_iR))

f_x, f_y = ([i for i, j in new_face_test],
            [j for i, j in new_face_test])

print(type(f_x))

iL_x, iL_y = [[i for i, j in new_iris_L_test],
              [j for i, j in new_iris_L_test]]
iR_x, iR_y = [[i for i, j in new_iris_R_test],
              [j for i, j in new_iris_R_test]]

face_pred, face_acc = knn_utilities.predict(
    f_x, f_y, model=face_model)
iris_L_pred, iris_L_acc = knn_utilities.predict(
    iL_x, iL_y, model=iris_model_L)
iris_R_pred, iris_R_acc = knn_utilities.predict(
    iR_x, iR_y, model=iris_model_R)

iris_acc = (iris_L_acc + iris_R_acc) / 2

face_weight = 0.4

acc = face_acc * face_weight + iris_acc * (1-face_weight)

print('Face Accuracy:', face_acc)
print('Iris_L Accuracy:', iris_L_acc)
print('Iris_R Accuracy:', iris_R_acc)
print('Iris Accuracy:', iris_acc)
print('Overall:', acc)

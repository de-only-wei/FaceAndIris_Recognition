import math
from sklearn import neighbors
import numpy as np
import os
import os.path
import cv2
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

# /Users/zhengweing/Desktop/Current/CSCI158/FaceAndIris_Recognition/Face_Output/Face_Output_Split_Train/X_train.npy
# /Users/zhengweing/Desktop/Current/CSCI158/FaceAndIris_Recognition/Face_Output/Face_Output_Split_Train/y_train.npy
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(
    x_train,
    y_train,
    n_neighbors: int | None = None,
    knn_algo: str = 'ball_tree',
    verbose: bool = False,
):
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(x_train))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm=knn_algo,
        weights='distance',
    )
    knn_clf.fit(x_train, y_train)

    return knn_clf


def predict(x_test, y_test, knn_clf=None, model: neighbors.KNeighborsClassifier = None, distance_threshold=0.6):
    # print(x_test)
    predictions = model.predict(x_test)

    # Display each data's label and actual class
    # for i in range(len(y_test)):
    #     print("Data", i+1, "Predicted Label:",
    #           predictions[i], "Actual Label:", y_test[i])

    # Calculate accuracy
    correct_predictions = np.sum(predictions == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions
    # print("Accuracy:", accuracy)

    return predictions, accuracy


def fuse(face_test, iris_L_test, iris_R_test):

    face_labels = [label for feature, label in face_test]
    iris_L_labels = [label for feature, label in iris_L_test]
    iris_R_labels = [label for feature, label in iris_R_test]

    common_labels = [
        e for e in face_labels if (e in iris_L_labels) and (e in iris_R_labels)]
    common_labels = list(set(common_labels))

    # print(common_labels)

    new_face_test = []

    for face_tes in face_test:
        if filllllter(face_tes, common_labels()):
            new_face_test.append(face_tes)
    # list(filter(
    #     lambda f: filllllter(f, common_labels), face_test))

    print(new_face_test)

    new_iris_L_test = list(filter(
        lambda iL: filllllter(iL, common_labels), iris_L_test))

    new_iris_R_test = list(filter(
        lambda iR: filllllter(iR, common_labels), iris_R_test))

    # face_f = [feature for feature, label in new_face_test]
    # iris_L_f = [feature for feature, label in new_iris_L_test]
    # iris_R_f = [feature for feature, label in new_iris_R_test]

    # print(len(face_f), '|', len(
    #     iris_L_f), '|', len(iris_R_f))

    return new_face_test, new_iris_L_test, new_iris_R_test


def filllllter(tuuuple, labels):
    feat, lbl = tuuuple
    print(feat)
    if lbl in labels:
        return True
    else:
        return False

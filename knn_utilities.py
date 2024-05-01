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
    # x = []
    # y = []

    # # Loop through each training sample
    # for features, label in zip(x_train, y_train):
    #     # Append features and corresponding label to X and y
    #     x.append(features)
    #     y.append(label)

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
    predictions = model.predict(x_test)

    # Display each data's label and actual class
    for i in range(len(y_test)):
        print("Data", i+1, "Predicted Label:",
              predictions[i], "Actual Label:", y_test[i])

    # Calculate accuracy
    correct_predictions = np.sum(predictions == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)

    return predictions

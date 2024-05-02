import os
import re
import random
import glob
import cv2
import dlib
import face_utilities
import iris_utilities

face_directory = 'Dataset/VISA_Face/VISA_Face'
iris_directory = 'Dataset/VISA_Iris/VISA_Iris'


def get_biometrics(subject_entity: str):
    face = get_face(subject_entity)
    iris_L, iris_R = get_iris(subject_entity)

    face_feature = extract_face_feature(face)
    iris_feature_L = iris_utilities.feature_extraction(iris_L)
    iris_feature_R = iris_utilities.feature_extraction(iris_R)

    return face, face_feature, iris_L, iris_feature_L, iris_R, iris_feature_R


def get_face(subject_entity: str):
    subject_path = ''

    for path in os.listdir(face_directory):
        if re.match(subject_entity, os.path.basename(path)):
            subject_path = path
            break

    if subject_path == '':
        raise 'subject_entity not found'

    while True:
        pathhh = os.path.join(face_directory, subject_path)
        random_image_path = random.choice(os.listdir(pathhh))

        print(path+'/'+random_image_path)

        image = cv2.imread(pathhh+'/'+random_image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        return cv2.resize(image, (400, 300))


def extract_face_feature(_face):
    face_cascade = cv2.CascadeClassifier(
        'Dependencies/haarcascade_frontalface_alt2.xml',
    )

    x, y, width, height = random.choice(
        face_cascade.detectMultiScale(_face, 1.1, 4))
    face = _face[y:y + height, x:x + width]

    detector = dlib.get_frontal_face_detector()
    predictor_path = 'Dependencies/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    dets = detector(face, 1)

    # Iterate over detected faces
    for i, d in enumerate(dets):
        # Predict facial landmarks
        shape = predictor(face, d)

        # Extract features
        eye_distance = shape.part(45).x - shape.part(36).x
        nose_shape = face_utilities.calculate_nose_shape(shape)
        lips_contour = face_utilities.calculate_lips_contour(shape)
        mouth_wrinkles = face_utilities.calculate_mouth_wrinkles(shape)

        # Append features to the feature vector
        return [eye_distance] + nose_shape + lips_contour + mouth_wrinkles


def get_iris(subject_entity: str):
    subject_path = ''
    iris_L = None
    iris_R = None

    for path in glob.iglob(iris_directory + '/*'):
        if re.match(subject_entity, os.path.basename(path)):
            subject_path = path
            print(subject_path)
            break

    if subject_path == '':
        raise 'subject_entity not found'

    while True:
        random_image_path_L = random.choice(os.listdir(subject_path + '/L'))
        print(random_image_path_L)

        image_L = cv2.imread(subject_path + '/L/' + random_image_path_L)

        if image_L is None:
            continue

        hough_image_L, success = iris_utilities.process_hough(
            random_image_path_L, image_L, 50)

        if not success:
            continue

        dougman_image_L = iris_utilities.generate_rubber_sheet_model(
            hough_image_L)

        if dougman_image_L is not None:
            iris_L = dougman_image_L
            break

    while True:
        random_image_path_R = random.choice(os.listdir(subject_path + '/R'))

        image_R = cv2.imread(subject_path + '/R/' + random_image_path_R)

        if image_R is None:
            continue

        hough_image_R, success = iris_utilities.process_hough(
            random_image_path_R, image_R, 50)

        if not success:
            continue

        dougman_image_R = iris_utilities.generate_rubber_sheet_model(
            hough_image_R)

        if dougman_image_R is not None:
            iris_R = dougman_image_R
            break

    return iris_L, iris_R

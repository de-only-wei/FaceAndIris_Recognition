import math
import dlib
import cv2
import numpy as np
import os
import re
import glob
import warnings
import shutil
import sklearn.model_selection as model_selection
import typing

# Define the base directory where the face dataset is located
base_directory = 'Dataset/VISA_Face/VISA_Face'


def parse_face_dataset() -> tuple[list[cv2.typing.MatLike], list[str]]:
    images: list[cv2.typing.MatLike] = []
    labels: list[str] = []

    # Iterate over each directory in the base directory
    for path in glob.iglob(base_directory + '/*'):
        # Extract the filename from the path
        filename = os.path.basename(path)

        # Parse the filename to extract the label
        match = re.search(r'(.*?)_2017_001', filename)
        if match:
            label = match.group(1)
        else:
            warnings.warn(f"No match found for filename: {filename}")
            continue

        for image_path in glob.iglob(path + '/*'):
            try:
                # Read the image as grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    warnings.warn(f"Failed to load image: {image_path}")
                    continue

                # Reduce memory usage
                image = cv2.resize(image, (400, 300))

                images.append(image)
                labels.append(label)

            except Exception as e:
                warnings.warn(f"Error processing image: {image_path}\n{e}")

    print('Total Face Images Found:', len(images))

    return images, labels


def detect_faces(datas: list[tuple[cv2.typing.MatLike, str]], display: bool = False) -> tuple[list[cv2.typing.MatLike], list[str]]:

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(
        'Dependencies/haarcascade_frontalface_alt2.xml',
    )

    cropped_images: list[cv2.typing.MatLike] = []
    cropped_labels: list[str] = []

    for data in datas:
        image, label = data
        for x, y, width, height in face_cascade.detectMultiScale(image, 1.1, 4):
            face = image[y:y + height, x:x + width]
            cropped_images.append(face)
            cropped_labels.append(label)

    return cropped_images, cropped_labels

# PHASE 3 - FACIAL FEATURE EXTRACTION FUNCTION


def extract_features(datas: list[tuple[cv2.typing.MatLike, str]]) -> tuple[list[list], list[str]]:
    # Initialize face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    # detector = cv2.get_frontal_face_detector()
    predictor_path = 'Dependencies/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    # Create output directory if it doesn't exist
    # output_dir = os.path.join(output_dir, 'Face_Output_Feature_Extraction')

    # # Clear output directory if it already exists
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)  # Remove the directory and its contents

    # os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to store features and labels
    features: list[list] = []
    feature_labels: list[str] = []

    # Iterate over images in the input directory
    for data in datas:
        image, label = data
        # Detect faces in the image
        dets = detector(image, 1)

        # Iterate over detected faces
        for i, d in enumerate(dets):
            # Predict facial landmarks
            shape = predictor(image, d)

            # Extract features
            # Distance between the eyes
            eye_distance = shape.part(45).x - shape.part(36).x
            nose_shape = calculate_nose_shape(shape)  # Shape of the nose
            lips_contour = calculate_lips_contour(shape)  # Contour of the lips
            # Patterns of wrinkles around the mouth
            mouth_wrinkles = calculate_mouth_wrinkles(shape)

            # Append features to the feature vector
            feature_vector = [eye_distance] + \
                nose_shape + lips_contour + mouth_wrinkles

            # Add feature vector and filename as label
            features.append(feature_vector)
            feature_labels.append(label)

            # Draw lines between facial landmarks on the image
            # draw_lines(image, shape)

        # Save image with landmarks and detected faces
        # output_path = os.path.join(output_dir, filename)
        # cv2.imwrite(output_path, image)

    return features, feature_labels


def calculate_eye_distance(shape) -> float:
    # Calculate the Euclidean distance between the outer corners of the eyes
    left_eye_outer_corner = (shape.part(36).x, shape.part(36).y)
    right_eye_outer_corner = (shape.part(45).x, shape.part(45).y)
    eye_distance = math.sqrt((right_eye_outer_corner[0] - left_eye_outer_corner[0]) ** 2 + (
        right_eye_outer_corner[1] - left_eye_outer_corner[1])**2)
    return eye_distance


def calculate_nose_shape(shape) -> list:
    nose_shape = []
    # Calculate the width of the nose
    nose_width = shape.part(35).x - shape.part(31).x
    # Calculate the height of the nose
    nose_height = shape.part(50).y - shape.part(30).y
    nose_shape.extend([nose_width, nose_height])
    return nose_shape


def calculate_lips_contour(shape) -> list:
    lips_contour = []
    # Calculate the width of the lips
    lips_width = shape.part(54).x - shape.part(48).x
    # Calculate the height of the lips
    lips_height = shape.part(57).y - shape.part(51).y
    lips_contour.extend([lips_width, lips_height])
    return lips_contour


def calculate_mouth_wrinkles(shape) -> list:
    mouth_wrinkles = []
    # Calculate the difference in y-coordinates between upper and lower lip
    upper_lip_y = shape.part(51).y
    lower_lip_y = shape.part(57).y
    mouth_height = lower_lip_y - upper_lip_y
    # Calculate the width of the mouth
    mouth_width = shape.part(54).x - shape.part(48).x
    mouth_wrinkles.extend([mouth_width, mouth_height])
    return mouth_wrinkles


def draw_lines(image, shape) -> None:
    # Draw lines between specific facial landmarks
    lines = [(30, 33), (48, 54), (48, 57), (36, 45)]  # Nose, lips, eyes
    for start, end in lines:
        # Draw a line between each pair of landmarks
        cv2.line(image, (shape.part(start).x, shape.part(start).y),
                 (shape.part(end).x, shape.part(end).y), (255, 0, 0), 2)


# PHASE 4 - EXTRACT FACIAL LANDMARKS FUNCTION
def extract_facial_landmarks(input_dir, output_dir) -> None:
    # Initialize face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'Dependencies/shape_predictor_68_face_landmarks.dat')

    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, 'Face_Output_Landmark_Extraction')

    # Clear output directory if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the directory and its contents
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over images in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file is an image (JPEG or PNG)
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Read the image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Detect faces in the image
            dets = detector(image, 1)

            # Iterate over detected faces
            for i, d in enumerate(dets):
                # Predict facial landmarks
                shape = predictor(image, d)
                # Extract (x, y) coordinates of all 68 facial landmarks
                landmarks = [(shape.part(i).x, shape.part(i).y)
                             for i in range(68)]

                # Draw landmarks on the image
                for (x, y) in landmarks:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

                # Save the image with landmarks
                output_path = os.path.join(
                    output_dir, f'{os.path.splitext(filename)[0]}_landmarks_{i}.jpg')
                cv2.imwrite(output_path, image)

# PHASE 5 - LANDMARKS TO FEATURES CONVERSION VECTOR FUNCTION


def landmarks_to_features(landmarks, output_dir) -> None:
    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, 'Face_Output_LFCV')

    # Clear output directory if it already exists
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)  # Remove the directory and its contents
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each set of landmarks
    for i, landmark_set in enumerate(landmarks):
        # Flatten the landmark set into a feature vector
        feature_vector = np.array(landmark_set).flatten()
        # Define the output path for saving the feature vector
        output_path = os.path.join(output_dir, f'landmarks_{i}.npy')
        # Save the feature vector as a NumPy binary file
        np.save(output_path, feature_vector)

# PHASE 6 - SPLIT DATA FUNCTION


def split_data(
    features,
    labels,
    test_size: float = 0.2,
    train_directory: str = 'Face_Output/Face_Output_Split_Train',
    test_directory: str = 'Face_Output/Face_Output_Split_Test',
) -> tuple[typing.Any, typing.Any, typing.Any, typing.Any]:
    # Create directories
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        features, labels,
        test_size=test_size,
        random_state=42,
    )

    # Save training data
    np.save(os.path.join(train_directory, 'X_train.npy'), x_train)
    np.save(os.path.join(train_directory, 'y_train.npy'), y_train)

    # Save testing data
    np.save(os.path.join(test_directory, 'X_test.npy'), x_test)
    np.save(os.path.join(test_directory, 'y_test.npy'), y_test)

    # Print the sizes of the training and testing sets
    print("Training set size:", len(x_train))
    print("Testing set size:", len(x_test))

    # View saved data
    print("\nTraining Data:")
    view_saved_data(train_directory)

    print("\nTesting Data:")
    view_saved_data(test_directory)

    return x_train, x_test, y_train, y_test


# Define the function to view saved data
def view_saved_data(directory):
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):  # Check if the file is a NumPy binary file
            # Construct the file path
            filepath = os.path.join(directory, filename)
            # Load the data from the file
            data = np.load(filepath)
            # Print filename and shape of the loaded data
            print(f"Filename: {filename}, Shape: {data.shape}")

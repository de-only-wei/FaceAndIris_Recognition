import math
import dlib
import cv2
import numpy as np
import os
import re
import glob
import warnings
import shutil
from sklearn.model_selection import train_test_split

# Define the base directory where the face dataset is located
base_directory = 'Dataset/VISA_Face/VISA_Face'


def parse_face_dataset() -> list:
    # hello
    face_images = []

    # Iterate over each directory in the base directory
    for path in glob.iglob(base_directory + '/*'):
        # Extract the filename from the path
        filename = os.path.basename(path)

        # Parse the filename to extract the label
        underscore_index = filename.find("_")
        parsed_filename = filename[:underscore_index]
        match = re.search(r"(.*?)_2017_001", filename)
        if match:
            parsed_filename = match.group(1)
        else:
            # Issue a warning if no match is found and skip processing this file
            warnings.warn(f"No match found for filename: {filename}")
            continue

        # Assign the label parsed from the filename
        label = parsed_filename
        # Initialize an image ID counter
        image_id = 0

        # Iterate over each image file in the current directory
        for image_path in glob.iglob(path + '/*'):
            try:
                # Read the image as grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    # Issue a warning if the image fails to load and continue to the next image
                    warnings.warn(f"Failed to load image: {image_path}")
                    continue
                # Resize the image to reduce memory usage
                image = cv2.resize(image, (400, 300))
                # Append the image, image ID, and label to the face_images list
                face_images.append([image, image_id, label])
                # Increment the image ID
                image_id += 1
            except Exception as e:
                # Issue a warning if there's an error processing the image and continue to the next image
                warnings.warn(f"Error processing image: {image_path}\n{e}")

    # Print the total number of face images found
    print('Total Face Images Found: ' + str(len(face_images)))

    # Return the list of face images and their associated metadata
    return face_images


# PHASE 2 - FACE DETECTION FUNCTION
# Writes detected face images to folder
def face_detection(face_images, display):
    # Initialize an empty list to store pre-processed images
    pre_processed_images = []

    # Load the pre-trained face cascade classifier
    face_cascade = cv2.CascadeClassifier(
        'Dependencies/haarcascade_frontalface_alt2.xml')

    # Output directory for storing the detected faces
    output_dir = os.path.join('Face_Output', 'Face_Output_Detection')

    # Clear output directory if it already exists
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each face image in the input list
    for face_image in face_images:
        # Unpack the face image tuple into image, image_id, and label
        (image, image_id, label) = face_image
        image_id += 1  # Increment image ID

        # Detect faces in the image using the cascade classifier
        faces = face_cascade.detectMultiScale(image, 1.1, 4)

        # Iterate over each detected face
        for (x, y, width, height) in faces:
            # Crop the detected face from the original image
            face = image[y:y + height, x:x + width]

            # Save the cropped face image to the output directory
            output_path = os.path.join(
                output_dir, f'{label}_{image_id}_Cropped.jpg')
            cv2.imwrite(output_path, face)

            # Append the cropped face image, image ID, and label to the pre_processed_images list
            pre_processed_images.append([face, image_id, label])

# PHASE 3 - FACIAL FEATURE EXTRACTION FUNCTION


def facial_feature_extraction(input_directory, output_dir):
    # Initialize face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    # detector = cv2.get_frontal_face_detector()
    predictor_path = 'Dependencies/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, 'Face_Output_Feature_Extraction')

    # Clear output directory if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the directory and its contents

    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to store features and labels
    features = []
    labels = []

    # Iterate over images in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file is an image (JPEG or PNG)
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # Read the image
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

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
                lips_contour = calculate_lips_contour(
                    shape)  # Contour of the lips
                # Patterns of wrinkles around the mouth
                mouth_wrinkles = calculate_mouth_wrinkles(shape)

                # Append features to the feature vector
                feature_vector = [eye_distance] + \
                    nose_shape + lips_contour + mouth_wrinkles

                # Add feature vector and filename as label
                features.append(feature_vector)
                labels.append(filename)

                # Draw lines between facial landmarks on the image
                draw_lines(image, shape)

            # Save image with landmarks and detected faces
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, image)

    # Return extracted features and corresponding labels
    return features, labels


def calculate_eye_distance(shape):
    # Calculate the Euclidean distance between the outer corners of the eyes
    left_eye_outer_corner = (shape.part(36).x, shape.part(36).y)
    right_eye_outer_corner = (shape.part(45).x, shape.part(45).y)
    eye_distance = math.sqrt((right_eye_outer_corner[0] - left_eye_outer_corner[0]) ** 2 + (
        right_eye_outer_corner[1] - left_eye_outer_corner[1])**2)
    return eye_distance


def calculate_nose_shape(shape):
    nose_shape = []
    # Calculate the width of the nose
    nose_width = shape.part(35).x - shape.part(31).x
    # Calculate the height of the nose
    nose_height = shape.part(50).y - shape.part(30).y
    nose_shape.extend([nose_width, nose_height])
    return nose_shape


def calculate_lips_contour(shape):
    lips_contour = []
    # Calculate the width of the lips
    lips_width = shape.part(54).x - shape.part(48).x
    # Calculate the height of the lips
    lips_height = shape.part(57).y - shape.part(51).y
    lips_contour.extend([lips_width, lips_height])
    return lips_contour


def calculate_mouth_wrinkles(shape):
    mouth_wrinkles = []
    # Calculate the difference in y-coordinates between upper and lower lip
    upper_lip_y = shape.part(51).y
    lower_lip_y = shape.part(57).y
    mouth_height = lower_lip_y - upper_lip_y
    # Calculate the width of the mouth
    mouth_width = shape.part(54).x - shape.part(48).x
    mouth_wrinkles.extend([mouth_width, mouth_height])
    return mouth_wrinkles


def draw_lines(image, shape):
    # Draw lines between specific facial landmarks
    lines = [(30, 33), (48, 54), (48, 57), (36, 45)]  # Nose, lips, eyes
    for start, end in lines:
        # Draw a line between each pair of landmarks
        cv2.line(image, (shape.part(start).x, shape.part(start).y),
                 (shape.part(end).x, shape.part(end).y), (255, 0, 0), 2)


# PHASE 4 - EXTRACT FACIAL LANDMARKS FUNCTION
def extract_facial_landmarks(input_dir, output_dir):
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


def landmarks_to_features(landmarks, output_dir):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, 'Face_Output_LFCV')

    # Clear output directory if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the directory and its contents
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


def split_data(features, labels, train_dir, test_dir):
    # Clear existing directories if they exist
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Create directories
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)

    # Save training data
    np.save(os.path.join(train_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(train_dir, 'y_train.npy'), y_train)

    # Save testing data
    np.save(os.path.join(test_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(test_dir, 'y_test.npy'), y_test)

    # Print the sizes of the training and testing sets
    print("Training set size:", len(X_train))
    print("Testing set size:", len(X_test))

    # View saved data
    print("\nTraining Data:")
    view_saved_data(train_dir)

    print("\nTesting Data:")
    view_saved_data(test_dir)


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

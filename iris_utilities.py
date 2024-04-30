import glob
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
from scipy.interpolate import interp1d

# Hough Transform
# Returns (image, radius, success<true>) / (image, image.shape[0], success<false>)


def process_hough(imagepath, image, radius):
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(image, 11)
    edge = cv2.Canny(image, 100, 200)
    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                               50, param1=ret, param2=30, minRadius=20, maxRadius=100)

    try:
        circles = circles[0, :, :]

        circles = np.int16(np.array(circles))

        success = False

        for i in circles[:]:
            image = image[
                i[1] - i[2] - radius: i[1] + i[2] + radius,
                i[0] - i[2] - radius: i[0] + i[2] + radius
                # i[1] - i[2] : i[1] + i[2], i[0] - i[2] : i[0] + i[2]
            ]
            radius = i[2]

        if (image.size > 0):
            success = True

        return (image, radius, success)

    except:
        image[:] = 255
        print(f"{imagepath} -> No circles (iris) found.")
        success = False
        cv2.imshow("Image", image)
        # Wait for a key press (blocks execution)
        cv2.waitKey(0)
        return (image, image.shape[0], success)

# Fix Images


def remove_reflection(image):  # returns image
    ret, mask = cv2.threshold(
        image, 150, 255, cv2.THRESH_BINARY
    )
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    image_rr = cv2.inpaint(
        image, dilation, 5, cv2.INPAINT_TELEA
    )

    return image_rr

# Daugman Rubber Sheet Model


def generate_rubber_sheet_model(image):  # returns image
    q = np.arange(0.00, np.pi * 2, 0.01)
    inn = np.arange(0, int(image.shape[0] / 2), 1)

    cartisian_image = np.empty(shape=[inn.size, int(image.shape[1]), 3])
    m = interp1d([np.pi * 2, 0], [0, image.shape[1]])

    for r in inn:
        for t in q:
            polarX = int((r * np.cos(t)) + image.shape[1] / 2)
            polarY = int((r * np.sin(t)) + image.shape[0] / 2)
            try:
                cartisian_image[r][int(m(t) - 1)] = image[polarY][polarX]
            except:
                pass

    # for image in cartisian_image:
    #     print(image.size)
    # print(cartisian_image.size)
    # cartisian_image = (filter(lambda image: print((image.size)), cartisian_image))

    return cartisian_image.astype("uint8")

# Parse Iris Dataset
# Returns images(image, radius, success, image_id, label)


def parse_iris_dataset(keep_reflections):
    # eye_num_2 = 0
    # final_output = []
    # lables = []
    label = 0
    eye_images = []
    eye_L_images = []
    eye_R_images = []

    base_directory = 'Dataset/VISA_Iris/VISA_Iris'

    for path in glob.iglob(base_directory+'/*'):
        foldername = os.path.basename(path)
        label = foldername
        print('label: ' + label)
        image_id = 1

        # Process Left Eye
        for image_path in glob.iglob(path+'/L/*'):
            eye = '-left'
            image = cv2.imread(image_path)

            # hough transform
            (hough_image, radius, success) = process_hough(image_path, image, 50)

            if (keep_reflections):
                hough_image = remove_reflection(hough_image)

            if (success):
                # just left iris
                eye_L_images.append((hough_image, radius,
                                    success, image_id, label))
                eye_images.append((hough_image, radius,
                                  success, image_id, label + eye))
                image_id += 1
            else:
                pass
        print('L eye: ' + str(len(eye_L_images)))

        image_id = 1
        # Process Right Eye
        for image_path in glob.iglob(path+'/R/*'):
            eye = '-right'
            image = cv2.imread(image_path)

            # hough transform
            (hough_image, radius, success) = process_hough(image_path, image, 50)

            if (keep_reflections):
                hough_image = remove_reflection(hough_image)

            # image = cv2.resize(image, (400, 300))
            if (success):
                eye_R_images.append((hough_image, radius,
                                    success, image_id, label))
                eye_images.append((hough_image, radius,
                                  success, image_id, label + eye))
                image_id += 1
            else:
                pass
        print('R iris: ' + str(len(eye_R_images)))

        # old code insert here
    print('iris images: ' + str(len(eye_images)))

    return eye_images

    # OLD CODE
    # for image_path in glob.iglob(path+'/L/*'):
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     image = cv2.resize(image, (400, 300))
    #     eye_L_images.append([image, image_id, label]) #just left iris
    #     eye_images.append([image, image_id, label])
    #     image_id += 1
    # print('L eye: ' + str(len(eye_L_images)))

    # for image_path in glob.iglob(path+'/R/*'):
    #     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #     image = cv2.resize(image, (400, 300))
    #     eye_R_images.append([image, image_id, label]) #just right iris
    #     eye_images.append([image, image_id, label])
    #     image_id += 1
    # print('R iris: ' + str(len(eye_R_images)))


# Proccess to Daugman


def proccess_images(eye_images):
    for eye_image in eye_images:
        (image, radius, success, image_id, label) = eye_image
        print(str(image_id) + ': ' + str(success))
        if (success):
            image_daugman = generate_rubber_sheet_model(image)
        # else:
        #     plt.imshow(image)
        #     cv2.waitKey(0)
        print("id" + str(image_id))
        cv2.imwrite(
            f'Iris_Output/{str(label)}.{str(image_id)}.Iris.bmp',
            image_daugman
        )

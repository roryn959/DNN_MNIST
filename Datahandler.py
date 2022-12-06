import numpy as np
import matplotlib.pyplot as plt
import cv2

MAX_PIXEL_VALUE = 255

def load_data(dataset):
    with open('Data/' + dataset + '/images', 'rb') as f:
        # Get magic number
        magic_number = f.read(4)
        # Interpret hex properly
        magic_number = ['{:02X}'.format(byte) for byte in magic_number]
        # Join bytes
        magic_number = ''.join(magic_number)
        # Convert to int
        magic_number = int(magic_number, 16)

        # Image number
        num_images = f.read(4)
        num_images = ['{:02X}'.format(byte) for byte in num_images]
        num_images = ''.join(num_images)
        num_images = int(num_images, 16)

        # Row number
        num_rows = f.read(4)
        num_rows = ['{:02X}'.format(byte) for byte in num_rows]
        num_rows = ''.join(num_rows)
        num_rows = int(num_rows, 16)

        # Image number
        num_columns = f.read(4)
        num_columns = ['{:02X}'.format(byte) for byte in num_columns]
        num_columns = ''.join(num_columns)
        num_columns = int(num_columns, 16)

        X = []
        for i in range(num_images):
            image = []
            for j in range(num_rows*num_columns):
                byte = f.read(1)
                byte = '{:02X}'.format(byte[0])
                byte = int(byte, 16)
                image.append(byte)
            X.append(image)

    with open('Data/' + dataset + '/labels', 'rb') as f:
        # Get magic number
        magic_number = f.read(4)
        # Interpret hex properly
        magic_number = ['{:02X}'.format(byte) for byte in magic_number]
        # Join bytes
        magic_number = ''.join(magic_number)
        # Convert to int
        magic_number = int(magic_number, 16)

        # Labels number
        num_labels = f.read(4)
        num_labels = ['{:02X}'.format(byte) for byte in num_labels]
        num_labels = ''.join(num_labels)
        num_labels = int(num_labels, 16)

        y = []
        for i in range(num_labels):
            byte = f.read(1)
            byte = '{:02X}'.format(byte[0])
            byte = int(byte, 16)
            y.append(byte)

    return np.array(X), np.array(y).astype('uint8')

def scale_data(data):
    # Scale pixel values to be between -1 and 1
    data = (data.astype(np.float32) - MAX_PIXEL_VALUE/2) / (MAX_PIXEL_VALUE/2)
    data = data.reshape(-1, 28**2)
    return data

def load_personal():
    image_data = cv2.imread('submitted_image.png', cv2.IMREAD_GRAYSCALE)
    return image_data
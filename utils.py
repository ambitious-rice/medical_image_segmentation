import cv2
import numpy as np
import PIL.Image as Image
import os
# import matplotlib.pyplot as plt


def load_image(path: str):
    return np.array(Image.open(path))


def convert_to_gray(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def convert_to_gradient(image: np.ndarray):
    return cv2.Laplacian(image, cv2.CV_64F)


def save_mask(image: np.ndarray, path: str):
    # if path does not exist, create it
    image = image * 255
    image = image.astype(np.uint8)
    cv2.imwrite(path, image)


def save_image(image: np.ndarray, path: str):
    cv2.imwrite(path, image)


if __name__ == "__main__":
    img = load_image("image/Data/Image/ISIC_0000000.png")
    gray = convert_to_gray(img)
    gradient = convert_to_gradient(gray)
    #show
    cv2.imshow("original_image", img)
    cv2.imshow("gray", gray)
    cv2.imshow("gradient", gradient)
    print(gradient.shape)
    cv2.waitKey(0)

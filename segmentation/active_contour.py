import cv2
import numpy as np
import skimage.segmentation as seg
import matplotlib.pyplot as plt
from utils import *

feature_dict = {
    'gray': convert_to_gray,
    'gradient': convert_to_gradient,
    'edge': convert_to_edge,
    'laws': convert_to_laws_features,
    'gabor': convert_to_gabor,
    'rgb': lambda x: x,
    'cnn': convert_to_cnn_feature
}


def active_contour(image_path: str, feature='gradient'):
    # Read the image
    raw_image = cv2.imread(image_path)

    # gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    # print(gray_image.shape, raw_image.shape)
    # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # image = cv2.Canny(blurred_image, 100, 200)
    if feature in feature_dict:
        image = feature_dict[feature](raw_image)
    else:
        raise ValueError(
            'Feature should be either "rgb", "gradient" or "gray".')
    # Define the initial contour - a circle as an initial point for the active contour
    height, width = raw_image.shape[:2]
    s = np.linspace(0, 2 * np.pi, 800)
    x = width // 2 + (width // 3) * np.cos(
        s)  # Center at (100, 100), radius 80
    y = height // 2 + (height // 3) * np.sin(s)
    init = np.array([x, y]).T
    # Perform active contour model segmentation
    snake = seg.active_contour(image,
                               snake=init,
                               alpha=0.9,
                               beta=100,
                               gamma=0.001)

    # Create a binary mask
    mask2 = np.zeros(raw_image.shape[:2], dtype=np.uint8)
    snake = np.round(snake).astype(int)
    cv2.fillConvexPoly(mask2, snake, 1)
    # Generate segmented image
    segmented_image = raw_image * mask2[:, :, np.newaxis]

    return mask2, segmented_image, raw_image


if __name__ == "__main__":
    # 示例用法
    mask2, segmented_image, image = active_contour(
        '/home/fanzhijie/code/homework/medical_image_process/image/Data/Image/ISIC_0000012.png'
    )  # 替换为你的图像路径
    # 显示原始图像和分割结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Image')
    plt.axis('off')
    plt.show()

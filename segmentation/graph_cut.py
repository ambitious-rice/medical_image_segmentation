import cv2
import numpy as np
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


def graph_cut(image_path, feature='rgb', iteration=6):
    # 读取图像
    raw_image = cv2.imread(image_path)
    if feature in feature_dict:
        image = feature_dict[feature](raw_image)
    else:
        raise ValueError(
            'Feature should be either "rgb", "gradient" or "gray".')
    mask = np.zeros(image.shape[:2], np.uint8)  # 创建一个与图像大小相同的掩码

    # 创建前景和背景模型
    bgd_model = np.zeros((1, 65), np.float64)  # 背景模型
    fgd_model = np.zeros((1, 65), np.float64)  # 前景模型

    # 定义矩形区域 (x, y, width, height) - 根据需要调整
    height, width = image.shape[:2]
    rect = (int(width / 6), int(height / 6), int(width / 1.5),
            int(height / 1.5))
    # 应用 GrabCut 算法
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iteration,
                cv2.GC_INIT_WITH_RECT)

    # 将掩码转换为二进制图像
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 生成分割后的图像
    segmented_image = raw_image * mask2[:, :, np.newaxis]

    return mask2, segmented_image, raw_image


if __name__ == "__main__":
    # 示例用法
    mask2, segmented_image, image = graph_cut(
        '/home/fanzhijie/code/homework/medical_image_process/image/Data/Image/ISIC_0000043.png',
        'gradient')  # 替换为你的图像路径
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

import cv2
import numpy as np
import matplotlib.pyplot as plt


def graph_cut(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], np.uint8)  # 创建一个与图像大小相同的掩码

    # 创建前景和背景模型
    bgd_model = np.zeros((1, 65), np.float64)  # 背景模型
    fgd_model = np.zeros((1, 65), np.float64)  # 前景模型

    # 定义矩形区域 (x, y, width, height) - 根据需要调整
    height, width = image.shape[:2]
    rect = (width // 4, height // 4, width // 2, height // 2)

    # 应用 GrabCut 算法
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5,
                cv2.GC_INIT_WITH_RECT)

    # 将掩码转换为二进制图像
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 生成分割后的图像
    segmented_image = image * mask2[:, :, np.newaxis]

    return mask2, segmented_image, image


if __name__ == "__main__":
    # 示例用法
    mask2, segmented_image, image = graph_cut(
        '/home/fanzhijie/code/homework/medical_image_process/image/Data/Image/ISIC_0000028.png'
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

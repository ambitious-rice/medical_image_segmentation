import cv2
import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
from torchvision import models


def load_image(path: str):
    return np.array(Image.open(path))


def convert_to_gray(image: np.ndarray):  #强度特征
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #转换为三通道
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def convert_to_gradient(image: np.ndarray):  #强度特征
    gradient = cv2.Laplacian(image, cv2.CV_64F)
    # 将单通道梯度复制到三通道

    return np.uint8(gradient)


def convert_to_edge(image: np.ndarray):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    edge = cv2.Canny(gray_image, 100, 200)
    return edge


def laws_kernels():
    # 定义 Laws' kernels
    L5 = np.array([1, 4, 6, 4, 1])  # 平滑核
    E5 = np.array([-1, -4, 0, 4, 1])  # 边缘核
    S5 = np.array([-1, 0, 2, 0, -1])  # 细节核
    W5 = np.array([1, -4, 6, -4, 1])  # 纹理核

    # 生成二维 Laws' kernels
    kernels = {
        'L5': np.outer(L5, L5),
        'E5': np.outer(E5, E5),
        'S5': np.outer(S5, S5),
        'W5': np.outer(W5, W5)
    }
    return kernels


def convert_to_laws_features(image: np.ndarray) -> np.ndarray:
    """
    提取 Laws' kernel 特征
    :param image: 输入 3 通道图像 (H, W, C)
    :return: 输出 3 通道图像 (H, W, C)，每个通道为提取的特征
    """
    # 确保输入图像为 RGB
    if image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel (RGB) image.")

    # 获取 Laws' kernels
    kernels = laws_kernels()

    # 创建一个空的特征图像，形状为 (H, W, 3)
    feature_image = np.zeros_like(image, dtype=np.float32)

    # 对每个通道应用所有的 Laws' kernels，并将结果累加到特征图像中
    for channel in range(3):  # 对 RGB 三个通道
        for kernel in kernels.values():
            filtered_image = cv2.filter2D(image[:, :, channel], -1, kernel)
            # 将过滤后的图像累加到特征图像对应的通道
            feature_image[:, :,
                          channel] += np.abs(filtered_image)  # 使用绝对值以保留特征

    # 归一化特征图像到 [0, 255] 范围
    feature_image = np.clip(feature_image, 0, 255).astype(np.uint8)

    return feature_image


def gabor_filter(image: np.ndarray,
                 ksize=31,
                 sigma=4.0,
                 theta=0,
                 lambd=10.0,
                 gamma=0.5) -> np.ndarray:
    """
    创建 Gabor 滤波器并应用于图像
    :param image: 输入图像
    :param ksize: 滤波器大小
    :param sigma: 高斯标准差
    :param theta: 滤波器方向
    :param lambd: 波长
    :param gamma: 空间纵横比
    :return: 滤波后的图像
    """
    # 创建 Gabor 滤波器
    gabor_kernel = cv2.getGaborKernel((ksize, ksize),
                                      sigma,
                                      theta,
                                      lambd,
                                      gamma,
                                      ktype=cv2.CV_32F)

    # 对图像进行卷积
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)

    return filtered_image


def convert_to_gabor(image: np.ndarray) -> np.ndarray:
    """
    提取图像的 Gabor 特征
    :param image: 输入图像 (H, W, C)
    :return: Gabor 特征图像
    """
    # 确保输入图像为 RGB
    if image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel (RGB) image.")

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 定义 Gabor 滤波器参数
    thetas = [0, np.pi / 4, np.pi / 2]  # 不同方向
    feature_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3),
                             dtype=np.uint8)

    for i, theta in enumerate(thetas):
        filtered_image = gabor_filter(gray_image, theta=theta)

        # 将 Gabor 滤波结果归一化到 0-255 范围
        filtered_image = cv2.normalize(filtered_image, None, 0, 255,
                                       cv2.NORM_MINMAX)

        # 转换为 uint8
        feature_image[:, :, i] = filtered_image.astype(np.uint8)

    return feature_image


def convert_to_cnn_feature(image: np.ndarray) -> np.ndarray:
    """
    将输入图像转换为 CNN 特征，输出维度与输入相同
    :param image: 输入图像，形状为 (H, W, C)
    :return: 提取的特征，形状与输入相同
    """
    # 检查输入图像的维度
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel (H, W, C) image.")

    # 将图像转换为 PIL 图像并进行预处理
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # 转换为 PIL 图像
        transforms.Resize((256, 256)),  # 调整大小
        transforms.CenterCrop((224, 224)),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 预处理输入图像
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0)  # 增加批次维度

    # 加载预训练的 VGG16 模型
    model = models.vgg16(weights='DEFAULT').eval()  # 设置为评估模式

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_tensor = img_tensor.to(device)

    # 提取特征
    with torch.no_grad():
        features = model.features(img_tensor)  # 通过卷积层提取特征

    features = features.squeeze(0).permute(1, 2,
                                           0).cpu().numpy()  # 转换为 NumPy 数组
    # 使用上采样将特征图调整为与输入图像相同的大小
    features_resized = cv2.resize(features,
                                  (image.shape[1], image.shape[0], 3),
                                  interpolation=cv2.INTER_LINEAR)
    # 将特征值范围从 [0, 1] 扩展到 [0, 255]
    features_resized = (features_resized - features_resized.min()) / (
        features_resized.max() - features_resized.min()) * 255.0

    # 转换为 CV_8UC3 格式
    features_resized = np.clip(features_resized, 0,
                               255).astype(np.uint8)  # 确保在 0-255 范围内并转换类型
    print(features_resized.shape)
    return features_resized


def save_mask(image: np.ndarray, path: str):
    # if path does not exist, create it
    image = image * 255
    image = image.astype(np.uint8)
    cv2.imwrite(path, image)


def save_image(image: np.ndarray, path: str):
    cv2.imwrite(path, image)


if __name__ == "__main__":
    img = load_image("image/Data/Image/ISIC_0000000.png")
    convert_to_cnn_feature(img)
    # gray = convert_to_gray(img)
    # gradient = convert_to_gradient(gray)
    # #show
    # cv2.imshow("original_image", img)
    # cv2.imshow("gray", gray)
    # cv2.imshow("gradient", gradient)
    # print(gradient.shape)
    # cv2.waitKey(0)

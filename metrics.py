""""
这是一个计算指标的函数
使用方法为:
python metrics.py --gt 真实数据文件夹 --pred 预测数据文件夹
其中真实数据文件夹和预测数据文件夹中的png图片文件名必须一一对应
"""
from PIL import Image
import numpy as np
import argparse
import os
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import cv2


def parse_args():

    parser = argparse.ArgumentParser(
        description='Evaluate a segmentation result')
    parser.add_argument('--gt',
                        type=str,
                        help='Ground truth segmentation',
                        default='image/Data/Annotation')
    parser.add_argument('--pred',
                        type=str,
                        help='Predicted segmentation',
                        default='image/Data/Annotation')
    return parser.parse_args()


def load_segmentation_dataset(pred, gt):
    gt_pred_pairs = []
    for file in os.listdir(pred):
        if file.endswith('.png'):
            if file not in os.listdir(gt):
                raise ValueError(
                    'There is no corresponding file in the ground truth folder'
                )
            gt_pred_pairs.append(
                (os.path.join(gt, file), os.path.join(pred, file)))
    return gt_pred_pairs


def calculate_metrics(pred, gt):
    gt_pred_pairs = load_segmentation_dataset(pred, gt)
    results = {}
    for gt, pred in gt_pred_pairs:
        file_name = gt.split('/')[-1].replace('.png', '')
        gt_image = load_segmentation_image(gt)
        pred_image = load_segmentation_image(pred)
        dice = 2 * np.sum(
            gt_image * pred_image) / (np.sum(gt_image) + np.sum(pred_image))
        accuracy = accuracy_score(gt_image.flatten(), pred_image.flatten())
        auc = roc_auc_score(gt_image.flatten(), pred_image.flatten())
        tn, fp, fn, tp = confusion_matrix(gt_image.flatten(),
                                          pred_image.flatten()).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        results[file_name] = {
            'dice': dice,
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    for file_name in results:
        print("img :", file_name)
        for metric in results[file_name]:
            print(metric, ":", results[file_name][metric])
        print("-----------------------")
    print("Average results:")
    for metric in results[file_name]:
        print(metric, ":",
              np.mean([results[file_name][metric] for file_name in results]))
    return results


def load_segmentation_image(file):
    image = Image.open(file).convert('L')
    segmnetation_array = np.array(image)
    if segmnetation_array is None:
        raise ValueError('Image file not found')
    else:
        print("successful load image at path: ", file)
    segmnetation_array = segmnetation_array == 255
    return segmnetation_array


if __name__ == '__main__':
    args = parse_args()
    pred = args.pred
    gt = args.gt
    gt_pred_pairs = load_segmentation_dataset(pred, gt)
    results = calculate_metrics(pred, gt)

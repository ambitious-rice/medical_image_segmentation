import argparse
import os

from segmentation.graph_cut import graph_cut
from utils import save_image, save_mask
from metrics import calculate_metrics

methods = {'graph_cut': graph_cut}


def parse_args():
    parser = argparse.ArgumentParser(description='Segmentation function')
    parser.add_argument('--method',
                        choices=methods.keys(),
                        help='Segmentation method',
                        default='graph_cut')
    parser.add_argument('--data_path',
                        help='Path to the image',
                        default='image/Data/Image')
    parser.add_argument('--output_path',
                        help='Path to the output image',
                        default='result'),
    parser.add_argument('--gt',
                        type=str,
                        help='Ground truth segmentation',
                        default='image/Data/Annotation')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    method = args.method
    data_path = args.data_path
    gt_path = args.gt
    mask_path = os.path.join(args.output_path, method, 'mask')
    visual_path = os.path.join(args.output_path, method, 'visual')
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(visual_path, exist_ok=True)

    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        mask2, segmented_image, image = methods[method](img_path)
        save_mask(mask2, os.path.join(mask_path, img_name))
        save_image(segmented_image, os.path.join(visual_path, img_name))
        print(
            f"Segmentation result of {img_name} has been saved to {os.path.join(args.output_path, method)}"
        )

    calculate_metrics(mask_path, gt_path)

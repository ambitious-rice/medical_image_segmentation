# 简单的使用指南

## 1. 克隆仓库

```bash
   git clone https://github.com/ambitious-rice/medical_image_segmentation.git
   
   cd medical_image_process
```

## 2. 下载数据集
* 从群里下载数据集, 在项目目录中创建image文件夹, 将解压结果放到文件夹中, 项目的最终目录为:
```
medical_image_segmentation/
│
├── image/
│ ├── Data
│   ├── Annotation
│   │ ├──xxxx.png
│   │
.   └── Image
.     ├──xxx.png
.
.
├── environment.yml
├── readme.md
└── main.py
```

## 3.配置环境
* 使用conda配置环境
```
conda env create -f environment.yml -n medical_image_seg
conda activate medical_image_seg
```

## 4.运行检测
* 运行`main.py`文件进行图像分割, `--method`指定分割方法, `--data_path`待分割图片地址, `--output_path`结果输出地址, `--gt`ground truth标签地址.
```
python main.py --method graph_cut --data_path image/Data/Image --output_path result --gt image/Data/Annotation
```

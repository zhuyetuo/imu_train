# 狗狗牙齿健康检测（YOLO目标检测）

跟`skin_health/`是同一个组织方式，这个目录管"拍口腔照片 → 检测口腔区域
框 → 判断正常/异常"这个独立任务，跟皮肤评估（抓挠/IMU）是两条不同的
产品线，数据/代码/文档都分开放，不混在一起。

## 任务定义

**目标检测**（不是纯图像分类）：在狗的口腔照片上框出口腔区域，框内
判断"正常"/"异常"两个类别。之所以是检测不是分类，是因为标注时在
Label Studio里对每张照片画了矩形框（框住牛齿/口腔区域），框本身带
类别标签，不是对整张照片打一个标签——这样训练出来的模型除了分类，
还能输出"口腔在图片里的具体位置"，即使后续拍摧照片构图不完全一致
（口腔不在画面正中央、有手指遮挡等），也能先定位再判断。

## 目录结构

```
tooth_health/
├── code/     训练/数据处理脚本
├── data/     Label Studio导出的原始数据 + 合并后的YOLO训练集（.gitignore管理，见data/README.md）
└── docs/     数据集/训练方案设计文档
```

## 数据来源

两个Label Studio project分别导出（各自选"YOLO with Images"导出格式）：
- 一个project以"正常"样本为主
- 一个project以"异常"样本为主

两边导出的zip解压后放进`data/raw_exports/`（见`data/README.md`），用
`code/merge_labelstudio_yolo_exports.py`合并成统一的YOLO训练集（自动
处理两边`classes.txt`顺序可能不一致的问题、去重复文件名、切train/val）。

## 快速开始

```bash
# 1. 把两个Label Studio导出的zip解压到 data/raw_exports/normal_project/ 和
#    data/raw_exports/abnormal_project/ 下（各自保留Label Studio导出的
#    images/labels/classes.txt原始目录结构，不用手动改）

# 2. 合并成统一数据集
python3 tooth_health/code/merge_labelstudio_yolo_exports.py \
    --exports data/raw_exports/normal_project data/raw_exports/abnormal_project \
    --out_dir tooth_health/data/yolo_dataset

# 3. 训练
python3 tooth_health/code/train_yolo26.py --data tooth_health/data/yolo_dataset/data.yaml
```

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

Label Studio project各自导出"YOLO with Images"格式的zip——project数量
不固定，以后每次标完一批新数据（不管是新project还是老project重新导出）
都直接把zip扔进`data/`，不用解压、不用整理目录，`prepare_dataset.py`
自动扫描全部zip、解压、合并。

## 快速开始（也是以后每次补数据的日常操作）

```bash
# 1. Label Studio里导出"YOLO with Images"格式，zip直接拖进 tooth_health/data/
#    （不用解压，不用改文件名，多少个zip都行）

# 2. 一条命令：自动发现所有zip -> 解压 -> 合并成统一训练集
python3 tooth_health/code/prepare_dataset.py

# 3. 训练
python3 tooth_health/code/train_yolo26.py --data tooth_health/data/yolo_dataset/data.yaml
```

`prepare_dataset.py`是幂等的——没变化的zip会跳过重新解压，每次都会用
`data/`下当前所有zip重新生成完整的`yolo_dataset/`（全量重建，不是增量
追加，数据集这个体量下更简单可靠）。自动处理两个容易踩的坑：不同
project的`classes.txt`顺序可能不一致（会导致同一个class_id在不同
project里代表不同类别，且没有任何报错提示）、需要按project分层切
train/val（避免某个类别在验证集里缺失）。

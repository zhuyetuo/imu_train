# 数据目录说明

这个目录本身（除了这份README）不进git——牙齿照片是真实狗的图像数据，
体量也可能比较大，不适合放进代码仓库，`.gitignore`里已经排除。

## 目录约定

```
data/
├── README.md                     这份文件（进git）
├── raw_exports/                  Label Studio原始导出（不进git）
│   ├── normal_project/           "YOLO with Images"格式解压后的内容
│   │   ├── images/
│   │   ├── labels/
│   │   ├── classes.txt
│   │   └── notes.json
│   └── abnormal_project/         同上，另一个project导出的
└── yolo_dataset/                 merge_labelstudio_yolo_exports.py合并后的
    ├── images/train/ ...val/     训练脚本实际读取的位置（不进git）
    ├── labels/train/ ...val/
    └── data.yaml                 ultralytics训练用的数据集描述文件
```

## 怎么拿到`raw_exports/`下的内容

1. Label Studio里打开对应project → Export → 格式选**"YOLO with Images"**
   （不是"YOLO"，那个可能不含实际图片文件；不是"YOLOv8 OBB"，那是给
   旋转框用的，我们标的是普通矩形框）
2. 下载的zip解压到`raw_exports/<project名字>/`下，保留Label Studio给的
   原始目录结构（`images/`/`labels/`/`classes.txt`），不用手动改文件名
   或格式

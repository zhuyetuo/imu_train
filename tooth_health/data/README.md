# 数据目录说明

这个目录本身（除了这份README）不进git——牙齿照片是真实狗的图像数据，
体量也可能比较大，不适合放进代码仓库，`.gitignore`里已经排除。

## 目录约定

```
data/
├── README.md                     这份文件（进git）
├── *.zip                         Label Studio导出的原始zip，直接扔这里，
│                                  不用解压、不用改名，有几个放几个
├── raw_exports/                  prepare_dataset.py自动解压的位置（不进git，自动生成）
│   └── <zip文件名>/               每个zip各自解压成一个子目录
└── yolo_dataset/                 prepare_dataset.py合并后的最终训练集（不进git，自动生成）
    ├── images/train/ ...val/     训练脚本实际读取的位置
    ├── labels/train/ ...val/
    └── data.yaml                 ultralytics训练用的数据集描述文件
```

## 怎么补充新数据（以后每次都这样）

1. Label Studio里打开对应project → Export → 格式选**"YOLO with Images"**
   （不是"YOLO"，那个可能不含实际图片文件；不是"YOLOv8 OBB"，那是给
   旋转框用的，我们标的是普通矩形框）
2. 下载的zip**直接扔进这个`data/`目录**，不用解压、不用改名——不管是
   新project还是老project重新导出的更新版，文件名不冲突就行（Label
   Studio默认导出文件名带project编号+时间戳，天然不会重名）
3. 跑`python3 tooth_health/code/prepare_dataset.py`，自动发现所有zip、
   解压、合并成`yolo_dataset/`——这一步全自动，不用手动进`raw_exports/`
   里整理

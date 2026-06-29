# 视觉行为识别模块

基于视频/图片的狗行为识别，支持三种方式：零样本（CLIP）、大模型API（豆包）、关键点分类器（YOLO26 + MLP）。

## 项目结构

```
imu_train/
├── src/vision/
│   ├── clip_infer.py          ← CLIP 零样本推理（无需训练）
│   ├── llm_infer.py           ← 火山引擎豆包多模态推理
│   ├── label_with_llm.py      ← 批量用 LLM 标注数据集行为标签
│   ├── relabel_for_yolo.py    ← 将行为标签写入 YOLO label 文件
│   ├── build_behavior_dataset.py ← 生成关键点+行为标签 npz
│   ├── quick_validate.py      ← 可视化 YOLO 关键点检测效果
│   ├── train.py               ← MLP/LSTM 行为分类器训练
│   ├── dataset.py             ← 数据集加载
│   └── models/
│       ├── mlp.py             ← PoseMLP
│       └── lstm_classifier.py ← PoseLSTM
├── configs/vision.yaml        ← 视觉模块配置
├── train_dog_pose.py          ← YOLO26 训练脚本（含内存自动清理）
└── train_dog_pose.sh          ← 训练启动脚本（自动选 batch，清理共享内存）
```

---

## 方式一：CLIP 零样本（最快，无需训练，无需 API Key）

预期准确率约 60-75%，适合快速验证。

```bash
pip install transformers torch opencv-python pillow

# 单张图片
python src/vision/clip_infer.py --image 你的图片.jpg

# 视频
python src/vision/clip_infer.py --video 你的视频.mp4 --fps 2 --no_video

# 批量目录
python src/vision/clip_infer.py --video data/raw_vision/videos/
```

模型首次运行自动下载并缓存到 `models/clip/`，之后离线使用。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--image` | — | 图片文件或目录 |
| `--video` | — | 视频文件或目录 |
| `--fps` | 5 | 视频采样帧率 |
| `--no_video` | — | 只输出 CSV，不生成标注视频 |
| `--model` | `clip-vit-base-patch32` | 也可用 `clip-vit-large-patch14`（更准） |
| `--model_dir` | `models/clip` | 本地模型缓存目录 |

结果保存到 `results/vision/clip/`。

---

## 方式二：大模型 API（最准，需要 API Key）

使用火山引擎豆包多模态模型，支持自然语言描述行为细节。

```bash
pip install openai

export ARK_API_KEY="your_key"   # 从 https://console.volcengine.com/ark 获取

# 单张图片
python src/vision/llm_infer.py --image 你的图片.jpg

# 视频（默认 1fps，省钱）
python src/vision/llm_infer.py --video 你的视频.mp4 --fps 1
```

输出示例：
```
行为: Lying  (置信度 95%)
描述: The dog is lying down on the ground, resting with its body flat and relaxed.
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | `doubao-seed-1-6-flash-250828` | 豆包模型名称 |
| `--fps` | 1 | 视频采样帧率（API 有费用，不宜过高） |

结果保存到 `results/vision/llm/`。

---

## 方式三：YOLO26 + 行为分类（训练后最准）

### 步骤 1：训练 YOLO26 关键点检测

```bash
# 自动下载 dog-pose 数据集（24 关键点，6773 训练图）并训练
bash train_dog_pose.sh
# 或指定 batch size
bash train_dog_pose.sh --batch 256
```

训练完成后权重保存至 `runs/pose/train/weights/best.pt`。

指标参考（YOLO26n-pose，100 epoch，RTX 5090）：
- Box mAP50: **0.988**
- Pose mAP50: **0.908**

### 步骤 2：用 LLM 给数据集打行为标签

```bash
export ARK_API_KEY="your_key"

python src/vision/label_with_llm.py --split val   --workers 8
python src/vision/label_with_llm.py --split train --workers 8 --resume
```

约 50 分钟标注全部 8476 张图片，支持断点续跑（`--resume`）。

标注分布参考：
```
Standing  39%   Sitting  26%   Lying  22%
Walking    7%   Trotting  5%   Sniffing  2%
```

### 步骤 3：生成带行为类别的 YOLO 数据集并重新训练

```bash
# 将行为标签写入 YOLO label 文件，生成新数据集
python src/vision/relabel_for_yolo.py

# 从已训练的关键点模型继续训练（加入行为分类头）
bash train_dog_pose.sh \
    --data datasets/dog-pose-behavior/data.yaml \
    --model runs/pose/train/weights/best.pt \
    --batch 256
```

训练后 YOLO 单次推理同时输出：
- **24 个关键点**（body pose）
- **行为类别**（Lying / Sitting / Standing / Walking / Trotting / Sniffing）

### 验证关键点效果

```bash
python src/vision/quick_validate.py \
    --video 你的视频.mp4 \
    --model runs/pose/train/weights/best.pt \
    --conf 0.3
```

输出标注骨骼点的视频 + 检测率统计。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--fps` | 10 | 输出帧率（同时控制推理密度，降低可加快速度） |
| `--conf` | 0.5 | 检测置信度阈值，降低可提高检测率 |
| `--imgsz` | 1280 | 推理分辨率，4K 视频建议 1280 或 1920 |
| `--max_dogs` | 1 | 每帧最多保留几只狗（默认 1，防止误检多框；0=不限制） |
| `--no_skeleton` | — | 只画关键点，不绘制骨骼连线 |

推理使用多线程加速（帧读取线程 + GPU 推理 + 视频写入线程并行），并启用 FP16 和 BN 融合。

### 进一步提速：TensorRT 导出

使用 TensorRT 可比 PyTorch FP16 再提速 2-3 倍：

```bash
# 导出（只需一次，耗时约 2-5 分钟）
yolo export model=runs/pose/train/weights/best.pt format=engine half=True imgsz=1280

# 使用 TensorRT 引擎推理（与 PyTorch 用法相同，自动识别 .engine 后缀）
python src/vision/quick_validate.py \
    --video 你的视频.mp4 \
    --model runs/pose/train/weights/best.engine \
    --conf 0.3 --fps 30
```

> 注意：`.engine` 文件绑定当前机器的 GPU 型号和 TensorRT 版本，换机器需重新导出。导出时的 `imgsz` 要与推理时一致。

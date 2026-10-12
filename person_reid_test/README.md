# 非人脸三人身份识别：第一阶段验证

这是一个独立于机器人、ROS、导航、YOLO、BoxMOT 的可行性验证工程。目标是判断 **OSNet-AIN 的人体外观特征** 是否足以在比赛场景中区分 3 位登记主人。

- 不做、也不保存人脸特征。
- 每个输入必须是一个人的**完整人体裁剪图**；第一阶段不运行检测器。
- 为阻断脸部捷径，特征提取前默认遮蔽人体图上方 22% 的头部区域。
- 每名主人保留全部 20--50 个归一化 embedding，不采用单一均值向量。
- 身份匹配是所有图库特征的 cosine similarity，再取每位主人 Top-K（默认 3）平均。
- 低于阈值或第一、第二候选差距太小，一律输出 `Unknown`。

## 隔离环境

本工程使用独立 Docker 镜像 `wpb-person-reid:test`，只以已有 `wpb-owner-search:noetic` 作为 CUDA/Python 基础层，不会修改正在使用的机器人容器、ROS 工作空间或导航配置。

```bash
cd /home/yanzy/wpb_owner_search/person_reid_test
./scripts/build_image.sh
./scripts/run.sh python3 scripts/download_weights.py
```

下载后应存在：

```
models/osnet_ain_x1_0_msmt17.pth
```

默认请求 CUDA；没有 CUDA 时把 `config.yaml` 的 `model.device` 改为 `cpu`。

## 登记 3 位主人

先将每位主人 20--50 张**人体裁剪图**放入目录。建议正面、轻微左右转、背面、站姿和坐姿都包括。衣服或明显外观变化后应重新登记。

```
data/owner1/
data/owner2/
data/owner3/
```

然后依次执行：

```bash
./scripts/run.sh python3 src/register_person.py --name owner1
./scripts/run.sh python3 src/register_person.py --name owner2
./scripts/run.sh python3 src/register_person.py --name owner3
```

第一次实验可临时用 `--allow-fewer` 烟雾测试；它不适合记录正式成绩。重复登记某人用 `--replace`，避免将旧衣服特征混入新图库。

生成的多样本图库是 `outputs/embeddings/gallery.npz`。

## 单图和视频识别

```bash
./scripts/run.sh python3 src/identify_person.py --input data/query/owner1/example.jpg
./scripts/run.sh python3 src/video_test.py --input data/query/owner1/cropped_person.mp4
```

视频输入也必须是一个人的人体裁剪视频。它默认每 3 帧提取一次特征，对最近 8 次结果做 score averaging，输出：

```
outputs/visualization/video_reid.mp4
outputs/visualization/video_reid.csv
```

## 评测矩阵和姿态指标

查询集可按身份放置：

```
data/query/owner1/*.jpg
data/query/owner2/*.jpg
data/query/owner3/*.jpg
```

或者使用 `data/query/labels.csv`：

```csv
path,owner,pose
owner1/standing_01.jpg,owner1,standing
owner1/back_01.jpg,owner1,back_view
owner2/falling_01.jpg,owner2,falling
```

执行：

```bash
./scripts/run.sh python3 src/evaluate.py
```

将生成：

- `outputs/similarity_matrix/similarity_matrix.csv`
- `outputs/similarity_matrix/similarity_matrix.png`
- `outputs/similarity_matrix/metrics.csv`
- `outputs/similarity_matrix/per_image_results.csv`

`metrics.csv` 记录总体与每个姿态的 Accuracy、Unknown Rate、False Match Rate。只有采集真实赛场样本后才评估是否达到站立/坐姿、背面、躺姿/摔倒等目标；工程不会预先虚报成功率。

## 阈值校准

初始值在 `config.yaml`：

```yaml
reid:
  similarity_threshold: 0.65
  margin_threshold: 0.08
  top_k: 3
  temporal_window: 8
```

先完成 3 人、10 类姿态的真实测试，再基于 `per_image_results.csv` 调整阈值。若躺姿/摔倒显著下降，先保留并分析实验结果；本阶段不改用大模型、不微调。

## 范围边界

第一阶段通过后才按顺序增加：

1. YOLO 的 person crop；
2. BoT-SORT 的短期 `track_id`；
3. ROS1 Noetic 节点与调试图。

`track_id` 只用于连续帧缓存，绝不等同于主人身份。

# 部署清单（不含权重和运行时数据）

本仓库的源码分支不包含模型权重、语音模型、已登记人脸/ReID 数据、Docker 构建缓存或 ROS 编译产物。部署者必须在自己的机器上准备它们。

## 基础环境

- Ubuntu 20.04 + ROS Noetic
- Docker、NVIDIA 驱动、NVIDIA Container Toolkit
- Kinect v2 + libfreenect2 / iai_kinect2 支持
- FTDI 底盘串口、RPLidar
- CUDA GPU（YOLO、YOLO-Pose、OSNet ReID、Qwen 使用）

代码基础依赖在：

- `Dockerfile`
- `requirements-robot.txt`
- 各 ROS 包的 `package.xml`

推荐用 Docker 构建，避免直接改宿主机 Python 环境：

```bash
docker build -t wpb-owner-search:reid-runtime -f Dockerfile .
```

Dockerfile 会安装 ROS、Python/CUDA 运行时、YOLO/InsightFace/Torchreid 依赖以及 Ollama；构建期间需要联网。

编译 ROS 工作空间：

```bash
./scripts/build_workspace.sh
```

## 需要自行准备的模型

请放在 launch 文件设定的路径，或在启动时显式传参覆盖。

| 模块 | 需要的模型/目录 | 默认位置 |
|---|---|---|
| 人体检测 | YOLO World person 权重 | `catkin_ws/src/yoloworld_perception/models/yolov8s-world-person.pt` |
| 动作识别 | YOLO11 pose 权重 | `catkin_ws/src/wpr_task1_owner_search/models/pose/yolo11n-pose.pt` |
| 外观 ReID | OSNet 预训练权重 | `catkin_ws/src/wpb_task1_owner_search/models/reid/` |
| 人脸识别 | InsightFace 模型目录 | `~/.insightface`（容器内 `/home/ubuntu20/.insightface`） |
| ASR | faster-whisper small 中文模型 | `catkin_ws/src/offline_voice_bridge/models/whisper/faster-whisper-small` |
| TTS | Piper 中文模型与配置 | `catkin_ws/src/offline_voice_bridge/models/piper/zh_CN-huayan-medium.onnx` 及 `.json` |
| Qwen | `qwen3.5:0.8b` | Ollama：`ollama pull qwen3.5:0.8b` |

不要提交上述文件，也不要提交 `data/reid_owner_voice`、人脸采集照片或运行日志。

## 真机启动

确认设备：

```bash
lsusb -d 0403:6001   # FTDI 底盘
lsusb -d 045e:02c4   # Kinect v2
ls -l /dev/rplidar   # 雷达
```

完整启动：

```bash
./scripts/run_owner_registration.sh
```

该脚本会通过 `/dev/serial/by-id/*FTDI*` 自动识别底盘串口，不能把底盘端口硬编码为某个 `ttyUSB` 编号。

标定 Kinect：

```bash
roslaunch wpb_home_bringup kinect_test.launch
```

在 GUI 中调节 `kinect_height`、`kinect_pitch`，将最终值写入 `wpb_home_bringup/config/wpb_home.yaml`。

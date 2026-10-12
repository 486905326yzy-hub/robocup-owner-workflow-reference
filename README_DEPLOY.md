# WPB Owner Search Docker Image Bundle

这个目录采用“本地源码 + 固定环境镜像 + Git 版本”的工作方式。Docker 镜像只保存 ROS、CUDA、Python 和硬件依赖；运行和编译时挂载本地 `catkin_ws`，因此改代码不需要重新构建镜像。

## 内容

- `catkin_ws/src/`: ROS Noetic 源码工作区，包含 `wpb_task1_owner_search`、`wpb_home`、`iai_kinect2`、`yoloworld_perception`、`offline_voice_bridge` 等源码；由 Git 管理。
- `catkin_ws/build/`、`catkin_ws/devel/`: 容器内编译生成的产物，可重新生成，不进入 Git。
- `libfreenect2/` 和 `freenect2/`: Kinect v2 依赖源码和安装前缀，镜像构建时会重新安装到 `/home/ubuntu20/freenect2`。
- `home_cache/insightface/`: InsightFace `buffalo_sc` 模型缓存，避免目标机首次运行再下载。
- `home_cache/weights/`: 本机额外权重目录。
- `waypoints.xml`: 当前机器人航点文件，包含 `living_room`。
- `Dockerfile`: 基于 Ubuntu 20.04 / ROS Noetic desktop-full 构建稳定运行环境，不复制 ROS 源码或模型缓存。
- `scripts/`: 构建、保存、加载、运行脚本。

## 首次构建环境镜像

```bash
cd /home/yanzy/wpb_owner_search
./scripts/build_image.sh
```

默认镜像名是 `wpb-owner-search:noetic`。环境或 Dockerfile 依赖变动后才需要重新构建；普通 ROS/Python 代码修改不需要执行 `docker build`。

如需迁移镜像到另一台机器，可执行 `./scripts/build_and_save_image.sh`；它会生成 `wpb-owner-search-noetic.tar.gz` 和校验文件。

## 在另一台电脑加载镜像

```bash
cd /path/to/wpb_owner_search_image_context
./scripts/load_image.sh ./wpb-owner-search-noetic.tar.gz
```

如果只拷贝了镜像 tar，没有整个目录，也可以执行：

```bash
gunzip -c wpb-owner-search-noetic.tar.gz | docker load
```

## 运行真实机器人任务

目标电脑需要满足：

- Ubuntu / Linux 主机可访问机器人硬件设备。
- Docker 已安装。
- 如果使用 YOLO GPU 推理，安装 NVIDIA 驱动和 `nvidia-container-toolkit`。
- Kinect v2 走 USB 3.0，RPLidar 和底盘串口设备在主机可见。
- 推荐主机上存在 `/dev/ftdi` 和 `/dev/rplidar`；如果没有，可以通过 launch 参数覆盖。

首次迁移后，以及每次修改需要重新编译的源码后，先在容器环境中编译本地工作区：

```bash
./scripts/build_workspace.sh
```

完整启动（脚本自动将本地 `catkin_ws`、模型缓存和航点文件挂载进临时运行容器）：

```bash
./scripts/run_robot.sh full
```

推荐重复测试方式，两个终端运行：

```bash
./scripts/run_robot.sh bringup
```

```bash
./scripts/run_robot.sh task
```

如果设备名不同，例如底盘是 `/dev/ttyUSB0`、雷达是 `/dev/ttyUSB1`：

```bash
./scripts/run_robot.sh full serial_port:=/dev/ttyUSB0 rplidar_port:=/dev/ttyUSB1
```

进入容器调试：

```bash
./scripts/run_robot.sh shell
```

容器内常用检查：

```bash
rostopic hz /odom
rostopic hz /scan
rostopic hz /kinect2/qhd/image_color_rect
rostopic hz /kinect2/qhd/points
rostopic echo /perception/person_detections_2d
```

## 日常开发流程

```text
修改本地源码 → ./scripts/build_workspace.sh → ./scripts/run_robot.sh full
```

容器退出后会自动删除；本地源码、编译产物和模型缓存会保留。只有修改 Dockerfile、ROS/CUDA/Python 依赖时才重新构建环境镜像。

## 注意

- 容器内路径保持为 `/home/ubuntu20/catkin_ws`，因为现有 launch/config 中有固定路径引用。
- `build/`、`devel/`、模型缓存、ROS 日志、`.git`、`.codex`、`.agents`、`__pycache__` 均不进入环境镜像。
- 真实运行时脚本使用 `--network host`、`--ipc host`、`--privileged`、`/dev` 挂载，以便 ROS、串口、Kinect、雷达、音频和 RViz 正常工作。
- YOLO 默认使用 `cuda:0`，没有 NVIDIA GPU 或没有配置 Docker GPU Runtime 时任务会失败。

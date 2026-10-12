# RoboCup@Home 主人寻找系统：交接说明

> 工作目录：`/home/yanzy/wpb_owner_search`  
> ROS：Noetic；主运行镜像：`wpb-owner-search:reid-runtime`  
> 原则：**不要重装/下载依赖、不要覆盖现有模型和标定、不要修改现有主人寻找流程。** 新的垃圾识别必须独立验证后再以 ROS 话题方式接入。

## 1. 当前系统目标与完整流程

当前程序用于 RoboCup@Home 的“登记主人 → 巡航找人 → 识别身份和动作 → 接近/对话/响应”。

1. 启动底盘、雷达、定位、导航、RViz、语音、YOLO、主人登记任务。
2. 人靠近机器人时，先近距离采集多角度人脸，再采集全身外观 ReID 特征，并用 ASR 记录/确认名字。
3. 完成登记后机器人按 `living_room,kitchen,bedroom,canteen` 航点巡航。
4. Kinect 图像中检测到人后：
   - YOLO 检测人体；
   - 人脸可用时使用人脸特征；侧脸、远距离或弱人脸时使用 OSNet 外观 ReID；
   - YOLO-Pose 判断挥手、坐姿、躺下/摔倒等动作。
5. 对已登记主人执行对应分支：
   - 挥手：两阶段、安全候选点接近；成功后询问“需要什么帮助”，ASR 获取需求，由 Qwen 0.8B **原意复述**。
   - 坐姿/躺姿：保留电气开关问答分支（Qwen）。
   - 摔倒：先尝试安全接近；即便接近不完整，仍会执行协助机械臂动作。

## 2. 一键启动与容器

### 推荐入口

```bash
cd /home/yanzy/wpb_owner_search
./scripts/run_owner_registration.sh
```

该脚本启动两个容器：

| 容器 | 职责 |
|---|---|
| `wpb-owner-voice-reid` | ROS 主程序：底盘、雷达、导航、RViz、ASR/TTS、YOLO、ReID、任务逻辑、Qwen/Ollama |
| `wpb-kinect-camera` | 独立 `kinect2_bridge`，发布 RGB、深度、彩色点云 |

日志：

```bash
docker logs -f wpb-owner-voice-reid
docker logs -f wpb-kinect-camera
```

停止：

```bash
docker rm -f wpb-owner-voice-reid wpb-kinect-camera
```

### 设备注意事项

- Kinect USB ID：`045e:02c4`。若 `lsusb -d 045e:02c4` 无输出，完整流程不会启动；应检查 Kinect 供电/USB，而不是安装软件。
- 底盘是 FTDI（`0403:6001`），插拔后 `ttyUSB` 编号会变化。
- `run_owner_registration.sh` 已通过 `/dev/serial/by-id/*FTDI*` 自动发现 FTDI，不要再把端口写死为 `ttyUSB0`。
- 雷达使用 `/dev/rplidar`。
- 当前启动脚本会在 Kinect 缺失时**明确失败**，不会伪装成已启动。

## 3. 关键 ROS 入口和文件

| 路径 | 作用 |
|---|---|
| `scripts/run_owner_registration.sh` | 当前完整流程的一键启动脚本；不安装、不下载、不编译 |
| `catkin_ws/src/wpb_task1_owner_search/launch/owner_voice_reid_yolo_pose.launch` | 当前任务顶层 launch；全流程应使用它 |
| `catkin_ws/src/wpb_task1_owner_search/launch/owner_voice_reid_test.launch` | 通用登记、导航、语音、YOLO/ReID 编排 |
| `catkin_ws/src/wpb_task1_owner_search/launch/wpb_real_navigation.launch` | 底盘、雷达、地图、定位、move_base |
| `catkin_ws/src/wpb_task1_owner_search/scripts/owner_voice_reid_test.py` | 人脸/ReID 登记与身份识别核心 |
| `catkin_ws/src/wpb_task1_owner_search/scripts/owner_voice_reid_optimized.py` | 当前优化后的任务状态机入口 |
| `catkin_ws/src/wpb_task1_owner_search/scripts/legacy_action_response_adapter.py` | 动作后接近、挥手问答、摔倒协助、电气开关分支 |
| `catkin_ws/src/wpb_task1_owner_search/scripts/task1_find_owner_real.py` | 导航和接近辅助逻辑 |
| `catkin_ws/src/wpb_home/wpb_home_bringup/config/wpb_home.yaml` | Kinect 安装高度/俯仰标定，**不可随意覆盖** |

旧的 `scripts/run_robot.sh full` 是旧巡航入口，**不等同于当前人脸登记 + ReID 完整流程**，不要用它替代新入口。

## 4. 视觉、识别与算力分工

| 模块 | 用途 | 设备 |
|---|---|---|
| Kinect v2 / `kinect2_bridge` | RGB、深度、`/kinect2/qhd/points` 彩色点云 | CPU/USB |
| YOLO person | 人体框 | GPU `cuda:0` |
| YOLO-Pose | 挥手、坐下、躺下/摔倒动作 | GPU `cuda:0` |
| OSNet ReID | 衣服/人体整体外观身份匹配 | GPU `cuda:0` |
| InsightFace | 近距离/清晰正脸身份匹配 | CPU（当前设定） |
| Faster-Whisper | 离线中文 ASR | CPU / int8 |
| Piper | TTS | CPU |
| Qwen `qwen3.5:0.8b`（Ollama） | 需求原意复述和电气开关对话 | GPU，`OLLAMA_NUM_GPU=999` |

显存高峰是 YOLO + YOLO-Pose + ReID + Qwen 同时常驻。不要为垃圾识别再加载第二套长期驻留的大模型；优先复用已有 YOLO 运行时，或在不需要时延迟加载/卸载。

## 5. 身份识别策略

- 登记不只保存一张图；近距离人脸多角度采集 + 多帧全身 ReID gallery。
- 正脸清晰：人脸优先。
- 侧脸、小脸、置信度低、多脸或远距离：ReID 优先；对镜像脸/人体和上半身外观特征也做匹配。
- ReID 使用多帧 Top-K、相似度阈值、第二名 margin、连续帧一致性，允许 `Unknown`，不强制三选一。
- 动作识别与身份识别分离；身份确定后再执行动作分支。

## 6. 接近与导航策略

- 使用 `move_base`、雷达 costmap、Kinect 点云安全检查。
- 挥手采用队友版本的两阶段接近：先安全粗接近，再以多人候选扇区做精接近；不是只尝试正前方一个点。
- 精接近规划失败但粗接近已经足够安全时，可降级进入挥手问答，避免“物理上已靠近但逻辑上反复离开”。
- 不允许用绕过导航/障碍检查的直接速度控制替代规划。

## 7. Kinect 标定（保留现有值）

当前已保存的标定位于：

```yaml
kinect_height: 1.43548
kinect_pitch: -0.202216
```

要调节时需使用 RViz + `joint_state_publisher_gui`；调好后只保存这两个参数。不要用仓库默认文件覆盖 `wpb_home.yaml`。

## 8. Git 基线

近期关键提交：

| Commit | 含义 |
|---|---|
| `0198cea` | 稳定接近与身份流程检查点 |
| `3f81105` | 迁移队友的侧脸/弱脸 ReID 身份匹配逻辑 |
| `5240732` | 初版完整登记一键启动脚本 |
| `ec96df0` | 用稳定 FTDI USB 标识自动发现底盘端口 |
| `f3b4d7f` | Kinect 未连接时明确提示，不再静默退出 |

修改新功能前先执行：

```bash
git status --short
git log --oneline -8
```

不要 `git reset --hard`、不要覆盖上述标定和已有 owners 数据。

## 9. 新增“垃圾识别”模块的安全集成要求

第一阶段必须独立测试，**不能改动主人寻找 launch 或依赖环境**：

1. 在独立目录/独立 launch 启动垃圾检测。
2. 订阅已有 Kinect 图像话题，例如 `/kinect2/qhd/image_color_rect`；不要新开第二个 `kinect2_bridge`。
3. 先只发布检测结果和调试图，例如：
   - `/garbage_detection`：类别、置信度、bbox、时间戳；
   - `/garbage_detection/debug_image`：可视化画框。
4. 验证模型权重已在本地，再使用；禁止无确认地 `pip install`、`apt install`、下载权重或重建 Docker 镜像。
5. 确认独立节点稳定后，再把它作为**可选节点**加入一个新的 launch，不要直接修改 `owner_voice_reid_yolo_pose.launch` 的关键链路。
6. 如需 3D 坐标，复用 `/kinect2/qhd/points` 和现有标定；不修改 Kinect 高度/俯仰参数。
7. 显存不足时优先让垃圾识别按需运行或复用 YOLO，不要挤掉主人识别、动作识别和 Qwen。

## 10. 当前已知运行状态

- FTDI/雷达端口会随插拔改变，但启动脚本已自动处理 FTDI。
- Kinect 可能因 USB 供电或重新枚举出现 `0 Hz`、`LIBUSB_ERROR_NO_DEVICE`；这是硬件连接状态，不是缺 Python 依赖。先确认 `lsusb -d 045e:02c4`。
- ASR 有时在音频设备重新枚举时首次 `arecord ... No such device`，容器重启后若日志显示 `offline_asr_node ready` 即已恢复。

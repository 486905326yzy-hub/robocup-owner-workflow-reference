# 主人寻找与交互流程参考（给队友）

本文档描述当前本地已验证的 RoboCup@Home 主人登记、寻找、姿态响应和语音交互架构。目标是让其他成员能够参考实现，而不是复制依赖或覆盖现有环境。

## 1. 运行入口与边界

仓库目录：`/home/yanzy/wpb_owner_search`

完整流程入口：

```bash
cd /home/yanzy/wpb_owner_search
./scripts/run_owner_registration.sh
```

该脚本启动两个 Docker 容器：

| 容器 | 内容 |
|---|---|
| `wpb-owner-voice-reid` | ROS 主程序、底盘/雷达/导航、RViz、ASR/TTS、YOLO、ReID、姿态逻辑、Qwen/Ollama |
| `wpb-kinect-camera` | 独立 Kinect2 bridge，发布 RGB、深度和点云 |

关键原则：只使用本地已有镜像 `wpb-owner-search:reid-runtime`、权重和已编译工作空间；不要在运行脚本中 `pip install`、`apt install`、下载模型或重建镜像。

## 2. 设备与稳定启动

底盘为 FTDI USB 串口，插拔后设备号会从 `ttyUSB0` 变为任意 `ttyUSB*`。不要硬编码其编号。

启动脚本优先使用旧别名 `/dev/ftdi`；不存在时解析稳定的 udev 链接：

```text
/dev/serial/by-id/usb-FTDI_...-if00-port0
```

再把解析得到的实际端口传给：

```text
wpb_home_core/serial_port
```

雷达使用 `/dev/rplidar`。Kinect 检测 ID 是 `045e:02c4`。Kinect 未出现在 `lsusb -d 045e:02c4` 时，完整流程应明确终止，而非错误地进入任务状态机。

## 3. GPU / CPU 分工与设置

| 模块 | 当前实现 | 运行设备 | 关键设置 |
|---|---|---|---|
| 人体检测 | YOLO | GPU | `device=cuda:0` |
| 动作识别 | YOLO-Pose | GPU | `action_backend=yolo_pose`、`action_pose_device=cuda:0` |
| 人体外观身份识别 | OSNet ReID | GPU | OSNet embedding，`cuda:0` |
| 人脸身份识别 | InsightFace | CPU | 当前 `face_ctx_id=-1`，减少常驻显存竞争 |
| 中文语音识别 | Faster-Whisper | CPU | int8、输入设备 `plughw:CARD=Sensor,DEV=0` |
| 中文语音播报 | Piper | CPU | `plughw:CARD=sofhdadsp,DEV=0` |
| 语言模型 | Qwen `qwen3.5:0.8b` / Ollama | GPU | Docker 环境变量 `OLLAMA_NUM_GPU=999` |
| Kinect | kinect2_bridge | CPU/USB | `fps_limit=15`、`queue_size=2`、`worker_threads=2` |

容器 GPU 相关环境变量：

```bash
--gpus all \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-e OLLAMA_NUM_GPU=999
```

显存注意事项：YOLO、YOLO-Pose、OSNet、Qwen 同时常驻是峰值场景。因此不要在同一时段无控制地再常驻第二个大视觉模型。新模块优先复用已有 YOLO，或按需加载/释放。

## 4. 主人登记与身份识别

登记分两部分：

1. 近距离多角度人脸采集：正脸、轻微左/右转；保存姓名与人脸特征。
2. 多帧人体外观采集：保留衣服、裤子、体型等 OSNet embedding gallery，而非只存单个均值向量。

运行时的策略：

- 正脸清晰、面积和置信度满足条件：人脸优先。
- 侧脸、小脸、低置信度、多脸或远距离：使用 OSNet ReID 优先。
- ReID 同时使用人体、镜像人体、上半身等特征；多帧 Top-K 相似度、阈值、与第二名的 margin 和连续帧一致性共同决定。
- 允许 `Unknown`，不强制把所有陌生人匹配为三位主人。

身份识别代码：

```text
catkin_ws/src/wpb_task1_owner_search/scripts/owner_voice_reid_test.py
```

## 5. 寻人、动作与接近

流程顺序：

```text
Kinect RGB
  -> YOLO 人体框
  -> 人脸 / ReID 确认主人
  -> YOLO-Pose 识别动作
  -> 动作对应的接近或交互分支
```

身份与姿态是两件事：先确认人，再根据其姿态决定分支。

动作分支核心在：

```text
catkin_ws/src/wpb_task1_owner_search/scripts/legacy_action_response_adapter.py
```

接近约束：

- 使用 `move_base`、雷达 costmap 和 Kinect 点云安全检查。
- 不能用直接速度控制绕过路径规划与避障。
- 挥手采用两阶段方案：先安全粗接近，然后在目标周围的多个朝向/扇区候选点做精接近；不是只尝试“人正前方唯一候选点”。
- 若粗接近已安全到位、精接近因点云或局部规划失败，可降级进入问答，避免机器人实际靠近了却因过严状态判定而又离开。
- 摔倒：先尝试安全接近；接近未完全成功时仍保留协助动作分支。

## 6. 挥手后 Qwen 问答：当前优化逻辑

挥手并确认到达安全交互距离后，执行 `run_waving_help_dialogue`：

1. TTS：询问主人需要什么帮助。
2. 等待 TTS 播放结束，并增加 echo guard 时间，避免扬声器回声被 ASR 当作输入。
3. ASR 在限定时长内收集主人需求。
4. 对 ASR 空结果或回声结果直接重试/给出未听清提示，不发送给 LLM。
5. 将需求发给本地 Qwen 0.8B。
6. Qwen 的 system prompt 约束为：**只原意复述用户刚才的请求，不执行、不判断当前状态、不补充建议、不宣称任务已完成。**
7. 限制最大输出长度；播报固定前缀：`我理解您的需求是：` + Qwen 复述。

示例：用户说“把房间的灯打开”。

正确输出：

```text
我理解您的需求是：把房间的灯打开。
```

错误输出（禁止）：

```text
房间里的灯已经开着。
```

相关 launch 参数位于：

```text
owner_voice_reid_yolo_pose.launch
```

重点参数：

```xml
waving_help_dialogue_enabled:=true
waving_help_listen_seconds:=8.0
waving_help_echo_guard_seconds:=0.8
waving_help_repeat_prefix:="我理解您的需求是："
waving_help_max_repeat_chars:=80
electrical_switch_ollama_model:=qwen3.5:0.8b
electrical_switch_ollama_num_gpu:=999
electrical_switch_ollama_warmup_enabled:=true
```

Qwen 采用启动后台预热，`keep_alive=5m`，避免第一次正式问答加载模型过慢。若预热超时，任务不应崩溃，应走有限的关键词/固定回复后备分支，并在下一次调用继续尝试模型。

## 7. Kinect 点云标定

使用手册对应的专用标定流程：

```bash
roslaunch wpb_home_bringup kinect_test.launch
```

它会同时打开 RViz 与 `Joint State Publisher GUI`。在平整地面前调节：

- `kinect_height`
- `kinect_pitch`

直到地面彩色点云与 RViz 网格重合。保存位置：

```text
catkin_ws/src/wpb_home/wpb_home_bringup/config/wpb_home.yaml
```

当前已保存值：

```yaml
zeros:
  kinect_height: 1.30713
  kinect_pitch: -0.202216
```

不要用上游默认 YAML 覆盖这些值。

## 8. 关键入口清单

| 文件 | 职责 |
|---|---|
| `scripts/run_owner_registration.sh` | 自动发现 FTDI、完整启动两个容器 |
| `launch/owner_voice_reid_yolo_pose.launch` | 当前完整任务 launch |
| `launch/owner_voice_reid_test.launch` | 语音、登记、导航、视觉的通用编排 |
| `launch/wpb_real_navigation.launch` | 真机导航栈 |
| `scripts/owner_voice_reid_optimized.py` | 当前主任务状态机 |
| `scripts/owner_voice_reid_test.py` | 人脸/ReID 核心 |
| `scripts/legacy_action_response_adapter.py` | 挥手、摔倒、坐/躺后续逻辑 |

## 9. 扩展模块要求

新增垃圾识别等功能时，先独立 launch 验证，订阅已有 `/kinect2/qhd/image_color_rect`，发布独立结果/调试话题。不得开启第二个 Kinect bridge，不得覆盖标定 YAML，不得修改主人寻找入口来“顺便集成”。确认稳定后再以可选 ROS 节点方式接入。

# 面向家庭服务机器人的主人识别与人机协同控制技术方案

## 1. 文档目的

本文档用于将当前 `wpr_simulation` + `wpr_task1_owner_search` 仿真工程整理为可扩展的科研实验方案，目标是支撑一篇面向 EI 会议投稿的系统类论文。论文主题建议围绕“人机协同控制”和“具身智能控制”，将现有比赛任务从工程实现提升为具备实验对比、性能指标和方法贡献的研究工作。

建议论文题目方向：

1. **基于视觉-点云融合的具身服务机器人主人识别与人机协同控制方法**
2. **面向家庭场景的服务机器人主人意图感知与安全靠近控制研究**
3. **Identity-aware Human-Robot Collaborative Control for Embodied Domestic Service Robots**

## 2. 当前工程基础

### 2.1 可复用工程模块

当前工程已经具备一个较完整的具身任务闭环，适合作为论文实验平台。

| 模块 | 当前实现 | 对论文的作用 |
| --- | --- | --- |
| 仿真环境 | `wpr_simulation` 中 four rooms / RoboCup Home 场景 | 提供可重复实验环境 |
| 导航 | AMCL + `move_base` + `wpb_home` 局部规划器 | 支撑从初始点到客厅航点的自主移动 |
| 人体检测 | YOLO-World，GPU 推理 | 检测场景中的人员候选目标 |
| 主人确认 | InsightFace 轻量人脸验证 | 区分主人和干扰人员 |
| 动作识别 | YOLO-Pose 关键点 + 规则判断；仿真 hint 可做真值辅助 | 判断摔倒、挥手、坐/躺等主人状态 |
| 三维定位 | RGB-D 点云投影到 `/perception/person_detections_3d` | 估计主人三维位置 |
| 安全靠近 | `move_base` 一段式导航 + 3D 点云精停 + 激光雷达前向安全约束 | 作为论文的主要控制实验模块 |
| 交互响应 | TTS 播报 + 仿真机械臂动作 | 体现人机协同响应 |

核心代码位置：

```text
/home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/scripts/task1_find_owner.py
/home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/config/task1_owner_search.yaml
/home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/launch/task1_four_rooms_owner_search.launch
```

### 2.2 当前任务流程

现有任务流程可以抽象为：

```text
进入客厅航点
  -> 原地扫描并检测人员
  -> 对人员候选进行主人身份确认
  -> 将主人居中到相机画面
  -> 识别主人动作/状态
  -> 生成靠近目标点并导航
  -> 基于3D点云和雷达进行末端精停
  -> 语音/机械臂执行协同响应
```

其中，论文最值得突出的部分不是单个检测模型，而是：

1. **身份约束**：机器人不是看到人就响应，而是确认“主人”。
2. **状态约束**：不同主人动作触发不同协同策略。
3. **空间约束**：靠近主人时兼顾点云定位精度和雷达安全约束。
4. **具身闭环**：感知结果会改变机器人底盘、语音和机械臂动作。

## 3. 研究问题定义

### 3.1 应用场景

家庭服务机器人进入客厅后，需要在可能存在一名干扰人员的情况下自主寻找主人，识别主人的动作状态，并根据不同状态完成协同行为：

| 主人状态 | 机器人响应 |
| --- | --- |
| 主人挥手 | 靠近主人并询问“请问您需要什么帮助？” |
| 主人坐/躺在床、沙发、椅子上 | 靠近主人并说“请指示。” |
| 主人突然摔倒 | 播报“主人突然摔倒。”，靠近后伸出机械臂模拟辅助 |
| 无法确认主人 | 继续扫描或提示未确认 |

### 3.2 科研问题

可将工程问题提升为以下科研问题：

1. **多人员干扰下的目标身份确认问题**：如何结合开放词汇人体检测和轻量人脸验证，降低把客人误认为主人的概率。
2. **主人状态驱动的人机协同决策问题**：如何将动作识别结果转化为机器人可执行的语音、移动和机械臂响应。
3. **人周围安全靠近控制问题**：如何利用 RGB-D 点云估计主人位置，并结合激光雷达限制前向运动，保证机器人停在主人面前而不发生碰撞。
4. **低算力平台实时部署问题**：如何在 RTX 3050 4GB 等有限 GPU 资源下保持实时检测、动作识别和控制闭环。

## 4. 相关工作整理

### 4.1 家庭服务机器人与 RoboCup@Home

RoboCup@Home 和家庭服务机器人研究通常强调机器人在真实或仿真家庭环境中完成导航、物体/人员识别、语音交互、任务执行等综合能力。这类研究说明，论文可以不局限于单一视觉模型，而应强调服务任务中的完整感知-决策-控制闭环。

可参考方向：

- RoboCup@Home benchmark and domestic service robot evaluation.
- Domestic service robot navigation and person-aware interaction.
- Service robot task execution under dynamic human presence.

### 4.2 人体识别与主人身份确认

YOLO-World 属于开放词汇目标检测方法，可以通过文本类别如 `person` 快速检测人员，适合部署到现有 ROS 相机话题中。主人确认可以使用 InsightFace 等轻量化人脸特征模型，在只提供一张主人照片的条件下进行身份验证。

该方向适合在论文中作为“身份约束感知模块”，重点不是发明新检测器，而是证明身份约束能显著降低干扰人员场景下的错误响应。

### 4.3 人体动作识别与意图理解

已有工作常使用 RGB-D、骨架关键点、YOLO-Pose、MediaPipe 或时序模型进行人体动作识别和跌倒检测。你的系统可以采用轻量 YOLO-Pose 关键点规则识别摔倒、挥手、坐/躺，并在仿真中使用 model state hint 生成自动标签或真值对照。

论文写作时应注意：

- 不建议把仿真 hint 作为最终算法主体。
- 仿真 hint 更适合作为自动标注、评测真值或上限对照。
- 真正可迁移到实体机器人的方法应是视觉关键点和深度点云。

### 4.4 人本导航与安全靠近控制

社会导航、人本导航和人员跟随研究通常关注机器人如何在人的周围保持合理距离、朝向和舒适空间。你的工程当前具备一段式导航、3D 点云位置估计和雷达前向保护，可以抽象为一种面向主人的安全靠近控制方法。

建议论文中的控制模块描述为：

```text
全局导航阶段：move_base 生成可通行路径
局部目标生成阶段：根据主人3D位置计算主人前方停靠点
末端精停阶段：3D点云闭环控制距离和朝向，激光雷达限制前方安全距离
```

## 5. 拟提出方法

### 5.1 系统架构

建议将系统架构划分为四层：

| 层级 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| 多模态感知层 | 检测人、确认主人、识别动作、估计三维位置 | RGB 图像、深度点云、激光雷达 | 主人身份、动作标签、3D 坐标 |
| 任务认知层 | 维护任务状态机和主人状态 | 感知结果、导航状态 | 当前任务阶段和响应策略 |
| 协同控制层 | 导航、扫描、居中、靠近、精停、机械臂动作 | 目标状态和控制参数 | `/cmd_vel`、`move_base` goal、机械臂控制 |
| 人机交互层 | 语音播报和询问 | 动作识别结果 | TTS 语音输出 |

### 5.2 主人识别模块

候选人员由 YOLO-World 检测得到，主人确认由 InsightFace 完成。

处理流程：

```text
RGB图像
  -> YOLO-World检测person候选框
  -> 按置信度和空间位置筛选候选
  -> 裁剪候选区域并进行人脸检测
  -> 与 owner.jpg 生成人脸特征相似度
  -> 相似度超过阈值则确认主人
```

建议论文指标：

- 主人确认成功率
- 客人误识别率
- 平均确认时间
- 不同距离/角度下的识别稳定性

### 5.3 动作状态识别模块

动作识别建议采用 YOLO-Pose 关键点时序规则，而不是使用大模型兜底。原因：

1. 轻量模型更适合 RTX 3050 4GB 实时部署。
2. 输出可解释，可以根据人体关键点几何关系写论文。
3. 可迁移到实体机器人，不依赖仿真内部信息。

支持动作：

| 动作 | 可用特征 |
| --- | --- |
| 挥手 | 手腕高于肩部，且多帧横向/纵向位移明显 |
| 摔倒 | 人体主轴接近水平，髋部/肩部高度接近地面，姿态突然变化 |
| 躺下 | 人体包围框宽高比变大，关键点整体接近水平 |
| 坐下 | 髋-膝-踝角度变化，人体高度降低但上半身仍近似竖直 |

建议将动作识别输出定义为：

```text
action_label in {waving, falling, lying, sitting, uncertain}
action_confidence in [0, 1]
action_reason: textual rule explanation
```

### 5.4 安全靠近控制模块

当前系统中靠近主人分为两段：

1. **导航靠近**：根据主人点云坐标生成 `move_base` 目标点，目标点位于主人前方约 `approach_standoff_distance` 米。
2. **末端精停**：到达后根据最新 3D 点云继续小速度微调，同时用激光雷达限制前方安全距离。

末端精停控制可抽象为：

```text
d = sqrt(x_person^2 + y_person^2)
theta = atan2(y_person, x_person)
e_d = d - d_target
e_theta = theta
```

线速度：

```text
v = clamp(k_d * e_d, -v_reverse_max, v_forward_max)
```

角速度：

```text
w = clamp(k_theta * e_theta, -w_max, w_max)
```

雷达安全约束：

```text
if front_lidar_distance <= lidar_stop_distance + margin and v > 0:
    v = 0
```

该模块是论文最适合体现“控制”的部分。可以重点证明融合 3D 点云和雷达约束后，机器人停靠误差和碰撞风险低于只用导航或只用二维图像的方法。

### 5.5 协同响应策略

识别主人动作后，任务状态机输出响应策略：

| 动作标签 | 机器人响应 |
| --- | --- |
| `falling` | 播报“主人突然摔倒。”，靠近后伸出机械臂模拟辅助 |
| `waving` | 靠近后询问“请问您需要什么帮助？” |
| `lying` / `sitting` | 靠近后播报“请指示。” |
| `uncertain` | 靠近或保持安全距离，并提示动作不确定 |

该部分可以写成“动作状态驱动的人机协同策略”。

## 6. 实验设计

### 6.1 实验平台

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Ubuntu 20.04 |
| ROS | Noetic |
| 仿真环境 | Gazebo + `wpr_simulation` four rooms |
| 机器人 | WPB Home / 带 RGB-D 相机、雷达、机械臂的仿真模型 |
| GPU | NVIDIA RTX 3050 Laptop GPU 4GB |
| 人体检测 | YOLO-World on `cuda:0` |
| 动作识别 | YOLO-Pose on `cuda:0` |
| 人脸确认 | InsightFace CPU |

### 6.2 实验场景变量

建议至少设计以下变量：

| 变量 | 取值 |
| --- | --- |
| 人员数量 | 仅主人；主人 + 1 名客人 |
| 主人动作 | 站立、挥手、坐下、躺下、摔倒 |
| 初始距离 | 近距离、中距离、远距离 |
| 初始角度 | 正前方、左侧、右侧 |
| 光照/遮挡 | 正常、轻微遮挡、侧脸 |

基础实验矩阵：

```text
5种动作 x 2种人员数量 x 3种距离 x 3种角度 = 90组实验
```

如果时间不足，可以先做最小版：

```text
4种动作 x 2种人员数量 x 3种距离 = 24组实验
每组重复3次，共72次实验
```

### 6.3 实验一：主人识别鲁棒性

目的：验证身份约束能否在干扰人员存在时减少误响应。

对比方法：

| 方法 | 描述 |
| --- | --- |
| M1 | 只使用 YOLO-World 检测 person，默认第一个人为主人 |
| M2 | YOLO-World + InsightFace 主人确认 |
| M3 | YOLO-World + InsightFace + 主动扫描策略 |

指标：

- 主人识别成功率
- 客人误识别率
- 平均确认时间
- 主人丢失后重新确认成功率

### 6.4 实验二：动作状态识别

目的：验证轻量 YOLO-Pose 规则能否在仿真家庭场景中完成动作状态识别。

对比方法：

| 方法 | 描述 |
| --- | --- |
| A1 | 单帧姿态规则 |
| A2 | 多帧时序姿态规则 |
| A3 | 多帧姿态规则 + 主人 ROI 锁定 |
| A4 | 仿真 hint 真值对照，不作为最终算法 |

指标：

- 动作分类准确率
- 摔倒召回率
- 挥手识别准确率
- 平均动作识别延迟
- 不确定样本比例

### 6.5 实验三：安全靠近控制

目的：验证 3D 点云 + 雷达辅助精停是否提升机器人停靠精度和安全性。

对比方法：

| 方法 | 描述 |
| --- | --- |
| C1 | 只用 `move_base` 导航到主人前方目标点 |
| C2 | `move_base` + 3D 点云末端精停 |
| C3 | `move_base` + 3D 点云末端精停 + 激光雷达安全约束 |

指标：

- 最终停靠距离误差
- 最终朝向误差
- 是否碰撞或距离过近
- 到达时间
- 规划失败率
- 精停阶段震荡次数

建议将理想停靠距离设为 `0.8m` 左右，并记录实际距离。

### 6.6 实验四：端到端任务成功率

目的：验证完整具身任务闭环是否稳定。

任务定义：

```text
机器人从初始点出发
  -> 导航到 living_room
  -> 找到并确认主人
  -> 识别主人动作
  -> 播报动作状态
  -> 靠近主人
  -> 执行动作对应的人机协同响应
```

指标：

- 端到端成功率
- 平均任务完成时间
- 各阶段失败率
- GPU 显存占用峰值
- 平均 YOLO 推理时间
- 平均动作识别时间

## 7. 数据记录建议

建议新增实验日志节点或在 `task1_find_owner.py` 中增加 CSV 记录，保存到：

```text
/home/ubuntu20/catkin_ws/src/wpr_task1_owner_search/experiments/logs/
```

建议每次实验记录字段：

```text
run_id
scenario_id
num_people
owner_action_gt
owner_action_pred
owner_identity_success
guest_false_accept
detect_time_sec
verify_time_sec
action_time_sec
approach_time_sec
final_distance_m
final_bearing_rad
lidar_min_distance_m
collision_flag
task_success
failure_reason
gpu_memory_mb
```

建议同时保存 ROS bag：

```text
/tf
/odom
/scan
/cmd_vel
/move_base/status
/kinect2/qhd/image_color_rect
/kinect2/sd/points
/perception/person_detections_3d
```

## 8. 消融实验设计

为了增强论文说服力，建议做以下消融：

| 消融项 | 目的 |
| --- | --- |
| 去掉 InsightFace | 证明主人身份确认对抗干扰人员有效 |
| 去掉主动扫描 | 证明具身主动观察比固定视角更稳定 |
| 去掉主人 ROI | 证明 ROI 锁定能减少客人动作干扰 |
| 去掉 3D 点云精停 | 证明仅导航无法稳定停在合适距离 |
| 去掉雷达安全约束 | 证明雷达能降低近距离碰撞风险 |
| 去掉动作状态策略 | 证明不同动作响应比统一响应更符合人机协同 |

## 9. 预期创新点

建议论文中使用以下创新点表述：

1. **提出一种身份约束的家庭服务机器人协同任务框架**：通过开放词汇人体检测和轻量人脸验证，使机器人在多人员干扰环境中优先响应主人。
2. **提出一种动作状态驱动的人机协同响应策略**：将主人挥手、坐/躺、摔倒等动作状态映射到底盘、语音和机械臂协同动作。
3. **设计一种视觉-点云-雷达融合的安全靠近控制方法**：通过 RGB-D 点云估计主人三维位置，结合激光雷达前向安全约束实现稳定停靠。
4. **完成低算力 GPU 平台上的实时具身闭环部署**：在 RTX 3050 4GB 条件下实现检测、识别、导航和交互响应。

## 10. 论文结构建议

### 摘要

说明家庭服务机器人在多人员场景中需要识别主人、理解动作状态并执行安全协同响应。提出一个多模态感知与协同控制框架，并在 Gazebo 家庭仿真环境中验证。

### 第1章 引言

介绍家庭服务机器人、人机协同、具身智能控制的背景；指出单纯视觉识别不能完成真实服务任务，需要与机器人移动、交互和安全控制结合。

### 第2章 相关工作

分为：家庭服务机器人、人体/主人识别、动作识别、人本导航与靠近控制。

### 第3章 系统方法

介绍四层架构、主人识别、动作识别、任务状态机和安全靠近控制。

### 第4章 实验设计

介绍仿真环境、硬件平台、实验变量、对比方法和评价指标。

### 第5章 实验结果与分析

展示主人识别、动作识别、靠近控制和端到端任务结果。重点放在消融实验表格和停靠误差曲线。

### 第6章 结论

总结方法有效性，并说明后续会迁移到实体机器人，进一步测试真实家庭环境中的鲁棒性。

## 11. 后续工程改造清单

为了从比赛工程转为科研实验工程，建议做以下改造：

1. 新增实验配置文件，例如：

```text
config/experiment_identity.yaml
config/experiment_action.yaml
config/experiment_approach.yaml
config/experiment_end_to_end.yaml
```

2. 新增实验日志脚本：

```text
scripts/experiment_logger.py
```

3. 新增批量实验启动脚本：

```text
tools/run_experiment_batch.sh
```

4. 新增结果分析脚本：

```text
tools/analyze_experiment_results.py
```

5. 将仿真 hint 从“动作识别算法”中剥离，改为“真值标签/评测辅助”。

6. 增加参数开关用于消融实验：

```yaml
use_identity_verification: true
use_owner_roi_for_action: true
use_3d_fine_stop: true
use_lidar_safety_guard: true
use_action_conditioned_response: true
```

7. 每次实验自动输出统一 CSV，避免只依赖终端日志。

## 12. 可参考文献与对标方向

后续写论文时，建议不要只引用 YOLO 或人脸识别论文，而要按“家庭服务机器人基准、HRI动作理解、人本导航、实时视觉模型”四条线组织相关工作。

### 12.1 家庭服务机器人与 RoboCup@Home

1. Wachsmuth, S., Holz, D., Rudinac, M., & Ruiz-del-Solar, J. **RoboCup@Home -- Benchmarking Domestic Service Robots**. AAAI Conference on Artificial Intelligence, 2015. DOI: `10.1609/aaai.v29i1.9350`.
   - 可用于说明 RoboCup@Home 强调家庭服务机器人在真实家庭任务中的导航、HRI、视觉识别、行为集成与系统评测。
2. RoboCup@Home official benchmark description.
   - 可用于说明家庭服务机器人竞赛关注 Human-Robot Interaction and Cooperation、Navigation and Mapping、Computer Vision、Object Manipulation、Adaptive Behaviors 等能力。

### 12.2 开放词汇人体检测与主人识别

3. Cheng, T., Song, L., Ge, Y., Liu, W., Wang, X., & Shan, Y. **YOLO-World: Real-Time Open-Vocabulary Object Detection**. CVPR, 2024.
   - 可用于支撑本文使用 YOLO-World 作为开放词汇实时人体检测模块。
   - 论文中可强调：本文没有提出新检测网络，而是将开放词汇检测嵌入具身服务机器人闭环。
4. Deng, J., Guo, J., Xue, N., & Zafeiriou, S. **ArcFace: Additive Angular Margin Loss for Deep Face Recognition**. CVPR, 2019.
   - 可用于支撑 InsightFace / ArcFace 类人脸特征验证方法。
   - 论文中可强调：只需主人单张参考照片即可完成轻量身份确认。

### 12.3 HRI动作理解与机器人响应

5. Li, K., Wu, J., Zhao, X., & Tan, M. **Real-Time Human-Robot Interaction for a Service Robot Based on 3D Human Activity Recognition and Human-mimicking Decision Mechanism**. arXiv, 2018.
   - 该工作将 Kinect 3D skeleton、LSTM 动作识别、机器人底盘/机械臂响应结合，和本文“识别主人动作后执行协同响应”的思路最接近。
   - 本文差异：增加主人身份约束、多人员干扰、RGB-D 点云安全靠近控制。
6. ROS4HRI / REP-155 related documents.
   - 可用于说明 ROS 社区正在标准化人脸、人体、骨架、语音等 HRI 表示。
   - 后续如果要把系统做得更规范，可以将主人、人体关键点和动作状态封装成 ROS4HRI 风格消息。

### 12.4 人本导航与社会导航

7. Moller, R., Furnari, A., Battiato, S., Harma, A., & Farinella, G. M. **A Survey on Human-Aware Robot Navigation**. Robotics and Autonomous Systems, 2021.
   - 可用于说明人本导航涉及主动视觉、动作识别、机器人导航和 HRI。
   - 本文对应点：机器人靠近主人时需要同时考虑人的位置、朝向和安全距离。
8. **Bridging Requirements, Planning, and Evaluation: A Review of Social Robot Navigation**. Sensors, 2024.
   - 可用于组织“安全距离、社会接受度、评价指标”的讨论。
   - 本文可将最终停靠距离误差、朝向误差、碰撞/过近次数作为面向人的导航指标。
9. **Social Robot Navigation: A Review and Benchmarking of Learning-Based Methods**. 2026.
   - 可作为近年社会导航综述参考，说明服务机器人在有人的环境中需要结合 RGB-D / LiDAR 等多模态感知。

### 12.5 跌倒检测与轻量姿态模型

10. **Real-Time Human Fall Detection From Video Using YOLOv11 With Pose Estimation: A Paradigm Shift Toward Efficient Transformer-Based Architectures**. IEEE Access, 2026. DOI: `10.1109/ACCESS.2026.3674843`.
    - 可用于支撑 YOLO-Pose 类方法适合实时跌倒检测。
    - 本文差异：不只做视频告警，还驱动机器人靠近和机械臂辅助动作。
11. **LFD-YOLO: A Lightweight Fall Detection Network with Enhanced Feature Extraction and Fusion**. Scientific Reports, 2025.
    - 可用于支撑轻量视觉模型在跌倒检测中的实时性和部署价值。
    - 本文可强调采用轻量模型而不是大模型，是为了保证机器人实时控制闭环。

### 12.6 相关工作总结写法

论文中可以这样总结差异：

```text
Existing work has studied domestic service robot benchmarks, open-vocabulary object detection,
face recognition, pose-based action recognition, and human-aware navigation separately.
However, fewer studies integrate owner identity verification, action-conditioned response,
and RGB-D/LiDAR-based safe approach control into a complete embodied domestic service task.
This paper focuses on this integrated closed loop and evaluates it in a repeatable home-like
simulation environment.
```

## 13. 推荐投稿定位

更适合投稿的 EI 会议方向：

| 会议主题关键词 | 本工程对应内容 |
| --- | --- |
| Human-Robot Interaction | 主人识别、动作理解、语音响应 |
| Human-Robot Collaborative Control | 根据主人状态调整靠近、询问和机械臂动作 |
| Embodied Intelligent Control | 感知结果驱动机器人移动和交互动作 |
| Service Robot | 家庭客厅服务任务 |
| Multimodal Perception and Control | RGB、深度点云、雷达、语音融合 |
| Autonomous Navigation | 到客厅航点、主动扫描、安全靠近 |

建议摘要和标题中尽量出现以下关键词：

```text
embodied service robot
human-robot collaborative control
identity-aware perception
action-conditioned response
RGB-D and LiDAR fusion
safe approach control
domestic environment
```

## 14. 最小可发表版本

如果时间有限，建议先完成以下最小论文版本：

1. 只做仿真，不做实体机器人。
2. 只研究四类动作：挥手、坐/躺、摔倒、站立。
3. 实验重点放在三个问题：
   - 主人识别是否能抗客人干扰；
   - 动作状态是否能触发正确响应；
   - 3D 点云 + 雷达精停是否比纯导航更安全、更准确。
4. 至少完成 72 次实验，形成统计表格。
5. 论文贡献写成系统框架和控制方法，不声称提出全新神经网络。

这样更稳，也更符合当前工程已经实现的能力边界。

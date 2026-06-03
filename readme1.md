# Isaac Sim + ROS2 + ZMQ + SmolVLA 工程记录

本文档记录当前 `isaacsim_vla_ws` 工作区的整体流程、环境配置、启动方法、当前代码状态、已知问题和后续想法。目的是让后来接手的人能快速知道：这个工程每一层在干什么，怎么启动，为什么现在这样配置，以及下一步应该排查什么。

## 1. 整体流程框架

当前工程目标是在 Isaac Sim 里运行 SO100 机械臂场景，通过视觉语言动作模型 SmolVLA 根据图像和任务文字输出动作，让机械臂执行类似 “pick up the red cube” 的任务。

整体链路如下：

```text
Isaac Sim 场景
  |
  |  ROS2 topic: 相机图像、关节状态、动作命令
  v
isaacsim_vla_ws / vla_center
  |
  |  ZMQ 5556: observation npz, 包含 joints + camera jpeg
  |  ZMQ 5555: action float32[6]
  v
tools/smolvla_zmq_official.py
  |
  |  LeRobot / SmolVLA
  v
lerobot/smolvla_base policy
```

各部分任务如下。

### Isaac Sim

Isaac Sim 负责仿真世界：

- 加载 SO100 机械臂。
- 加载桌面、红色方块、相机。
- 发布相机图像和关节状态。
- 接收动作目标并驱动机械臂。

当前使用的是 `MyLovelyAxe/isaacsim_vla_ws` 仓库中的 Isaac Sim 场景，而不是 Isaac 官方 Franka 示例。

### ROS2

ROS2 是仿真内部的机器人通信层。

它负责：

- 在 Isaac Sim 和 `vla_center` 节点之间传递图像、关节状态、动作命令。
- 让相机、joint state、action controller 这些模块保持机器人系统式的解耦。
- 方便用 `rviz2`、`ros2 topic list`、`ros2 topic echo` 等工具检查观测。

当前 ROS2 launch：

```bash
ros2 launch vla_center send_obs_get_act.launch.py
```

这个 launch 会启动：

- `send_observation_3cam`：从 ROS2 收 3 路相机和 joint state，打包为 ZMQ observation。
- `get_action`：从 ZMQ 收 action，然后发布回 ROS2/Isaac 控制机械臂。

### ZMQ

ZMQ 是 ROS2 和 VLA Python 推理程序之间的轻量 socket 桥。

它负责：

- `tcp://127.0.0.1:5556`：ROS2 bridge -> VLA adapter，发送 observation。
- `tcp://127.0.0.1:5555`：VLA adapter -> ROS2 bridge，发送 action。

为什么用了 ROS2 还要用 ZMQ：

- ROS2 适合机器人系统内部通信，强类型 topic，方便 Isaac Sim 和 RViz 集成。
- VLA 推理代码通常运行在独立 Python/conda 环境里，依赖 PyTorch、Transformers、LeRobot，不一定适合直接塞进 ROS2 workspace。
- ZMQ 让模型推理程序不用依赖 ROS2 Python 环境，可以单独用 conda `smolvla` 跑。
- ZMQ payload 可以很自由地塞 `npz`、JPEG bytes、float32 数组，适合快速实验。

所以当前架构不是 “ROS2 和 ZMQ 重复”，而是：

```text
ROS2: 仿真/机器人侧通信
ZMQ: ROS2 机器人侧和 VLA 模型侧之间的轻量桥
```

后续如果要工程化，可以把 SmolVLA 直接封装为 ROS2 node，去掉 ZMQ。但目前 ZMQ 更方便快速调模型。

### SmolVLA adapter

文件：

```text
/home/dreamland/isaacsim_vla_ws/tools/smolvla_zmq_official.py
```

它负责：

- 从 ZMQ 5556 接收 observation。
- 解码 joints 和 3 路 JPEG 图像。
- 按 camera-order 把图像映射成 SmolVLA 的 `camera1/camera2/camera3`。
- 把 Isaac joint radian 映射成 LeRobot SO100 degree state。
- 调用官方 LeRobot `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`。
- 把 SmolVLA 输出的 SO100 degree action 映射回 Isaac radian action。
- 可选择 dry-run，只打印不发布动作。
- 可选择 fake/scripted action，用来验证控制链路。

## 2. 整体配置

### 机器与硬件

当前机器从 Isaac Sim overlay 可见：

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Isaac Sim 中大约 26-30 FPS
- SmolVLA 推理使用 CUDA

### 主要目录

```text
/isaacsim
```

Isaac Sim 安装目录。

```text
/home/dreamland/isaacsim_vla_ws
```

VLA + Isaac Sim + ROS2 工作区，克隆自 `MyLovelyAxe/isaacsim_vla_ws`，并做了本地修改。

```text
/home/dreamland/lerobot_official
```

官方 LeRobot 仓库，安装为 editable package，用于 SmolVLA。

```text
/smolvla/hf_cache
```

Hugging Face 模型缓存。

```text
/smolvla/outputs
```

预留给后续微调、评估、记录输出。

### Conda 环境

SmolVLA 单独使用 conda 环境：

```bash
conda activate smolvla
```

安装位置：

```text
/home/dreamland/miniconda3/envs/smolvla
```

LeRobot 安装方式：

```bash
cd /home/dreamland/lerobot_official
pip install -e ".[smolvla]"
```

常用环境变量：

```bash
export HF_HOME=/smolvla/hf_cache
export HF_HUB_CACHE=/smolvla/hf_cache/hub
export HF_HUB_DISABLE_XET=1
unset HF_ENDPOINT
export PYTHONUNBUFFERED=1
```

说明：

- `HF_HOME` / `HF_HUB_CACHE` 把模型缓存放到 `/smolvla`。
- `HF_HUB_DISABLE_XET=1` 用于避免 Hugging Face Xet 下载路径导致的问题。
- `unset HF_ENDPOINT` 表示使用官方 Hugging Face endpoint；如果网络慢，可以临时换镜像，但之前镜像曾出现 safetensors 文件识别问题。
- `PYTHONUNBUFFERED=1` 让模型加载和推理日志实时打印。

## 3. 当前代码状态

### 已修改/新增点

#### 1. Isaac Sim 启动脚本

`bash/start_isaac_sim.sh` 已调整为本机 `/isaacsim` 路径。

#### 2. ROS2 observation 改为 3 相机

当前 launch 使用：

```text
src/vla_center/vla_center/send_observation_3cam.py
```

而不是原来的 2 camera sender。

入口在：

```text
src/vla_center/setup.py
src/vla_center/launch/send_obs_get_act.launch.py
```

#### 3. SmolVLA ZMQ adapter

文件：

```text
tools/smolvla_zmq_official.py
```

当前支持：

- `--policy fake`
- `--policy smolvla`
- `--dry-run`
- `--compare-cube-hover`
- `--camera-order`
- `--smolvla-state-map`
- `--smolvla-action-map`
- `--smolvla-sim-joint-signs`
- `--smolvla-state-offsets-deg`
- `--smolvla-action-offsets-deg`
- scripted fake action:
  - `home`
  - `test`
  - `mirror`
  - `alternate`
  - `reach_forward_grasp_return`
  - `cube_hover_grasp_return`
  - `cube_left_hover_grasp_return`

### 当前最佳 SmolVLA 映射配置

当前最合理的配置是：

```bash
--smolvla-stats-key so100
--smolvla-sim-joint-signs=-1,1,-1,1,1,1
--smolvla-state-offsets-deg 1.65,118.46,109.77,56.70,62.58,12.0
--smolvla-action-offsets-deg 1.65,118.46,109.77,56.70,62.58,12.0
--camera-order 1,3,2
```

这些 offset 不是随机调参。官方 `lerobot/smolvla_base` 的 SO100 action stats 中：

```text
so100.buffer.action.mean =
[1.6, 119.9, 109.8, 56.7, -27.4, 12.0]
```

也就是说 LeRobot SO100 数据里的 “常见/home” 关节值并不是 `[0,0,0,0,0,0]`，而是经过真实硬件 calibration 后的一组 degree 位置。Isaac reset 如果直接按 rad->degree 映射，会得到接近：

```text
[0, 1.5, 0, 0, -90, 0]
```

这和官方训练分布明显不一致。所以需要 offset 把 Isaac reset 对齐到 LeRobot SO100 convention。

### 当前参考方块上方姿态

用 scripted action 标定出的 “大致在红块上方” Isaac action：

```text
CUBE_LEFT_HOVER_ACTION =
[-0.35, 1.10, -1.35, 0.45, 0.0, -0.0065]
```

这个不是最终策略，只是用于检查 SmolVLA 输出是否朝合理方向走。

## 4. 详细启动流程

建议启动顺序固定为：

```text
Terminal 1: Isaac Sim
Terminal 2: ROS2 bridge
Terminal 3: SmolVLA / fake policy
```

### Terminal 1: 启动 Isaac Sim

```bash
cd /home/dreamland/isaacsim_vla_ws/bash
source setup_isaacsim.sh
./start_isaac_sim.sh
```

启动后：

1. 等 Isaac Sim 打开场景。
2. 点击 Play。
3. 红色方块会落下。
4. SO100 机械臂应处于 reset/home 姿态。

如果要复位：

- 在 Isaac Sim UI 中 Stop。
- 再点击 Play。
- 必要时重新打开 stage/重启 Terminal 1。

### Terminal 2: 启动 ROS2 bridge

```bash
cd /home/dreamland/isaacsim_vla_ws
source bash/setup_systemros.sh
source install/setup.bash
ros2 launch vla_center send_obs_get_act.launch.py
```

正常输出应包含类似：

```text
[get_action]: Ready to receive action commands ......
[send_observation_3cam]: Start to send image and joint states ......
```

如果只想看图像，可以另开 RViz，但不是必须。RViz 只是观测可视化工具，不参与控制链路。

### Terminal 3A: dry-run 诊断，不控制机械臂

这个命令只打印 SmolVLA 输出，不发布 action，适合排查模型是否看图、输出是否合理：

```bash
source /home/dreamland/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

export HF_HOME=/smolvla/hf_cache
export HF_HUB_CACHE=/smolvla/hf_cache/hub
export HF_HUB_DISABLE_XET=1
unset HF_ENDPOINT
export PYTHONUNBUFFERED=1

/home/dreamland/miniconda3/envs/smolvla/bin/python \
  /home/dreamland/isaacsim_vla_ws/tools/smolvla_zmq_official.py \
  --policy smolvla \
  --model-id lerobot/smolvla_base \
  --task "pick up the red cube" \
  --robot-type so100_follower \
  --smolvla-state-map sim-rad-to-so100-deg \
  --smolvla-action-map so100-deg-to-sim-rad \
  --smolvla-stats-key so100 \
  --smolvla-replan-interval 5 \
  --smolvla-gripper-scale 0.01 \
  --smolvla-sim-joint-signs=-1,1,-1,1,1,1 \
  --smolvla-state-offsets-deg 1.65,118.46,109.77,56.70,62.58,12.0 \
  --smolvla-action-offsets-deg 1.65,118.46,109.77,56.70,62.58,12.0 \
  --camera-order 1,3,2 \
  --action-mode absolute \
  --max-delta 0 \
  --action-clip 0 \
  --rate-hz 1 \
  --log-interval-s 0 \
  --max-steps 3 \
  --dry-run \
  --compare-cube-hover \
  --save-debug-dir /tmp/vla_debug_smolvla_dryrun \
  --save-debug-every 1
```

如果正常，会打印：

```text
SmolVLA loaded.
Observation SUB connected to tcp://127.0.0.1:5556
Action PUB bound to tcp://127.0.0.1:5555
Policy: smolvla
dry-run action step=...
```

### Terminal 3B: 真正发布 SmolVLA action

谨慎使用。建议先 dry-run 看输出合理，再跑这个。

```bash
source /home/dreamland/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

export HF_HOME=/smolvla/hf_cache
export HF_HUB_CACHE=/smolvla/hf_cache/hub
export HF_HUB_DISABLE_XET=1
unset HF_ENDPOINT
export PYTHONUNBUFFERED=1

/home/dreamland/miniconda3/envs/smolvla/bin/python \
  /home/dreamland/isaacsim_vla_ws/tools/smolvla_zmq_official.py \
  --policy smolvla \
  --model-id lerobot/smolvla_base \
  --task "pick up the red cube" \
  --robot-type so100_follower \
  --smolvla-state-map sim-rad-to-so100-deg \
  --smolvla-action-map so100-deg-to-sim-rad \
  --smolvla-stats-key so100 \
  --smolvla-replan-interval 5 \
  --smolvla-gripper-scale 0.01 \
  --smolvla-sim-joint-signs=-1,1,-1,1,1,1 \
  --smolvla-state-offsets-deg 1.65,118.46,109.77,56.70,62.58,12.0 \
  --smolvla-action-offsets-deg 1.65,118.46,109.77,56.70,62.58,12.0 \
  --camera-order 1,3,2 \
  --action-mode absolute \
  --max-delta 0.06 \
  --action-clip 0 \
  --rate-hz 2 \
  --log-interval-s 0.5 \
  --save-debug-dir /tmp/vla_debug_smolvla_run \
  --save-debug-every 20
```

说明：

- `--smolvla-replan-interval 5` 表示 SmolVLA 仍生成 action chunk，但只执行前 5 步就重新看图规划，避免默认 50 步开环执行太久。
- `--max-delta 0.06` 用于限速，避免每一步目标跳太大。
- `--rate-hz 2` 暂时保守；可以后续试 5 或 10。
- 如果出现明显乱动，按 Ctrl+C 停掉 Terminal 3。

### Terminal 3C: scripted action 验证控制链路

这个不经过 SmolVLA，用固定动作验证 Isaac/ROS/ZMQ/control 是否正常。

方块上方空夹并返回：

```bash
source /home/dreamland/miniconda3/etc/profile.d/conda.sh
conda activate smolvla

/home/dreamland/miniconda3/envs/smolvla/bin/python \
  /home/dreamland/isaacsim_vla_ws/tools/smolvla_zmq_official.py \
  --policy fake \
  --fake-action cube_left_hover_grasp_return \
  --action-mode absolute \
  --max-delta 0.06 \
  --action-clip 0 \
  --rate-hz 10 \
  --log-interval-s 0.5 \
  --max-steps 210 \
  --save-debug-dir /tmp/vla_debug_script_cube_left \
  --save-debug-every 20
```

预期：

1. 机械臂从当前姿态移动到红块上方附近。
2. 夹爪在空中闭合。
3. 返回 home。

这个脚本主要用来确认后半段控制链路没问题。

### 快速复位方法

原仓库提供了一个直接把 SO100 发布回初始关节状态的脚本：

```bash
cd /home/dreamland/isaacsim_vla_ws/bash
source setup_systemros.sh
./reset_joint_states.sh
```

它本质上是向 ROS2 topic `/joint_command` 发布一次 `sensor_msgs/msg/JointState`：

```text
name:
  ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
position:
  [0.0009, 0.0258, 0.0, 0.0, 0.0, -0.0065]
```

使用条件：

- Isaac Sim 已经启动并点击 Play。
- ROS2 环境已经 source。
- `/joint_command` 对应的 Isaac action graph/controller 正在工作。

这个脚本只复位机械臂关节目标，不会重置整个物理世界。如果红块已经被碰飞、掉落位置改变，还是需要在 Isaac Sim 里 Stop/Play 或重新打开场景。

完整世界复位的稳妥流程：

1. 停掉 Terminal 3。
2. Isaac Sim 里 Stop。
3. Isaac Sim 里 Play。
4. 必要时重启 Terminal 2。
5. 再跑 Terminal 3。

如果不想用 ROS2 脚本，也可以用 adapter 的 fake home 作为备用：

```bash
/home/dreamland/miniconda3/envs/smolvla/bin/python \
  /home/dreamland/isaacsim_vla_ws/tools/smolvla_zmq_official.py \
  --policy fake \
  --fake-action home \
  --action-mode absolute \
  --max-delta 0.06 \
  --action-clip 0 \
  --rate-hz 10 \
  --max-steps 80
```

## 5. 当前问题、可能原因、后续想法

### 当前效果

已经确认：

- ROS2/ZMQ/Isaac 后半段能执行正确关节目标。
- scripted action 能移动到红块上方附近、闭合夹爪、返回。
- SmolVLA 能加载、能接收图像和 state、能输出 6 维 action。
- 通过完整 joint offset 和 camera order 调整后，SmolVLA 输出比之前更接近方块上方参考姿态。

但还没有确认：

- SmolVLA 能稳定完成真实抓取。
- 当前 zero-shot 能否直接 pick up red cube。

### 已排查问题

#### 1. Action 语义

官方 SO100 follower action 是绝对关节目标，单位 degree，不是 delta，也不是末端位姿。

#### 2. Joint offset

Isaac reset 的 joint radian 不能直接当作 LeRobot SO100 的 0 度。LeRobot SO100 使用硬件 calibration 后的 degree convention。当前通过 offset 对齐到官方 `so100.buffer.action.mean` 附近。

#### 3. Camera order

通过同一帧 observation 扫描，当前最优是：

```text
camera-order 1,3,2
```

原来的 `1,2,3` 不是当前最优。

### 仍可能存在的问题

#### 1. Offset 还只是近似

当前 offset 是用官方 stats mean 作为 home/reference 推出来的，不是真正来自 SO100 calibration 文件。更严谨的方法是：

- 找到官方数据/真实 SO100 calibration 的具体 joint zero convention。
- 用 Isaac SO100 URDF/USD 的 joint zero 和真实 SO100 的 calibrated degree 做对应。
- 用 FK/末端位置验证每个 joint 的 sign 和 offset。

#### 2. SmolVLA 可能不适配当前 Isaac 视觉分布

即使动作空间对齐，模型也可能没见过这种画面：

- Isaac 灰色桌面和背景。
- 黄色 SO100 模型。
- 红色方块位置/尺度。
- 相机视角和真实数据不同。

这可能需要：

- 调整相机到更接近训练数据。
- 收集少量 Isaac 数据微调。
- 或先做 imitation / scripted data，再 fine-tune SmolVLA。

#### 3. 任务文字可能影响输出

当前用：

```text
pick up the red cube
```

后续可以 dry-run 对比：

```text
pick up the cube
grab the red cube
pick up the red block
pick up the object
```

#### 4. 控制频率和 action chunk

SmolVLA 内部是 action chunk policy，当前 adapter 每次只取一个 action 并循环调用。后续可以确认：

- 是否应该一次输出 chunk 后按 chunk 执行。
- 是否应使用 policy 内部 queue 的节奏。
- `rate-hz` 应该是多少。
- `max-delta` 是否过小导致动作迟缓。

#### 5. Gripper 标定

当前 gripper scale 用：

```text
--smolvla-gripper-scale 0.01
```

脚本动作证明 gripper 能闭合，但 SmolVLA 输出的 gripper 值是否和 Isaac 夹爪真实开合范围完全一致，还需要单独验证。

### 建议下一步

优先级从高到低：

1. 用当前最佳配置跑一次真实发布动作，观察是否比之前明显更靠近红块。
2. 如果动作方向对但慢，调 `rate-hz` 和 `max-delta`。
3. 如果到方块上方但不下降/不夹，继续调 gripper scale 和 action execution。
4. 系统扫 task 文本，dry-run 比较和参考姿态的差距。
5. 更严谨地做 SO100 Isaac joint 到 LeRobot calibrated degree 的映射表。
6. 如果以上都不能稳定抓取，再进入数据采集和微调。

## 6. 当前结论

当前不是简单的 “没微调所以不行”。

已经发现并修正了两个重要工程问题：

1. Isaac joint zero 和 LeRobot SO100 degree convention 不一致，需要 offset。
2. 三路相机顺序原来不是最优，当前更合理的是 `1,3,2`。

现在应该先用修正后的配置实测 VLA 控制效果。只有在 action/state/camera 都确认对齐后，仍然无法完成任务，才应该把主要精力转向微调和数据采集。

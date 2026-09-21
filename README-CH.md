# Evo-RL

**SJTU & Evo-Tech**

## 🎯 Evo-RL 项目重点

* **在两个平台上开放真实世界 RL：** 在 SO101 和 AgileX（PiPER/PiPER-X）两个平台上构建并发布完整的真实世界强化学习（Real-World RL）流程。
* **开放代码、模型和数据集以实现可复现：** 持续发布可以直接运行的 Offline RL 资源，使更多人能够复现实验结果，并将其应用于真实世界任务。
* **开放算法与社区共同演进：** 复现已有的真实世界 RL 方法，提出新的方法，并持续发布数据和 Benchmark，推动协作式开源社区的发展。

---

# 🚀 最新动态

* **[2026-06-30]** 添加了详细的 Hugging Face RW-RL Dataset 数据集卡片，包括可视化示例、发布统计、文件结构、数据模态以及下载说明。
* **[2026-03-07]** 增加 AgileX（PiPER/PiPER-X）真实世界 RL 支持。
* **[2026-02-26]** 首个 SO101 真实世界 RL Baseline 以及可复现的 CLI 工作流正式发布。

---

# 🧭 目录

| 入门     | 训练流程                | 项目信息               |
| ------ | ------------------- | ------------------ |
| ⚡ 快速开始 | 4）Value Function 训练 | Model & Dataset    |
| 1）安装   | 5）Value 推理          | Community Channels |
| 2）硬件设置 | 6）Policy 训练         | Affiliations       |
| 3）数据采集 | 7）闭环 Rollout 与下一轮训练 | Citation / License |

---

# ⚡ Quick Start 快速开始

Evo-RL 以 **LeRobot** 作为代码库基础，因为 LeRobot 的推理和数据采集逻辑与真实世界 RL 工作流高度一致。

---

# 1）Installation 安装

```bash
git clone https://github.com/MINT-SJTU/Evo-RL.git

cd Evo-RL

conda create -y -n evo-rl python=3.10

conda activate evo-rl

pip install -e .
```

安装配置和不同机器人平台所需的依赖，请参考官方 LeRobot 配置指南。

---

# 2）Hardware Setup 硬件设置

## SO 系列（SO100/SO101）

SO 系列机器人请按照官方教程完成详细的安装和配置步骤，然后再继续后面的流程。

下面的示例均以 **SO101** 作为参考配置。

### 设备路径推荐

推荐使用以下设备路径：

* **机器人串口：** 使用 `/dev/serial/by-id/`，该路径在重启后仍然保持稳定。
* **摄像头：** 优先使用 `/dev/v4l/by-id/`；如果不同设备的 ID 不唯一，则使用 `/dev/v4l/by-path/`。
* 后续示例中：

  * 机器人端口使用 `by-id`
  * 摄像头路径使用 `by-path`

可以使用下面的命令查看当前可用的稳定设备路径：

```bash
ls -l /dev/serial/by-id/
ls -l /dev/v4l/by-id/
ls -l /dev/v4l/by-path/
```

---

## 单臂 SO101

对于单臂用户，不需要进行太大的修改。

完成硬件配置之后，运行下面的命令检查系统是否已经准备好：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<SO101_FOLLOWER_PORT> \
  --robot.id=my_so101_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<SO101_LEADER_PORT> \
  --teleop.id=my_so101_leader
```

---

## 双臂 SO101

对于双臂用户，项目建议：

**将左侧 Leader 和左侧 Follower 机械臂中对应 Servo 4/5/6 的机械部件进行镜像安装。**

这样通常能够获得更加自然的双臂操作体验。

在运行双臂命令之前，需要确认标定文件已经存在于：

```text
~/.cache/huggingface/lerobot/calibration/
```

目录结构类似：

```text
calibration/
├── robots
│   └── so_follower
│       ├── bi_so101_follower_left.json
│       └── bi_so101_follower_right.json
└── teleoperators
    └── so_leader
        ├── bi_so101_leader_left.json
        └── bi_so101_leader_right.json
```

这个目录结构与单臂配置略有不同。

然后运行下面的命令验证双臂配置：

```bash
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=bi_so101_follower \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=bi_so101_leader
```

---

# 摄像头配置

开始数据采集之前，首先要验证摄像头的映射关系。

检查每个摄像头是否支持目标配置，例如：

```text
640 × 480 @ 30 FPS
```

可以运行：

```bash
v4l2-ctl -d /dev/v4l/by-path/<CAM_PATH> --list-formats-ext
```

### 单臂摄像头检查示例

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<SO101_FOLLOWER_PORT> \
  --robot.id=my_so101_follower \
  --robot.cameras='{ front: {type: opencv, index_or_path: "/dev/v4l/by-path/<FRONT_CAM>", width: 640, height: 480, fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<SO101_LEADER_PORT> \
  --teleop.id=my_so101_leader \
  --display_data=true
```

### 双臂摄像头检查示例

```bash
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=my_bi_so101_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<LEFT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<RIGHT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: "<REALSENSE_SN>", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=my_bi_so101_leader \
  --display_data=true
```

对于双臂摄像头映射：

`front` 摄像头放在左臂或右臂的 camera config 下都可以。

如果使用更多摄像头视角，也可以将它们放到左臂或右臂的 camera config 下。

如果只是进行初步调试，也可以暂时使用：

```text
/dev/ttyACM*
/dev/video*
```

这类临时设备路径。

---

# AgileX（PiPER/PiPER-X）

PiPER 使用 USB-CAN 适配器的稳定：

```text
ID_SERIAL_SHORT
```

而不是临时的 Linux：

```text
canN
```

设备名称。

查看已经连接的 USB-CAN 适配器：

```bash
for device in /sys/class/net/*; do
  [ "$(cat "$device/type" 2>/dev/null)" = "280" ] || continue
  serial=$(udevadm info --query=property --path="$device" | sed -n 's/^ID_SERIAL_SHORT=//p')
  echo "$(basename "$device"): $serial"
done
```

然后配置本次运行所需要的 USB-CAN 适配器：

```bash
lerobot-setup-can --mode=setup \
  --usb_can_serials=<USB_CAN_SERIAL_1>,<USB_CAN_SERIAL_2>,<USB_CAN_SERIAL_3>,<USB_CAN_SERIAL_4>
```

---

## PiPER/PiPER-X 单臂

运行下面的命令验证系统：

```bash
lerobot-teleoperate \
  --robot.type=piperx_follower \
  --robot.port=<FOLLOWER_USB_CAN_SERIAL> \
  --robot.id=my_piperx_follower \
  --teleop.type=piperx_leader \
  --teleop.port=<LEADER_USB_CAN_SERIAL> \
  --teleop.id=my_piperx_leader
```

---

## PiPER/PiPER-X 双臂

```bash
lerobot-teleoperate \
  --robot.type=bi_piperx_follower \
  --robot.id=my_bi_piperx_follower \
  --robot.left_arm_config.port=<LEFT_FOLLOWER_USB_CAN_SERIAL> \
  --robot.right_arm_config.port=<RIGHT_FOLLOWER_USB_CAN_SERIAL> \
  --teleop.type=bi_piperx_leader \
  --teleop.id=my_bi_piperx_leader \
  --teleop.left_arm_config.port=<LEFT_LEADER_USB_CAN_SERIAL> \
  --teleop.right_arm_config.port=<RIGHT_LEADER_USB_CAN_SERIAL>
```

对于非 X 版本的 PiPER：

```text
bi_piperx_follower → bi_piper_follower
bi_piperx_leader   → bi_piper_leader
```

---

# 3）Data Collection 数据采集

使用：

```text
lerobot-human-inloop-record
```

采集 Rollout 数据。

---

## SO 系列（SO100/SO101）

### 双臂数据采集模板

```bash
lerobot-human-inloop-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=my_bi_so101_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<LEFT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<RIGHT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "<REALSENSE_SN>", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=my_bi_so101_leader \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --dataset.single_task="<YOUR_TASK_DESCRIPTION>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.episode_time_s=<EPISODE_SECONDS> \
  --dataset.reset_time_s=<RESET_SECONDS> \
  --dataset.push_to_hub=true \
  --display_data=true
```

推荐：

* OpenCV 摄像头使用：

```text
fourcc: "MJPG"
```

* RealSense 使用：

```text
warmup_s
```

上面的例子中，`front` 使用 RealSense，但也可以按照相同结构切换成 OpenCV。

---

# AgileX（PiPER/PiPER-X）

### 双臂数据采集模板

下面是 PiPER-X 的左右双臂示例：

```bash
lerobot-human-inloop-record \
  --robot.type=bi_piperx_follower \
  --robot.id=my_bi_piperx_follower \
  --robot.left_arm_config.port=<LEFT_FOLLOWER_USB_CAN_SERIAL> \
  --robot.right_arm_config.port=<RIGHT_FOLLOWER_USB_CAN_SERIAL> \
  --teleop.type=bi_piperx_leader \
  --teleop.id=my_bi_piperx_leader \
  --teleop.left_arm_config.port=<LEFT_LEADER_USB_CAN_SERIAL> \
  --teleop.right_arm_config.port=<RIGHT_LEADER_USB_CAN_SERIAL> \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --dataset.single_task="<YOUR_TASK_DESCRIPTION>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.episode_time_s=<EPISODE_SECONDS> \
  --dataset.reset_time_s=<RESET_SECONDS> \
  --dataset.push_to_hub=true \
  --display_data=true
```

### 快捷键

```text
i              切换 Intervention 模式（Policy ↔ Teleoperation 接管）

s              标记成功，并结束当前 Episode

f              标记失败，并结束当前 Episode

Right Arrow    提前结束当前 Loop

Left Arrow     提前结束，并重新录制当前 Episode

Esc            停止整个录制过程
```

### 快速数据质量检查

```bash
lerobot-dataset-report --dataset <HF_USERNAME_OR_ORG>/<DATASET_NAME>
```

该命令会输出：

* 数据集元信息；
* 数据总量；
* Episode 长度统计 / 直方图；
* 成功率 / Intervention 指标；
* Task 列表；
* 完整的 Feature Schema。

---

# 4）Value Function Training：价值函数训练

在当前数据集上训练 Value Function。

当前默认使用：

```text
Pi*0.6
```

对应：

```text
--value.type=pistar06
```

### 单 GPU 模板

```bash
lerobot-value-train \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.push_to_hub=true \
  --value.repo_id=<HF_USERNAME_OR_ORG>/<VALUE_MODEL_REPO> \
  --batch_size=64 \
  --output_dir=outputs/value_train/<RUN_NAME> \
  --job_name=<RUN_NAME> \
  --wandb.enable=true
```

### 多 GPU 模板

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID_LIST> accelerate launch \
  --multi_gpu \
  --num_processes=<NUM_GPUS> \
  --mixed_precision=bf16 \
  $(which lerobot-value-train) \
  --batch_size=32/<NUM_GPUS> \
  <VALUE_TRAIN_ARGS>
```

---

## 使用其他 Value Function

如果想接入不同的 Value Function，该仓库提供的最小修改路径为：

1. 添加：

```text
src/lerobot/values/<your_value>/configuration_<your_value>.py
```

并使用：

```python
@PreTrainedConfig.register_subclass("<your_value>")
```

2. 添加：

```text
src/lerobot/values/<your_value>/modeling_<your_value>.py
```

其中实现：

```text
<YourValue>Policy(PreTrainedPolicy)
```

至少需要实现：

```text
forward
predict_value
build_training_raw_batch_hook
```

以支持：

```text
lerobot-value-train
```

3. 添加：

```text
src/lerobot/values/<your_value>/processor_<your_value>.py
```

并实现：

```text
make_<your_value>_pre_post_processors(...)
```

4. 删除或替换：

```text
src/lerobot/configs/value_train.py
src/lerobot/scripts/lerobot_value_infer.py
```

中目前仅支持 `pistar06` 的类型检查。

---

# 5）Value Inference：Value 推理

对数据集进行 Value 推理，并将：

* Value；
* Advantage；
* Indicator；

写回数据集。

三个字段含义：

### `value`

当前 Frame 的估计 **Return-to-Go（剩余回报）**。

### `advantage`

相对于基线的改进程度。

数值越高，表示轨迹质量相对于 Baseline 越好。

### `indicator`

根据 Advantage 二值化得到的训练标签。

---

## 单 GPU

```bash
lerobot-value-infer \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --inference.checkpoint_path=outputs/value_train/<RUN_NAME> \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_<TAG> \
  --acp.advantage_field=complementary_info.advantage_<TAG> \
  --acp.indicator_field=complementary_info.acp_indicator_<TAG> \
  --output_dir=outputs/value_infer/<RUN_NAME> \
  --job_name=<RUN_NAME>.infer
```

---

## 多 GPU

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID_LIST> accelerate launch \
  --multi_gpu \
  --num_processes=<NUM_GPUS> \
  --mixed_precision=bf16 \
  $(which lerobot-value-infer) \
  <VALUE_INFER_ARGS>
```

---

## 参数说明

```text
--acp.n_step
```

表示 **n-step Advantage 的时间范围**。

```text
--acp.positive_ratio
```

表示 Advantage 二值化之后的正样本比例。

例如：

```text
0.3
```

表示每个 Task 中取 Advantage 排名前 **30%** 的数据作为正样本。

---

## 预期新增字段

```text
complementary_info.value_<TAG>

complementary_info.advantage_<TAG>

complementary_info.acp_indicator_<TAG>
```

这些字段会被写回：

```text
--dataset.repo_id
```

指定的原始数据集。

---

# 6）Policy Training：策略训练

使用 **Advantage-Conditioned Tags（基于 Advantage 条件的标签）**训练 Policy。

### Policy 要求

Policy 必须支持：

```text
text/task input
```

因为 Advantage-Conditioned Tags 会被注入到：

```text
task text
```

中。

---

## 单 GPU

```bash
lerobot-train \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --policy.type=<POLICY_TYPE> \
  --policy.pretrained_path=<POLICY_PRETRAINED_PATH> \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=32 \
  --steps=30000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_<TAG> \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=outputs/train/<RUN_NAME> \
  --job_name=<RUN_NAME> \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=<HF_USERNAME_OR_ORG>/<POLICY_REPO>
```

`--acp.indicator_dropout_prob` 控制 Task Text 中标签的 Drop 概率。

设置：

```text
0.3
```

有助于模型同时学习：

* 带标签条件；
* 不带标签条件。

---

## 重要检查

```text
--acp.indicator_field
```

必须：

1. 存在于数据集中；
2. 是二值数据，即：

```text
0 / 1
```

---

## 多 GPU

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID_LIST> accelerate launch \
  --multi_gpu \
  --num_processes=<NUM_GPUS> \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --batch_size=32/<NUM_GPUS> \
  <POLICY_TRAIN_ARGS>
```

---

# 7）Closed-loop Rollout and Next Round

## 闭环 Rollout 与下一轮训练

将训练好的 Policy 部署到 **Human-in-the-loop** 模式，并采集下一轮数据集：

```bash
lerobot-human-inloop-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=my_bi_so101_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<LEFT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<RIGHT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "<REALSENSE_SN>", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=my_bi_so101_leader \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME_NEXT_ROUND> \
  --dataset.single_task="<YOUR_TASK_DESCRIPTION>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.episode_time_s=<EPISODE_SECONDS> \
  --dataset.reset_time_s=<RESET_SECONDS> \
  --dataset.push_to_hub=true \
  --display_data=true \
  --policy.path=<POLICY_CHECKPOINT_OR_HUB_ID> \
  --resume=true
```

---

# 数据集继续采集方式

有两种方式：

### 方式 1：原地追加

保持：
使用主从臂遥操作进行数据采集？
```text
--resume=true
```

继续向同一个数据集录制。

### 方式 2：合并多个 Round

使用官方 Dataset Editor，将不同 Round 的数据集合并：

```bash
lerobot-edit-dataset \
  --repo_id=<HF_USERNAME_OR_ORG>/<MERGED_DATASET_NAME> \
  --operation.type=merge \
  --operation.repo_ids="['<HF_USERNAME_OR_ORG>/<DATASET_ROUND_1>','<HF_USERNAME_OR_ORG>/<DATASET_ROUND_2>']"
```

---

# 相比默认 `lerobot-record` 新增的数据属性

Evo-RL 会额外记录：

```text
complementary_info.policy_action
```

表示每一步 Policy 输出的 Action。

```text
complementary_info.is_intervention
```

表示当前 Step 是否处于人工 Intervention。

```text
complementary_info.state
```

表示 Intervention 状态机的当前状态。

```text
complementary_info.collector_policy_id
```

表示当前 Step 的动作来源 ID：

```text
human
```

或者：

```text
policy ID
```

此外，每个 Episode 还保存：

```text
episode_success
```

用于记录该 Episode 的：

```text
success / failure
```

标签。

---

# 🔄 迭代式训练流程

整个流程可以抽象为：

```text
[多任务示教数据池]
        |
        v
[针对 Vision-Language-Action Policy 的 Offline RL 预训练]
        |
        v
[基于示教数据进行任务特定初始化 / Fine-tuning]
        |
        v
|---- Iteration k = 1..K -------------------------------------|
| 1) 部署当前 Policy π_k，采集新的 Rollout 数据              |
| 2) 合并到数据池：D <- D U new_data                         |
| 3) 在 D 上训练 Value Function                              |
| 4) 推理 Advantage，并将其二值化为 Indicator Tags            |
| 5) 训练 Advantage-Conditioned Policy，得到 π_{k+1}          |
|-------------------------------------------------------------|
        |
        v
[成功率和吞吐量进一步提升的更强 Policy]
```

---

# 🤗 Model & Dataset

Hugging Face Model：

```text
Coming Soon
```

Hugging Face Dataset：

```text
MINT-SJTU/RW-RL-Dataset
```

**RW-RL Dataset** 是 Evo-RL 配套的真实世界强化学习数据集。

它围绕**真实机器人上的迭代式 Policy Improvement（策略改进）**进行组织，包含：

* Teleoperation 示教数据；
* Human-in-the-loop Intervention 数据；
* Policy Rollout Trace；
* Episode 级 Success / Failure 标签；
* Intervention 状态；
* 用于 Value / Reward Modeling 的辅助信号。

该数据集用于支持：

* Offline RL；
* Value Learning；
* Advantage-Conditioned Policy Training；
* Closed-loop Rollout Analysis；

并覆盖真实机器人任务。

具体的发布说明、Schema、数据划分以及版本 Tag，请参考 Hugging Face Dataset Card。

---

# 💬 Community Channels 社区交流

* 微信公众号文章：Coming Soon
* 文档：`docs/README.md`
* GitHub Issues：可以创建 Issue
* Email：

```text
business@evomind-tech.com
```

* 可以通过扫描二维码加入微信群。

---

# 🏫 Affiliations

项目关联机构：

* SJTU
* EvoMind

---

# 📄 Citation

```bibtex
@misc{evorl2026,
  title        = {Evo-RL: Towards Iterative Policy Improvement in Real-World Offline RL},
  author       = {Evo-RL Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/MINT-SJTU/Evo-RL}}
}
```

---

# 📜 License

采用：

```text
Apache-2.0
```

详见：

```text
LICENSE
```

---

# ⭐ Star History

项目提供 GitHub Star History 图表。

# Human-in-loop 额外保存 Piper EEF 位姿

在 `lerobot-human-inloop-record` 原有命令后添加：

```bash
  --eef.enable=true
```

默认关闭，仅支持 `piper_follower` 和 `piperx_follower`。
开启后在每个采集帧的 Parquet 行中额外保存 state 和 action 的 EEF 位姿，
不切换机械臂控制模式：

- `observation.eef_pose` 使用从臂 `GetArmEndPoseMsgs()` 的实际末端反馈；
- `action.eef_pose` 使用本帧选中并发送的 6 个目标关节角，通过 Piper 官方正运动学计算。

## 字段与单位

| 字段 | 维度 | 内容 |
| --- | --- | --- |
| `observation.state` | 7 | 原有 6 个关节角度（度）和夹爪开度（mm） |
| `action` | 7 | 原有目标关节角度和夹爪开度 |
| `observation.eef_pose` | 6 | 从臂实际末端反馈（state），float32 |
| `action.eef_pose` | 6 | 本帧目标关节的正运动学位姿（action），float32 |

EEF 各分量顺序为：

```text
[eef.x_mm, eef.y_mm, eef.z_mm, eef.rx_deg, eef.ry_deg, eef.rz_deg]
```

XYZ 以毫米为单位，RX/RY/RZ 为 SDK 返回的欧拉角、单位为度。
SDK 原始 XYZ 为 0.001 mm、角度为 0.001°，保存前统一乘以 0.001；
保留 SDK 的正负号和欧拉角约定，不转成四元数或弧度。

两个 EEF 字段都表达在从臂机械臂基座参考系，位置指 **SDK 报告的 J6 末端参考点**。
`action.eef_pose` 使用 `C_PiperForwardKinematics(dh_is_offset=1)`，输入目标关节角、输出基座系
J6 位姿；它不是再次读取当前反馈，因此不会把 state 错当成 action。
安装夹爪不会使此功能自动得到夹爪抓取中心（TCP）。此实现不假设夹爪长度、不额外施加 TCP 偏移，
字段元数据记录 `reference_point="sdk_end_effector"`、`tcp_offset_applied=false`。
若控制器本身配置了工具偏置，其影响沿用 SDK 反馈；本采集代码不会重新解释或改变它。

原有 7 维关节状态、动作、夹爪开度和任务标签保持原格式，EEF 以两个独立字段额外保存。
机器人反馈和相机通过各自接口读取，帧号对应同一次采集循环，不表示严格硬件同步。

## 与 RGB-D 一起采集

可以同时使用 EEF 和软件对齐的深度采集：

```bash
  --dataset.fps=30 \
  --depth.enable=true \
  --depth.cameras='[head,wrist]' \
  --depth.align_to_color=true \
  --depth.width=640 \
  --depth.height=480 \
  --depth.fps=30 \
  --eef.enable=true
```

RGB 的两路 `robot.cameras` 配置保持 640×480、30 FPS。
只保存 RGB 和 EEF 时无需开启深度，也不会因此加载 Orbbec SDK。

## 反馈与续录

默认等待反馈最多 3 秒。若先前已收到反馈，但 SDK 时间戳持续不更新超过 0.5 秒，
将等待更新；超时则报错，不用零位姿代替。也会拒绝非法分量和时间戳倒退。
可按实际设备情况调整：

```bash
  --eef.timeout_s=3.0 \
  --eef.max_age_s=0.5
```

`max_age_s` 检查的是本程序观察到的 SDK 时间戳停止更新时长，不是相机与 EEF 的时间差。
SDK 返回的是组合反馈，公开接口不提供每个 CAN 分量的独立采样时间；本功能不宣称严格同步。

首次开启 EEF 请使用新的 `dataset.repo_id` 和 `dataset.root`。
历史数据没有此字段，不能在原来的关节数据集上直接切换 `eef.enable` 续录。
新数据集后续使用相同字段配置加 `--resume=true` 可以继续录制。
既有数据不会自动补出 EEF。

## 读取

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/demo_eef_001",
    root=Path("datas/demo_eef_001"),
    video_backend="pyav",
)
eef_pose = dataset[0]["observation.eef_pose"]
print(eef_pose)
```

采集参数只控制是否保存 EEF，不会自动把已有训练配置改成 EEF 控制；
训练若需要使用额外状态，应显式配置相应输入字段。

# Human-in-loop 采集 Orbbec 深度图

此功能仅用于 `lerobot-human-inloop-record`，默认关闭。
支持 Orbbec SDK v2 兼容设备（例如 Gemini 336）。

在采集使用的 Python 环境安装可选依赖：

```bash
python -m pip install pyorbbecsdk2
```

`pyorbbecsdk2` 在 Python 3.10 环境会安装 `av==12.3.0`；视频元数据读取和多段视频拼接
兼容该版本与较新 PyAV。无需为了深度采集手动升级或降级 PyAV。

保留原有 `--robot.cameras` 中的 `head`、`wrist` OpenCV 配置，在采集命令末尾追加：

```bash
  --depth.enable=true \
  --depth.cameras='[head,wrist]' \
  --depth.align_to_color=true
```

`head` 是第三视角，`wrist` 是腕部视角；名称必须与 `robot.cameras` 一致。
只保存腕部深度时用 `--depth.cameras='[wrist]'`。
Linux 下默认从现有 RGB 视频设备路径解析 USB 序列号，因此无需再找深度视频节点。
也可以显式指定：

```bash
  --depth.serials='{head: HEAD_SERIAL, wrist: WRIST_SERIAL}'
```

启用深度的相机会改由 Orbbec SDK 同时采集 MJPEG 彩色流和 Y16 深度流，
RGB 继续使用原有数据字段和视频保存流程。未选中的相机使用原有后端。
不会让 OpenCV 和 SDK 同时打开选中的相机。

深度流默认 640×480、30 FPS，可以修改为设备支持的配置：

```bash
  --depth.width=640 \
  --depth.height=480 \
  --depth.fps=30 \
  --depth.timeout_s=3.0
```

RGB 分辨率与帧率仍由 `robot.cameras` 设置，建议 RGB、深度与采集帧率相同。
如果设备不支持所选配置，会明确报错，不会静默降级为仅保存 RGB。
双相机 RGB-D 比纯 RGB 占用更多 USB 带宽，建议使用 USB 3 数据线与接口。

## 软件对齐（640×480，30 FPS）

开启深度采集后，`depth.align_to_color` 默认是 `true`。
使用官方 SDK 的 `AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)`，
将深度重投影到 RGB 坐标系，并通过 `set_match_target_resolution(true)` 保持 RGB 分辨率。
Pipeline 的对齐模式显式设为 `OBAlignMode.DISABLE`，禁用管线内对齐，
软件对齐由 `AlignFilter` 完成，不启用硬件 D2C。

两台相机在 `robot.cameras` 中都使用 `width: 640, height: 480, fps: 30, fourcc: "MJPG"`，
采集命令同时使用：

```bash
  --dataset.fps=30 \
  --depth.enable=true \
  --depth.cameras='[head,wrist]' \
  --depth.align_to_color=true \
  --depth.width=640 \
  --depth.height=480 \
  --depth.fps=30
```

保存的 RGB 为 640×480，深度 PNG 为对应的 640×480 单通道 uint16。
若配置 RGB 旋转，对齐后的深度使用同样的旋转；内参是旋转前的 SDK 内参，
旋转角度另存于标定文件，默认 0°。
对齐失败不会退回保存未对齐的深度；缺帧会等待，持续缺帧则超时报错。
30 FPS 是请求的设备流和数据集帧率，实际吞吐仍取决于两台设备、USB 和主机性能。

使用新的数据集目录采集对齐数据。这个开关作用于后续采集，
不会自动转换以前保存的 `aligned_to_color=false` 数据，也不应把两种深度坐标系混在同一批训练数据中。
如确实需要继续保存原生深度，可以显式使用 `--depth.align_to_color=false`。

## 文件格式

在 `dataset.root` 下额外保存：

```text
depth/
  episode_000000/
    cameras.json
    frames.jsonl
    head/frame_000000.png
    head/frame_000001.png
    wrist/frame_000000.png
    wrist/frame_000001.png
```

- PNG 为单通道 `uint16` 深度，无损保存 SDK 输出数值，不经过 H.264；默认保存软件对齐后的结果。
- episode 和 frame 编号对应 LeRobot 数据集中的 `episode_index`、`frame_index`。
- `frames.jsonl` 保存对应 RGB 字段、设备时间戳、深度缩放系数和深度文件路径。
- 距离（毫米）= PNG 像素原始值 × `depth_scale_mm`；原始值 0 表示无效深度。
- `cameras.json` 保存序列号、RGB 内参、保存深度的内参 `depth_intrinsics`、
  原生深度内参 `native_depth_intrinsics`、RGB/深度旋转角度和对齐模式。
- 默认 `aligned_to_color=true`、`alignment_mode="software"`，深度与对应 RGB 像素坐标一致；
  关闭对齐时为 `false`、`"none"`，深度保留原生坐标且不旋转。
  重投影后的遮挡/无覆盖区域可能为 0，应按无效深度处理。
- RGB-D 来自同一相机的 SDK frameset；空间对齐不等于双相机硬件时间同步。
- 深度是附加文件，不加入现有 LeRobot feature schema，训练代码不会自动加载它。

深度写入队列大小有上限，避免整段深度图堆积在内存中。只有成功保存的 episode
会生成正式深度目录；重录会删除该段暂存深度，退出时丢弃未保存段。
如果已经开始保存 episode，但 RGB 视频编码或元数据写入失败，已完成写入的深度
会保留在 `depth/failed_episode_XXXXXX_<随机后缀>/`，并在日志中打印路径。
这个目录仅供检查和恢复，不表示该 episode 已完整保存，不能直接当作成功数据使用。
`--resume=true` 会按已有 episode 数量继续编号，拒绝覆盖已有深度目录。
为完整采集 RGB-D，建议使用新的数据集目录；对纯 RGB 数据集续录不会补出历史深度。

读取示例：

```python
import json
from pathlib import Path

import cv2
import numpy as np

episode = Path("datas/demo_depth_001/depth/episode_000000")
with (episode / "frames.jsonl").open() as stream:
    frame = json.loads(next(stream))
camera = frame["cameras"]["wrist"]
raw = cv2.imread(str(episode / camera["path"]), cv2.IMREAD_UNCHANGED)
depth_mm = raw.astype(np.float32) * camera["depth_scale_mm"]
valid = raw != 0
```

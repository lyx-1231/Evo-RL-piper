# 采集数据完整性校验

停止采集程序后运行：

```bash
lerobot-validate-dataset \
  --dataset datas/demo_init_003 \
  --require-depth \
  --require-eef
```

校验器只读取数据，不修改数据集。默认会完整解码所有 RGB 视频，并读取每一张深度 PNG。
检查通过时退出码为 `0`，存在结构或格式错误时退出码为 `1`。

主要检查内容：

- `meta/info.json`、统计文件、任务表、episode 元数据是否存在且格式正确；
- 声明的 episode/帧数和 Parquet 实际行数是否一致；
- 全局索引、episode 索引、逐段 frame 索引和 30 FPS 时间戳是否连续；
- 所有数值字段是否与声明的 dtype/shape 一致，是否含空值、NaN 或无穷值；
- 每个 MP4 是否存在、尺寸和帧率是否正确、能否完整解码、帧数是否匹配；
- 每段深度 manifest 是否与 RGB/Parquet 帧一一对应；
- 每张深度图是否为单通道 `uint16` PNG，尺寸是否与清单及对齐后的 RGB 一致；
- 深度内参、序列号、深度比例和软件对齐标记是否一致；
- EEF state/action 是否都是 6 维 `float32`，并声明为机械臂基坐标系。

纯 RGB 数据集不需要添加 `--require-depth`。旧数据只有 `observation.eef_pose` 时可以不加
`--require-eef`，校验器会给出警告；新数据开启 `--eef.enable=true` 后应包含
`observation.eef_pose` 和 `action.eef_pose`，建议始终添加 `--require-eef`。

机器处理可使用 JSON 输出：

```bash
lerobot-validate-dataset --dataset datas/demo_init_003 --require-depth --require-eef --json
```

若只想快速检查目录和引用，不完整解码视频，可临时添加 `--skip-video-decode`。正式验收数据时不要使用该选项。

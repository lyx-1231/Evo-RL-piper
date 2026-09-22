"""Read-only integrity and format validation for recorded LeRobot v3 datasets."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.scripts.lerobot_dataset_report import resolve_dataset_root

CORE_FEATURES = {
    "timestamp": ("float32", [1]),
    "frame_index": ("int64", [1]),
    "episode_index": ("int64", [1]),
    "index": ("int64", [1]),
    "task_index": ("int64", [1]),
}
EEF_KEYS = ("observation.eef_pose", "action.eef_pose")
INTRINSIC_FIELDS = ("width", "height", "fx", "fy", "cx", "cy")


class DatasetValidator:
    def __init__(
        self,
        root: Path,
        *,
        require_depth: bool = False,
        require_eef: bool = False,
        decode_videos: bool = True,
    ):
        self.root = root.resolve()
        self.require_depth = require_depth
        self.require_eef = require_eef
        self.decode_videos = decode_videos
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {
            "episodes": 0,
            "frames": 0,
            "video_files": 0,
            "decoded_video_frames": 0,
            "depth_cameras": [],
            "depth_pngs": 0,
            "eef_fields": [],
        }
        self.info: dict[str, Any] = {}
        self.features: dict[str, dict[str, Any]] = {}
        self.fps = 0.0
        self.episodes: list[dict[str, Any]] = []
        self.data: pa.Table | None = None
        self.data_arrays: dict[str, np.ndarray] = {}

    def issue(self, severity: str, code: str, message: str, **location: Any) -> None:
        issue = {"code": code, "message": message}
        issue.update({key: value for key, value in location.items() if value is not None})
        (self.errors if severity == "error" else self.warnings).append(issue)

    def error(self, code: str, message: str, **location: Any) -> None:
        self.issue("error", code, message, **location)

    def warning(self, code: str, message: str, **location: Any) -> None:
        self.issue("warning", code, message, **location)

    def _snapshot(self) -> dict[str, tuple[int, int]]:
        try:
            return {
                path.relative_to(self.root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in self.root.rglob("*")
                if path.is_file()
            }
        except OSError as exc:
            self.error("snapshot_failed", f"Could not snapshot dataset files: {exc}")
            return {}

    def _read_json(self, relative_path: str) -> Any | None:
        path = self.root / relative_path
        if not path.is_file():
            self.error("missing_file", "Required file is missing.", path=relative_path)
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.error("invalid_json", f"Cannot parse JSON: {exc}", path=relative_path)
            return None

    def validate(self) -> dict[str, Any]:
        if not self.root.is_dir():
            self.error("missing_dataset", "Dataset directory does not exist.", path=str(self.root))
            return self.report()

        initial_snapshot = self._snapshot()
        if self._validate_metadata():
            self._validate_parquet()
            if self.data is not None and self.episodes:
                self._validate_episode_rows()
                self._validate_videos()
                self._validate_depth()
        final_snapshot = self._snapshot()
        if initial_snapshot != final_snapshot:
            self.error(
                "dataset_changed",
                "Dataset files changed during validation. Stop recording and run validation again.",
            )
        return self.report()

    def report(self) -> dict[str, Any]:
        return {
            "valid": not self.errors,
            "dataset_root": str(self.root),
            "summary": self.summary,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def _validate_metadata(self) -> bool:
        info = self._read_json("meta/info.json")
        if not isinstance(info, dict):
            return False
        self.info = info
        self.features = info.get("features", {})
        if not isinstance(self.features, dict) or not self.features:
            self.error("invalid_features", "meta/info.json has no non-empty features mapping.")
            return False

        try:
            self.fps = float(info.get("fps", 0))
        except (TypeError, ValueError):
            self.fps = 0.0
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.error("invalid_fps", f"Dataset fps must be positive and finite, got {info.get('fps')!r}.")

        if not str(info.get("codebase_version", "")).startswith("v3"):
            self.error(
                "unsupported_version",
                f"Expected a LeRobot v3 dataset, got {info.get('codebase_version')!r}.",
            )

        for key, (dtype, shape) in CORE_FEATURES.items():
            feature = self.features.get(key)
            if feature is None:
                self.error("missing_feature", f"Required feature {key!r} is missing.", feature=key)
            elif feature.get("dtype") != dtype or list(feature.get("shape", [])) != shape:
                self.error(
                    "invalid_feature",
                    f"{key!r} must have dtype={dtype}, shape={shape}; got {feature!r}.",
                    feature=key,
                )

        for key, feature in self.features.items():
            if not isinstance(feature, dict) or "dtype" not in feature or "shape" not in feature:
                self.error("invalid_feature", "Feature must define dtype and shape.", feature=key)
                continue
            shape = feature.get("shape")
            if (
                not isinstance(shape, (list, tuple))
                or not shape
                or any(not isinstance(value, int) or value <= 0 for value in shape)
            ):
                self.error("invalid_feature_shape", f"Invalid feature shape: {shape!r}.", feature=key)
            names = feature.get("names")
            if (
                names is not None
                and feature.get("dtype") not in {"image", "video"}
                and len(shape) == 1
                and len(names) != shape[0]
            ):
                self.error(
                    "feature_names_mismatch",
                    f"Feature has shape {shape} but {len(names)} component names.",
                    feature=key,
                )

        present_eef = [key for key in EEF_KEYS if key in self.features]
        self.summary["eef_fields"] = present_eef
        if self.require_eef:
            for key in EEF_KEYS:
                if key not in self.features:
                    self.error("missing_eef", f"Required EEF feature {key!r} is missing.", feature=key)
        elif len(present_eef) == 1:
            self.warning(
                "incomplete_eef_pair",
                f"Only {present_eef[0]!r} exists; new EEF datasets should contain both state and action EEF.",
            )
        for key in present_eef:
            feature = self.features[key]
            if feature.get("dtype") != "float32" or list(feature.get("shape", [])) != [6]:
                self.error(
                    "invalid_eef_feature",
                    "EEF pose must have dtype=float32 and shape=[6].",
                    feature=key,
                )
            feature_info = feature.get("info", {})
            if feature_info.get("reference_frame") != "robot_base":
                self.error(
                    "invalid_eef_reference_frame",
                    "EEF pose must declare reference_frame='robot_base'.",
                    feature=key,
                )
            if feature_info.get("position_unit") != "mm" or feature_info.get("orientation_unit") != "degree":
                self.error(
                    "invalid_eef_units",
                    "EEF pose must use position_unit='mm' and orientation_unit='degree'.",
                    feature=key,
                )

        if self._read_json("meta/stats.json") is None:
            return False
        for relative_path in ("meta/tasks.parquet",):
            if not (self.root / relative_path).is_file():
                self.error("missing_file", "Required file is missing.", path=relative_path)
        return True

    def _read_parquet_tree(self, relative_path: str) -> pa.Table | None:
        directory = self.root / relative_path
        paths = sorted(directory.glob("*/*.parquet")) if directory.is_dir() else []
        if not paths:
            self.error("missing_parquet", "No parquet files found.", path=relative_path)
            return None
        try:
            tables = [pq.read_table(path) for path in paths]
            return pa.concat_tables(tables, promote_options="default")
        except Exception as exc:
            self.error("invalid_parquet", f"Cannot read parquet files: {exc}", path=relative_path)
            return None

    def _validate_parquet(self) -> None:
        episodes_table = self._read_parquet_tree("meta/episodes")
        data = self._read_parquet_tree("data")
        if episodes_table is None or data is None:
            return
        required_episode_columns = {
            "episode_index",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        }
        missing_episode_columns = required_episode_columns - set(episodes_table.column_names)
        if missing_episode_columns:
            self.error(
                "missing_episode_columns",
                f"Episode metadata is missing columns: {sorted(missing_episode_columns)}.",
            )
            return
        if episodes_table["episode_index"].null_count:
            self.error("null_episode_index", "Episode metadata contains null episode indexes.")
            return
        self.episodes = sorted(episodes_table.to_pylist(), key=lambda row: row["episode_index"])
        self.data = data
        self.summary["episodes"] = len(self.episodes)
        self.summary["frames"] = data.num_rows

        expected_columns = {key for key, feature in self.features.items() if feature.get("dtype") != "video"}
        actual_columns = set(data.column_names)
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        if missing_columns:
            self.error("missing_data_columns", f"Data parquet is missing columns: {sorted(missing_columns)}.")
        if extra_columns:
            self.error("extra_data_columns", f"Data parquet has undeclared columns: {sorted(extra_columns)}.")

        for key in expected_columns & actual_columns:
            column = data[key]
            if column.null_count:
                self.error(
                    "null_values",
                    f"Feature contains {column.null_count} null rows.",
                    feature=key,
                )
            self._validate_arrow_type(key, data.schema.field(key).type)
            dtype = self.features[key].get("dtype")
            if dtype in {"float16", "float32", "float64", "int8", "int16", "int32", "int64"}:
                try:
                    values = np.asarray(column.to_pylist(), dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    self.error(
                        "invalid_numeric_values", f"Cannot convert values to numbers: {exc}", feature=key
                    )
                    continue
                if not np.isfinite(values).all():
                    bad_count = int(np.count_nonzero(~np.isfinite(values)))
                    self.error(
                        "nonfinite_values",
                        f"Feature contains {bad_count} NaN or infinite values.",
                        feature=key,
                    )
                self.data_arrays[key] = values

        self._validate_tasks()
        self._validate_stats()

    def _validate_arrow_type(self, key: str, arrow_type: pa.DataType) -> None:
        feature = self.features[key]
        dtype = feature.get("dtype")
        shape = list(feature.get("shape", []))
        expected_scalar = {
            "float16": pa.float16(),
            "float32": pa.float32(),
            "float64": pa.float64(),
            "int8": pa.int8(),
            "int16": pa.int16(),
            "int32": pa.int32(),
            "int64": pa.int64(),
            "string": pa.string(),
        }.get(dtype)
        if expected_scalar is None or dtype == "image":
            return
        if shape == [1]:
            valid = arrow_type == expected_scalar
        elif len(shape) == 1:
            valid = (
                (pa.types.is_fixed_size_list(arrow_type) and arrow_type.list_size == shape[0])
                or pa.types.is_list(arrow_type)
            ) and arrow_type.value_type == expected_scalar
        else:
            valid = True
        if not valid:
            self.error(
                "invalid_arrow_type",
                f"Feature metadata expects dtype={dtype}, shape={shape}, but parquet type is {arrow_type}.",
                feature=key,
            )

    def _validate_tasks(self) -> None:
        path = self.root / "meta/tasks.parquet"
        if not path.is_file():
            return
        try:
            tasks = pq.read_table(path)
        except Exception as exc:
            self.error("invalid_tasks", f"Cannot read tasks parquet: {exc}", path="meta/tasks.parquet")
            return
        if "task_index" not in tasks.column_names:
            self.error("invalid_tasks", "Tasks parquet is missing task_index.", path="meta/tasks.parquet")
            return
        task_indexes = tasks["task_index"].to_pylist()
        if sorted(task_indexes) != list(range(len(task_indexes))):
            self.error("invalid_task_indexes", "Task indexes must be unique and contiguous from zero.")
        declared = self.info.get("total_tasks")
        if declared != len(task_indexes):
            self.error(
                "task_count_mismatch",
                f"meta/info.json declares {declared} tasks, but tasks.parquet contains {len(task_indexes)}.",
            )
        if "task_index" in self.data_arrays and task_indexes:
            used = set(self.data_arrays["task_index"].astype(np.int64).tolist())
            unknown = used - set(task_indexes)
            if unknown:
                self.error(
                    "unknown_task_index", f"Data rows reference unknown task indexes: {sorted(unknown)}."
                )

    def _validate_stats(self) -> None:
        stats = self._read_json("meta/stats.json")
        if not isinstance(stats, dict):
            return
        total_frames = self.data.num_rows if self.data is not None else 0
        for key, values in stats.items():
            if key not in self.features:
                self.warning("extra_stats", "Stats exist for an undeclared feature.", feature=key)
            if not isinstance(values, dict):
                self.error("invalid_stats", "Feature stats must be a mapping.", feature=key)
                continue
            counts = values.get("count")
            if (
                counts is not None
                and self.features.get(key, {}).get("dtype") not in {"image", "video"}
                and (not counts or int(counts[0]) != total_frames)
            ):
                self.error(
                    "stats_count_mismatch",
                    f"Stats count is {counts!r}, expected [{total_frames}].",
                    feature=key,
                )

    def _validate_episode_rows(self) -> None:
        assert self.data is not None
        declared_episodes = self.info.get("total_episodes")
        declared_frames = self.info.get("total_frames")
        if declared_episodes != len(self.episodes):
            self.error(
                "episode_count_mismatch",
                f"meta/info.json declares {declared_episodes} episodes, found {len(self.episodes)}.",
            )
        if declared_frames != self.data.num_rows:
            self.error(
                "frame_count_mismatch",
                f"meta/info.json declares {declared_frames} frames, found {self.data.num_rows}.",
            )
        episode_indexes = [row["episode_index"] for row in self.episodes]
        if episode_indexes != list(range(len(self.episodes))):
            self.error("episode_indexes", f"Episode indexes are not contiguous from zero: {episode_indexes}.")

        if "index" not in self.data_arrays:
            return
        order = np.argsort(self.data_arrays["index"].reshape(-1))
        for key, values in list(self.data_arrays.items()):
            self.data_arrays[key] = values[order]
        indexes = self.data_arrays["index"].reshape(-1).astype(np.int64)
        if not np.array_equal(indexes, np.arange(self.data.num_rows)):
            self.error(
                "global_indexes", "Global data indexes contain gaps, duplicates, or do not start at zero."
            )

        next_start = 0
        total_lengths = 0
        for row in self.episodes:
            episode_index = int(row["episode_index"])
            try:
                length = int(row["length"])
                start = int(row["dataset_from_index"])
                end = int(row["dataset_to_index"])
            except (TypeError, ValueError) as exc:
                self.error("invalid_episode_range", f"Invalid episode range: {exc}", episode=episode_index)
                continue
            total_lengths += length
            if length <= 0:
                self.error("empty_episode", "Committed episode has no frames.", episode=episode_index)
            if start != next_start or end - start != length:
                self.error(
                    "episode_range",
                    f"Expected range [{next_start}, {next_start + length}), got [{start}, {end}).",
                    episode=episode_index,
                )
            next_start = end
            if not (0 <= start <= end <= self.data.num_rows):
                self.error("episode_range", "Episode range falls outside data rows.", episode=episode_index)
                continue
            if "episode_index" in self.data_arrays:
                actual = self.data_arrays["episode_index"][start:end].reshape(-1).astype(np.int64)
                if not np.all(actual == episode_index):
                    self.error(
                        "episode_data_mismatch",
                        "Data rows do not all contain the matching episode_index.",
                        episode=episode_index,
                    )
            if "frame_index" in self.data_arrays:
                actual = self.data_arrays["frame_index"][start:end].reshape(-1).astype(np.int64)
                if not np.array_equal(actual, np.arange(length)):
                    self.error(
                        "frame_indexes",
                        "Frame indexes are not contiguous from zero.",
                        episode=episode_index,
                    )
            if "timestamp" in self.data_arrays and self.fps > 0:
                actual = self.data_arrays["timestamp"][start:end].reshape(-1)
                expected = np.arange(length) / self.fps
                if not np.allclose(actual, expected, atol=1e-4, rtol=0):
                    max_error = float(np.max(np.abs(actual - expected))) if length else 0.0
                    self.error(
                        "frame_timestamps",
                        f"Frame timestamps do not match frame_index/fps; max error={max_error:.6g}s.",
                        episode=episode_index,
                    )
            data_path = self.root / str(self.info.get("data_path", "")).format(
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            if not data_path.is_file():
                self.error(
                    "missing_data_file",
                    "Episode references a missing data parquet file.",
                    episode=episode_index,
                    path=self._relative(data_path),
                )
        if total_lengths != self.data.num_rows:
            self.error(
                "episode_lengths",
                f"Episode lengths sum to {total_lengths}, but data contains {self.data.num_rows} rows.",
            )

    def _video_keys(self) -> list[str]:
        return [key for key, feature in self.features.items() if feature.get("dtype") == "video"]

    def _image_hw(self, feature: dict[str, Any]) -> tuple[int, int] | None:
        shape = list(feature.get("shape", []))
        names = list(feature.get("names") or [])
        if "height" in names and "width" in names:
            return int(shape[names.index("height")]), int(shape[names.index("width")])
        return (int(shape[0]), int(shape[1])) if len(shape) >= 2 else None

    def _validate_videos(self) -> None:
        video_keys = self._video_keys()
        video_template = self.info.get("video_path")
        if video_keys and not video_template:
            self.error("missing_video_path", "Video features exist but meta/info.json has no video_path.")
            return
        referenced: dict[tuple[str, Path], list[tuple[int, int, float, float]]] = defaultdict(list)
        for key in video_keys:
            prefix = f"videos/{key}"
            columns = [
                f"{prefix}/{suffix}"
                for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
            ]
            for row in self.episodes:
                episode_index = int(row["episode_index"])
                missing = [column for column in columns if column not in row or row[column] is None]
                if missing:
                    self.error(
                        "missing_video_metadata",
                        f"Missing video metadata columns: {missing}.",
                        episode=episode_index,
                        feature=key,
                    )
                    continue
                path = self.root / str(video_template).format(
                    video_key=key,
                    chunk_index=int(row[columns[0]]),
                    file_index=int(row[columns[1]]),
                )
                referenced[(key, path)].append(
                    (
                        episode_index,
                        int(row["length"]),
                        float(row[columns[2]]),
                        float(row[columns[3]]),
                    )
                )
        self.summary["video_files"] = len(referenced)
        for (key, path), segments in referenced.items():
            if not path.is_file():
                self.error(
                    "missing_video",
                    "Referenced video file is missing.",
                    feature=key,
                    path=self._relative(path),
                )
                continue
            segments.sort(key=lambda item: item[2])
            expected_from = 0.0
            expected_frames = 0
            for episode_index, length, from_timestamp, to_timestamp in segments:
                expected_duration = length / self.fps if self.fps > 0 else 0.0
                if abs(from_timestamp - expected_from) > 1e-4:
                    self.error(
                        "video_segment_gap",
                        f"Expected segment to start at {expected_from:.6f}s, got {from_timestamp:.6f}s.",
                        episode=episode_index,
                        feature=key,
                        path=self._relative(path),
                    )
                if abs((to_timestamp - from_timestamp) - expected_duration) > 1e-4:
                    self.error(
                        "video_segment_duration",
                        "Video segment duration does not match episode length/fps.",
                        episode=episode_index,
                        feature=key,
                        path=self._relative(path),
                    )
                expected_from = to_timestamp
                expected_frames += length
            if self.decode_videos:
                self._decode_video(key, path, expected_frames)

        referenced_paths = {path.resolve() for _, path in referenced}
        videos_root = self.root / "videos"
        if videos_root.is_dir():
            extra = sorted(
                self._relative(path)
                for path in videos_root.glob("**/*.mp4")
                if path.resolve() not in referenced_paths
            )
            if extra:
                self.warning("unreferenced_videos", f"Found unreferenced video files: {extra}.")

    def _decode_video(self, key: str, path: Path, expected_frames: int) -> None:
        expected_hw = self._image_hw(self.features[key])
        frame_times: list[float] = []
        corrupt = 0
        try:
            with av.open(str(path), mode="r") as container:
                if not container.streams.video:
                    self.error(
                        "missing_video_stream", "File contains no video stream.", path=self._relative(path)
                    )
                    return
                stream = container.streams.video[0]
                if expected_hw and (stream.height, stream.width) != expected_hw:
                    self.error(
                        "video_dimensions",
                        f"Video is {stream.width}x{stream.height}, expected {expected_hw[1]}x{expected_hw[0]}.",
                        feature=key,
                        path=self._relative(path),
                    )
                if stream.average_rate and self.fps > 0 and abs(float(stream.average_rate) - self.fps) > 1e-3:
                    self.error(
                        "video_fps",
                        f"Video reports {float(stream.average_rate):.6g} fps, expected {self.fps:.6g}.",
                        feature=key,
                        path=self._relative(path),
                    )
                for frame in container.decode(stream):
                    corrupt += int(frame.is_corrupt)
                    if expected_hw and (frame.height, frame.width) != expected_hw:
                        self.error(
                            "video_frame_dimensions",
                            f"Decoded frame is {frame.width}x{frame.height}, expected {expected_hw[1]}x{expected_hw[0]}.",
                            feature=key,
                            path=self._relative(path),
                            frame=len(frame_times),
                        )
                        break
                    frame_times.append(float(frame.time) if frame.time is not None else math.nan)
        except Exception as exc:
            self.error(
                "video_decode",
                f"Video cannot be fully decoded: {exc}",
                feature=key,
                path=self._relative(path),
            )
            return
        self.summary["decoded_video_frames"] += len(frame_times)
        if len(frame_times) != expected_frames:
            self.error(
                "video_frame_count",
                f"Decoded {len(frame_times)} frames, expected {expected_frames}.",
                feature=key,
                path=self._relative(path),
            )
        if corrupt:
            self.error(
                "corrupt_video_frames",
                f"Decoder marked {corrupt} frames as corrupt.",
                feature=key,
                path=self._relative(path),
            )
        times = np.asarray(frame_times)
        if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            self.error(
                "video_timestamps",
                "Decoded timestamps are missing, repeated, or move backwards.",
                feature=key,
                path=self._relative(path),
            )
        elif len(times) and self.fps > 0:
            expected = np.arange(len(times)) / self.fps
            if np.max(np.abs(times - expected)) > (0.5 / self.fps + 1e-6):
                self.error(
                    "video_timestamps",
                    "Decoded timestamps do not follow the dataset frame rate.",
                    feature=key,
                    path=self._relative(path),
                )

    def _validate_depth(self) -> None:
        depth_root = self.root / "depth"
        if not depth_root.is_dir():
            if self.require_depth:
                self.error("missing_depth", "Depth data is required but the depth directory is missing.")
            return
        episode_dirs = {
            int(match.group(1)): path
            for path in depth_root.iterdir()
            if path.is_dir() and (match := re.fullmatch(r"episode_(\d{6})", path.name))
        }
        expected_indexes = {int(row["episode_index"]) for row in self.episodes}
        missing = sorted(expected_indexes - set(episode_dirs))
        extra = sorted(set(episode_dirs) - expected_indexes)
        if missing:
            self.error("missing_depth_episodes", f"Missing depth sidecars for episodes: {missing}.")
        if extra:
            self.error("orphan_depth_episodes", f"Depth sidecars have no committed episode: {extra}.")
        recovery_dirs = sorted(
            path.name
            for path in depth_root.iterdir()
            if path.is_dir() and not re.fullmatch(r"episode_\d{6}", path.name)
        )
        if recovery_dirs:
            self.warning(
                "depth_recovery_directories",
                f"Found uncommitted depth staging/recovery directories: {recovery_dirs}.",
            )

        expected_camera_names: set[str] | None = None
        all_zero_counts: defaultdict[str, int] = defaultdict(int)
        for row in self.episodes:
            episode_index = int(row["episode_index"])
            episode_dir = episode_dirs.get(episode_index)
            if episode_dir is None:
                continue
            cameras = self._read_episode_json(episode_dir, "cameras.json", episode_index)
            manifest = self._read_manifest(episode_dir, episode_index)
            if not isinstance(cameras, dict) or manifest is None:
                continue
            camera_names = set(cameras)
            if not camera_names:
                self.error("empty_depth_cameras", "No depth cameras declared.", episode=episode_index)
                continue
            if expected_camera_names is None:
                expected_camera_names = camera_names
                self.summary["depth_cameras"] = sorted(camera_names)
            elif camera_names != expected_camera_names:
                self.error(
                    "depth_camera_set",
                    f"Depth camera set changed from {sorted(expected_camera_names)} to {sorted(camera_names)}.",
                    episode=episode_index,
                )
            self._validate_calibration(cameras, episode_index)
            length = int(row["length"])
            if len(manifest) != length:
                self.error(
                    "depth_manifest_count",
                    f"Manifest contains {len(manifest)} frames, expected {length}.",
                    episode=episode_index,
                )
            referenced_paths: defaultdict[str, set[Path]] = defaultdict(set)
            timestamp_series: defaultdict[str, list[float]] = defaultdict(list)
            for expected_frame, entry in enumerate(manifest):
                self._validate_depth_entry(
                    entry,
                    episode_dir,
                    episode_index,
                    expected_frame,
                    cameras,
                    camera_names,
                    referenced_paths,
                    timestamp_series,
                    all_zero_counts,
                    row,
                )
            for name in camera_names:
                actual_paths = (
                    set((episode_dir / name).glob("*.png")) if (episode_dir / name).is_dir() else set()
                )
                missing_paths = referenced_paths[name] - actual_paths
                extra_paths = actual_paths - referenced_paths[name]
                if missing_paths:
                    self.error(
                        "missing_depth_pngs",
                        f"{len(missing_paths)} manifest-referenced PNG files are missing.",
                        episode=episode_index,
                        camera=name,
                    )
                if extra_paths:
                    self.error(
                        "extra_depth_pngs",
                        f"{len(extra_paths)} PNG files are not referenced by the manifest.",
                        episode=episode_index,
                        camera=name,
                    )
                timestamps = np.asarray(timestamp_series[name])
                if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
                    self.error(
                        "depth_device_timestamps",
                        "Depth device timestamps repeat or move backwards.",
                        episode=episode_index,
                        camera=name,
                    )
        for name, count in sorted(all_zero_counts.items()):
            if count:
                self.warning(
                    "all_zero_depth_frames",
                    f"Camera {name!r} contains {count} all-zero depth frames.",
                    camera=name,
                )

    def _read_episode_json(self, episode_dir: Path, filename: str, episode_index: int) -> Any | None:
        path = episode_dir / filename
        if not path.is_file():
            self.error(
                "missing_depth_file", f"Missing {filename}.", episode=episode_index, path=self._relative(path)
            )
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.error(
                "invalid_depth_json",
                f"Cannot parse {filename}: {exc}",
                episode=episode_index,
                path=self._relative(path),
            )
            return None

    def _read_manifest(self, episode_dir: Path, episode_index: int) -> list[dict[str, Any]] | None:
        path = episode_dir / "frames.jsonl"
        if not path.is_file():
            self.error(
                "missing_depth_file",
                "Missing frames.jsonl.",
                episode=episode_index,
                path=self._relative(path),
            )
            return None
        entries: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        self.error(
                            "blank_manifest_line",
                            "Depth manifest contains a blank line.",
                            episode=episode_index,
                            frame=line_number - 1,
                        )
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        self.error(
                            "invalid_manifest_json",
                            f"Cannot parse line {line_number}: {exc}",
                            episode=episode_index,
                        )
                        continue
                    if not isinstance(entry, dict):
                        self.error(
                            "invalid_manifest_entry",
                            f"Line {line_number} is not a JSON object.",
                            episode=episode_index,
                        )
                        continue
                    entries.append(entry)
        except (OSError, UnicodeError) as exc:
            self.error("invalid_depth_manifest", f"Cannot read manifest: {exc}", episode=episode_index)
            return None
        return entries

    def _validate_calibration(self, cameras: dict[str, Any], episode_index: int) -> None:
        for name, calibration in cameras.items():
            if not isinstance(calibration, dict) or not str(calibration.get("serial", "")).strip():
                self.error(
                    "invalid_depth_calibration",
                    "Camera serial is missing.",
                    episode=episode_index,
                    camera=name,
                )
                continue
            for intrinsic_name in ("color_intrinsics", "depth_intrinsics"):
                intrinsics = calibration.get(intrinsic_name)
                if not isinstance(intrinsics, dict):
                    self.error(
                        "missing_intrinsics",
                        f"Missing {intrinsic_name}.",
                        episode=episode_index,
                        camera=name,
                    )
                    continue
                try:
                    values = [float(intrinsics[field]) for field in INTRINSIC_FIELDS]
                except (KeyError, TypeError, ValueError) as exc:
                    self.error(
                        "invalid_intrinsics",
                        f"Invalid {intrinsic_name}: {exc}.",
                        episode=episode_index,
                        camera=name,
                    )
                    continue
                if not all(math.isfinite(value) for value in values) or min(values[:4]) <= 0:
                    self.error(
                        "invalid_intrinsics",
                        f"{intrinsic_name} contains non-finite or non-positive dimensions/focal lengths.",
                        episode=episode_index,
                        camera=name,
                    )
            if not calibration.get("aligned_to_color", False):
                self.warning(
                    "depth_not_aligned",
                    "Depth is not declared as aligned to RGB.",
                    episode=episode_index,
                    camera=name,
                )
            elif calibration.get("alignment_mode") != "software":
                self.error(
                    "depth_alignment_mode",
                    "Aligned depth must declare alignment_mode='software'.",
                    episode=episode_index,
                    camera=name,
                )

    def _validate_depth_entry(
        self,
        entry: dict[str, Any],
        episode_dir: Path,
        episode_index: int,
        expected_frame: int,
        calibrations: dict[str, Any],
        camera_names: set[str],
        referenced_paths: defaultdict[str, set[Path]],
        timestamp_series: defaultdict[str, list[float]],
        all_zero_counts: defaultdict[str, int],
        episode_row: dict[str, Any],
    ) -> None:
        if entry.get("episode_index") != episode_index or entry.get("frame_index") != expected_frame:
            self.error(
                "depth_manifest_indexes",
                "Manifest episode_index/frame_index does not match its position.",
                episode=episode_index,
                frame=expected_frame,
            )
        if self.fps > 0:
            expected_timestamp = expected_frame / self.fps
            try:
                timestamp = float(entry.get("timestamp"))
            except (TypeError, ValueError):
                timestamp = math.nan
            if not math.isfinite(timestamp) or abs(timestamp - expected_timestamp) > 1e-4:
                self.error(
                    "depth_manifest_timestamp",
                    f"Manifest timestamp must be {expected_timestamp:.6f}s, got {entry.get('timestamp')!r}.",
                    episode=episode_index,
                    frame=expected_frame,
                )
        cameras = entry.get("cameras")
        if not isinstance(cameras, dict) or set(cameras) != camera_names:
            self.error(
                "depth_manifest_cameras",
                f"Expected cameras {sorted(camera_names)}, got {sorted(cameras) if isinstance(cameras, dict) else cameras!r}.",
                episode=episode_index,
                frame=expected_frame,
            )
            return
        for name, metadata in cameras.items():
            if not isinstance(metadata, dict):
                self.error(
                    "invalid_depth_metadata",
                    "Camera metadata is not an object.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                )
                continue
            if not isinstance(calibrations[name], dict):
                continue
            relative = Path(str(metadata.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                self.error(
                    "unsafe_depth_path",
                    f"Invalid relative PNG path: {relative}.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                )
                continue
            path = episode_dir / relative
            referenced_paths[name].add(path)
            if metadata.get("serial") != calibrations[name].get("serial"):
                self.error(
                    "depth_serial_mismatch",
                    "Manifest serial does not match cameras.json.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                )
            rgb_key = metadata.get("rgb_key")
            if rgb_key not in self.features or self.features.get(rgb_key, {}).get("dtype") != "video":
                self.error(
                    "invalid_depth_rgb_key",
                    f"Manifest references unknown RGB video feature {rgb_key!r}.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                )
            try:
                scale = float(metadata.get("depth_scale_mm"))
            except (TypeError, ValueError):
                scale = math.nan
            if not math.isfinite(scale) or scale <= 0:
                self.error(
                    "invalid_depth_scale",
                    f"depth_scale_mm must be positive and finite, got {metadata.get('depth_scale_mm')!r}.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                )
            if metadata.get("aligned_to_color") != calibrations[name].get("aligned_to_color"):
                self.error(
                    "depth_alignment_mismatch",
                    "Manifest alignment flag does not match cameras.json.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                )
            try:
                timestamp_series[name].append(float(metadata["depth_timestamp_ms"]))
            except (KeyError, TypeError, ValueError):
                self.error(
                    "invalid_depth_timestamp",
                    "Missing or invalid depth_timestamp_ms.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                )
            if not path.is_file():
                return
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            self.summary["depth_pngs"] += 1
            if image is None:
                self.error(
                    "unreadable_depth_png",
                    "OpenCV cannot decode depth PNG.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                    path=self._relative(path),
                )
                return
            if image.dtype != np.uint16 or image.ndim != 2:
                self.error(
                    "invalid_depth_format",
                    f"Depth PNG must be single-channel uint16, got dtype={image.dtype}, shape={image.shape}.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                    path=self._relative(path),
                )
                return
            expected_shape = (metadata.get("height"), metadata.get("width"))
            if image.shape != expected_shape:
                self.error(
                    "depth_dimensions",
                    f"Depth PNG shape is {image.shape}, manifest declares {expected_shape}.",
                    episode=episode_index,
                    frame=expected_frame,
                    camera=name,
                    path=self._relative(path),
                )
            if calibrations[name].get("aligned_to_color") and rgb_key in self.features:
                rgb_hw = self._image_hw(self.features[rgb_key])
                if rgb_hw and image.shape != rgb_hw:
                    self.error(
                        "aligned_depth_dimensions",
                        f"Aligned depth shape {image.shape} does not match RGB shape {rgb_hw}.",
                        episode=episode_index,
                        frame=expected_frame,
                        camera=name,
                    )
            if not np.any(image):
                all_zero_counts[name] += 1

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)


def validate_dataset(
    dataset_root: Path,
    *,
    require_depth: bool = False,
    require_eef: bool = False,
    decode_videos: bool = True,
) -> dict[str, Any]:
    return DatasetValidator(
        dataset_root,
        require_depth=require_depth,
        require_eef=require_eef,
        decode_videos=decode_videos,
    ).validate()


def format_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    status = "PASS" if report["valid"] else "FAIL"
    lines = [
        f"=== Dataset Validation: {status} ===",
        f"Root: {report['dataset_root']}",
        (
            f"Episodes: {summary['episodes']} | Frames: {summary['frames']} | "
            f"Videos: {summary['video_files']} files / {summary['decoded_video_frames']} decoded frames"
        ),
        (
            f"Depth: {summary['depth_pngs']} PNGs, cameras={summary['depth_cameras']} | "
            f"EEF fields: {summary['eef_fields']}"
        ),
        f"Errors: {len(report['errors'])} | Warnings: {len(report['warnings'])}",
    ]
    for label, issues in (("ERROR", report["errors"]), ("WARNING", report["warnings"])):
        for issue in issues:
            location = ", ".join(
                f"{key}={value}" for key, value in issue.items() if key not in {"code", "message"}
            )
            suffix = f" ({location})" if location else ""
            lines.append(f"[{label}] {issue['code']}: {issue['message']}{suffix}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only integrity and format validation for a recorded LeRobot v3 dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset repo id or filesystem path, for example datas/demo_init_003.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional base directory used when --dataset is a repo id.",
    )
    parser.add_argument(
        "--require-depth",
        action="store_true",
        help="Fail if committed episodes do not have complete depth sidecars.",
    )
    parser.add_argument(
        "--require-eef",
        action="store_true",
        help="Fail unless both observation.eef_pose and action.eef_pose are present.",
    )
    parser.add_argument(
        "--skip-video-decode",
        action="store_true",
        help="Only check video references and files; skip full frame decoding.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        dataset_root = resolve_dataset_root(args.dataset, args.root)
        report = validate_dataset(
            dataset_root,
            require_depth=args.require_depth,
            require_eef=args.require_eef,
            decode_videos=not args.skip_video_decode,
        )
    except Exception as exc:
        report = {
            "valid": False,
            "dataset_root": str(args.dataset),
            "summary": {
                "episodes": 0,
                "frames": 0,
                "video_files": 0,
                "decoded_video_frames": 0,
                "depth_cameras": [],
                "depth_pngs": 0,
                "eef_fields": [],
            },
            "errors": [{"code": "validation_failed", "message": str(exc)}],
            "warnings": [],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_report(report))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

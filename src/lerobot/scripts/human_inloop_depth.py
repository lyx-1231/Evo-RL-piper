"""Optional Orbbec RGB-D capture and lossless depth sidecars for human-in-loop recording."""

import json
import logging
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Condition
from types import FunctionType

import cv2
import numpy as np

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.utils import get_cv2_rotation


@dataclass
class DepthRecordConfig:
    enable: bool = False
    cameras: list[str] = field(default_factory=lambda: ["head", "wrist"])
    serials: dict[str, str] = field(default_factory=dict)
    width: int = 640
    height: int = 480
    fps: int = 30
    align_to_color: bool = True
    timeout_s: float = 3.0

    def __post_init__(self):
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("`depth.cameras` must contain distinct camera names.")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in self.cameras):
            raise ValueError("Depth camera names may only contain letters, numbers, underscores and hyphens.")
        if set(self.serials) - set(self.cameras):
            raise ValueError("`depth.serials` contains names missing from `depth.cameras`.")
        if any(not serial.strip() for serial in self.serials.values()):
            raise ValueError("Depth camera serial numbers cannot be empty.")
        if (
            min(self.width, self.height, self.fps) <= 0
            or not np.isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("Depth width, height, fps and timeout_s must be positive.")


def serial_from_video_path(index_or_path, sys_video_root: Path = Path("/sys/class/video4linux")) -> str:
    path = Path(f"/dev/video{index_or_path}") if isinstance(index_or_path, int) else Path(index_or_path)
    device = (sys_video_root / path.resolve().name / "device").resolve()
    for parent in (device, *device.parents):
        vendor_path = parent / "idVendor"
        serial_path = parent / "serial"
        if vendor_path.is_file() and serial_path.is_file():
            if vendor_path.read_text().strip().lower() != "2bc5":
                raise ValueError(f"{path} is not an Orbbec USB camera.")
            serial = serial_path.read_text().strip()
            if serial:
                return serial
    raise ValueError(
        f"Cannot find the Orbbec serial for {path}. Check the RGB path or specify "
        "`--depth.serials='{head: HEAD_SERIAL, wrist: WRIST_SERIAL}'`."
    )


def _intrinsics(profile) -> dict:
    intrinsic = profile.get_intrinsic()
    return {name: getattr(intrinsic, name) for name in ("width", "height", "fx", "fy", "cx", "cy")}


class OrbbecRGBDCamera:
    def __init__(self, sdk, device, camera_config, depth_config: DepthRecordConfig, serial: str):
        self.sdk = sdk
        self.device = device
        self.config = camera_config
        self.depth_config = depth_config
        self.serial = serial
        self.width = camera_config.width
        self.height = camera_config.height
        self.fps = camera_config.fps
        self.pipeline = None
        self.align_filter = None
        self.is_connected = False
        self.condition = Condition()
        self.latest = None
        self.consumed = None
        self.sequence = 0
        self.consumed_sequence = 0
        self.error = None
        self.calibration = {}

    def connect(self, warmup: bool = True):
        del warmup
        if self.is_connected:
            raise RuntimeError(f"Orbbec {self.serial} is already connected.")
        self.latest = self.consumed = self.error = None
        self.sequence = self.consumed_sequence = 0
        sdk = self.sdk
        self.pipeline = sdk.Pipeline(self.device)
        config = sdk.Config()
        config.disable_all_stream()
        width, height = self.width, self.height
        if self.config.rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
            width, height = height, width
        try:
            color_profile = self.pipeline.get_stream_profile_list(
                sdk.OBSensorType.COLOR_SENSOR
            ).get_video_stream_profile(width, height, sdk.OBFormat.MJPG, self.fps)
            depth_profile = self.pipeline.get_stream_profile_list(
                sdk.OBSensorType.DEPTH_SENSOR
            ).get_video_stream_profile(
                self.depth_config.width, self.depth_config.height, sdk.OBFormat.Y16, self.depth_config.fps
            )
            self.calibration = {
                "color_intrinsics": _intrinsics(color_profile),
                "depth_intrinsics": _intrinsics(depth_profile),
                "native_depth_intrinsics": _intrinsics(depth_profile),
                "color_rotation": self.config.rotation.value,
                "depth_rotation": self.config.rotation.value if self.depth_config.align_to_color else 0,
                "aligned_to_color": self.depth_config.align_to_color,
                "alignment_mode": "software" if self.depth_config.align_to_color else "none",
            }
            config.enable_stream(color_profile)
            config.enable_stream(depth_profile)
            config.set_align_mode(sdk.OBAlignMode.DISABLE)
            if self.depth_config.align_to_color:
                self.align_filter = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
                self.align_filter.set_match_target_resolution(True)
            config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            self.pipeline.enable_frame_sync()
            self.pipeline.start(config, self._on_frames)
            self.is_connected = True
            self.async_read()
        except Exception as exc:
            self.disconnect()
            raise RuntimeError(
                f"Orbbec {self.serial}: cannot start RGB {width}x{height}@{self.fps} and depth "
                f"{self.depth_config.width}x{self.depth_config.height}@{self.depth_config.fps}. "
                f"Check supported profiles, USB bandwidth and camera ownership: {exc}"
            ) from exc
        logging.info(
            "Orbbec %s connected with RGB and uint16 depth; depth-to-color alignment: %s.",
            self.serial,
            self.calibration["alignment_mode"],
        )

    def _on_frames(self, frames):
        host_monotonic_s = time.monotonic()
        try:
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if color is None or depth is None:
                return
            if self.depth_config.align_to_color:
                if self.align_filter is None:
                    raise RuntimeError("Orbbec software depth-to-color alignment is not initialized.")
                aligned = self.align_filter.process(frames)
                if aligned is None:
                    return
                frames = aligned.as_frame_set()
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if color is None or depth is None:
                    return
            bgr = cv2.imdecode(np.frombuffer(color.get_data(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Cannot decode Orbbec MJPEG color frame.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if self.config.color_mode == ColorMode.RGB else bgr
            rotation = get_cv2_rotation(self.config.rotation)
            if rotation is not None:
                rgb = cv2.rotate(rgb, rotation)
            if rgb.shape != (self.height, self.width, 3):
                raise ValueError(f"Unexpected Orbbec RGB shape: {rgb.shape}.")
            depth_image = (
                np.frombuffer(depth.get_data(), dtype=np.uint16)
                .reshape(depth.get_height(), depth.get_width())
                .copy()
            )
            if self.depth_config.align_to_color:
                if rotation is not None:
                    depth_image = cv2.rotate(depth_image, rotation)
                if depth_image.shape != rgb.shape[:2]:
                    raise ValueError(
                        f"Aligned depth shape {depth_image.shape} does not match RGB {rgb.shape[:2]}."
                    )
                if self.sequence == 0:
                    self.calibration["depth_intrinsics"] = _intrinsics(
                        depth.get_stream_profile().as_video_stream_profile()
                    )
            scale = float(depth.get_depth_scale())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid depth scale: {scale}.")
            sample = {
                "color": rgb,
                "depth": depth_image,
                "metadata": {
                    "serial": self.serial,
                    "depth_scale_mm": scale,
                    "color_timestamp_ms": float(color.get_timestamp()),
                    "depth_timestamp_ms": float(depth.get_timestamp()),
                    "host_monotonic_s": host_monotonic_s,
                    "aligned_to_color": self.depth_config.align_to_color,
                    "alignment_mode": "software" if self.depth_config.align_to_color else "none",
                },
            }
            with self.condition:
                self.latest = sample
                self.sequence += 1
                self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def async_read(self, timeout_ms: float | None = None):
        timeout = self.depth_config.timeout_s if timeout_ms is None else timeout_ms / 1000
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error is not None or self.sequence > self.consumed_sequence,
                timeout=timeout,
            )
            if self.error is not None:
                raise RuntimeError(f"Orbbec {self.serial}: RGB-D capture failed.") from self.error
            if not ready:
                raise TimeoutError(f"Orbbec {self.serial}: no fresh RGB-D frame within {timeout}s.")
            self.consumed = self.latest
            self.consumed_sequence = self.sequence
            return self.consumed["color"]

    def disconnect(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                logging.exception("Could not stop Orbbec pipeline %s.", self.serial)
            finally:
                self.pipeline = None
                self.is_connected = False
                self.align_filter = None


def _write_depth(path: Path, depth: np.ndarray):
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError("Depth must be a uint16 H x W array.")
    if not cv2.imwrite(str(path), depth, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(f"Cannot write depth image: {path}")


class DepthEpisodeWriter:
    def __init__(self, root: Path, cameras: dict[str, OrbbecRGBDCamera]):
        self.root = Path(root) / "depth"
        self.cameras = cameras
        self.executor = ThreadPoolExecutor(max_workers=len(cameras), thread_name_prefix="depth-writer")
        self.pending = deque()
        self.temporary = None
        self.stage = None
        self.manifest = None
        self.destination = None
        self.count = 0

    def add_frame(self, episode_index: int, frame_index: int, timestamp: float):
        if self.temporary is None:
            self.root.mkdir(parents=True, exist_ok=True)
            self.destination = self.root / f"episode_{episode_index:06d}"
            if self.destination.exists():
                raise FileExistsError(f"Refusing to overwrite depth episode: {self.destination}")
            self.temporary = TemporaryDirectory(prefix=".pending_", dir=self.root)
            self.stage = Path(self.temporary.name)
            self.manifest = (self.stage / "frames.jsonl").open("w", encoding="utf-8")
            for name in self.cameras:
                (self.stage / name).mkdir()
            with (self.stage / "cameras.json").open("w", encoding="utf-8") as stream:
                json.dump(
                    {name: {"serial": cam.serial, **cam.calibration} for name, cam in self.cameras.items()},
                    stream,
                    indent=2,
                    allow_nan=False,
                )
        if frame_index != self.count:
            raise ValueError(f"Depth frame index {frame_index} does not match expected index {self.count}.")
        camera_metadata = {}
        for name, camera in self.cameras.items():
            sample = camera.consumed
            if sample is None:
                raise RuntimeError(f"No RGB-D observation has been read from {name}.")
            relative = Path(name) / f"frame_{frame_index:06d}.png"
            while len(self.pending) >= 8 * len(self.cameras):
                self.pending.popleft().result()
            self.pending.append(self.executor.submit(_write_depth, self.stage / relative, sample["depth"]))
            camera_metadata[name] = {
                **sample["metadata"],
                "path": relative.as_posix(),
                "rgb_key": f"observation.images.{name}",
                "height": sample["depth"].shape[0],
                "width": sample["depth"].shape[1],
            }
        self.manifest.write(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "cameras": camera_metadata,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self.count += 1

    def flush(self):
        while self.pending:
            self.pending.popleft().result()
        if self.manifest is not None:
            self.manifest.flush()

    def commit(self):
        self.flush()
        if self.stage is not None:
            if self.destination.exists():
                raise FileExistsError(f"Refusing to overwrite depth episode: {self.destination}")
            self.manifest.close()
            self.manifest = None
            self.stage.rename(self.destination)
        self.discard()

    def discard(self):
        try:
            self.flush()
        finally:
            for future in self.pending:
                try:
                    future.result()
                except Exception:
                    logging.exception("Depth image writer failed during cleanup.")
            self.pending.clear()
            if self.manifest is not None:
                self.manifest.close()
            if self.temporary is not None:
                self.temporary.cleanup()
            self.manifest = self.temporary = self.stage = None
            self.count = 0

    def preserve_failed_episode(self):
        if self.stage is None:
            return
        recovery_path = (
            self.root / f"failed_{self.destination.name}_{self.stage.name.removeprefix('.pending_')}"
        )
        self.destination = recovery_path
        self.commit()
        logging.error(
            "Episode save failed. Depth frames are preserved at %s; this is not a committed dataset episode.",
            recovery_path,
        )

    def close(self):
        try:
            self.discard()
        finally:
            self.executor.shutdown(wait=True)


class DepthDatasetMixin:
    depth_cameras: dict[str, OrbbecRGBDCamera]
    _depth_writer = None
    _saving_depth = False

    def add_frame(self, frame):
        if self._depth_writer is None:
            self._depth_writer = DepthEpisodeWriter(self.root, self.depth_cameras)
        buffer = self.episode_buffer
        episode_index = self.num_episodes if buffer is None else buffer["episode_index"]
        frame_index = 0 if buffer is None else buffer["size"]
        timestamp = frame.get("timestamp", frame_index / self.fps)
        self._depth_writer.add_frame(episode_index, frame_index, timestamp)
        return super().add_frame(frame)

    def save_episode(self, episode_data=None, **kwargs):
        if episode_data is not None:
            raise ValueError("Depth recording only supports the current episode buffer.")
        if self._depth_writer is None or self._depth_writer.count != self.episode_buffer["size"]:
            raise RuntimeError("RGB and depth frame counts do not match.")
        self._depth_writer.flush()
        self._saving_depth = True
        try:
            result = super().save_episode(**kwargs)
            self._depth_writer.commit()
        except Exception:
            try:
                self._depth_writer.preserve_failed_episode()
            except Exception:
                logging.exception("Could not preserve depth frames after episode save failure.")
            raise
        finally:
            self._saving_depth = False
        return result

    def clear_episode_buffer(self, *args, **kwargs):
        if not self._saving_depth and self._depth_writer is not None:
            self._depth_writer.discard()
        return super().clear_episode_buffer(*args, **kwargs)

    def finalize(self):
        try:
            return super().finalize()
        finally:
            if self._depth_writer is not None:
                self._depth_writer.close()
                self._depth_writer = None


def record_with_depth(cfg, record_function):
    """Use call-local factories without changing the shared recorder or its module globals."""
    try:
        import pyorbbecsdk as sdk
    except ImportError as exc:
        raise ImportError(
            "Depth recording requires Orbbec SDK v2. Install it in the recording environment with "
            "`python -m pip install pyorbbecsdk2`."
        ) from exc

    camera_configs = getattr(cfg.robot, "cameras", {})
    missing = set(cfg.depth.cameras) - set(camera_configs)
    if missing:
        raise ValueError(f"Depth cameras are missing from robot.cameras: {sorted(missing)}")
    serials = {}
    for name in cfg.depth.cameras:
        camera_config = camera_configs[name]
        if camera_config.type != "opencv":
            raise ValueError(f"Depth camera {name} requires an OpenCV RGB camera configuration.")
        if not all((camera_config.width, camera_config.height, camera_config.fps)):
            raise ValueError(f"Specify RGB width, height and fps for depth camera {name}.")
        serials[name] = cfg.depth.serials.get(name) or serial_from_video_path(camera_config.index_or_path)
    if len(set(serials.values())) != len(serials):
        raise ValueError("Each depth view must refer to a different Orbbec camera serial.")

    context = sdk.Context()
    devices = context.query_devices()
    cameras = {}
    for name, serial in serials.items():
        device = devices.get_device_by_serial_number(serial)
        if device is None:
            raise ValueError(f"Orbbec camera {name} with serial {serial} was not found.")
        cameras[name] = OrbbecRGBDCamera(sdk, device, camera_configs[name], cfg.depth, serial)
    dataset_class = record_function.__globals__["LeRobotDataset"]
    robot_factory = record_function.__globals__["make_robot_from_config"]

    class DepthDataset(DepthDatasetMixin, dataset_class):
        depth_cameras = cameras

    active_robot = None

    def make_depth_robot(robot_config):
        nonlocal active_robot
        active_robot = robot_factory(robot_config)
        active_robot.cameras.update(cameras)
        return active_robot

    local_globals = {
        **record_function.__globals__,
        "LeRobotDataset": DepthDataset,
        "make_robot_from_config": make_depth_robot,
    }
    depth_record = FunctionType(
        record_function.__code__,
        local_globals,
        record_function.__name__,
        record_function.__defaults__,
        record_function.__closure__,
    )
    depth_record.__kwdefaults__ = record_function.__kwdefaults__
    try:
        return depth_record(cfg)
    finally:
        try:
            if active_robot is not None and active_robot.is_connected:
                active_robot.disconnect()
        finally:
            for camera in cameras.values():
                camera.disconnect()

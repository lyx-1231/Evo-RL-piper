import json
import sys
from types import SimpleNamespace

import av
import cv2
import draccus
import numpy as np
import pytest

from lerobot.cameras.opencv.configuration_opencv import Cv2Rotation, OpenCVCameraConfig
from lerobot.cameras.utils import get_cv2_rotation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_audio_info, get_video_info
from lerobot.robots import make_robot_from_config
from lerobot.scripts import human_inloop_depth, lerobot_human_inloop_record
from lerobot.scripts.human_inloop_depth import (
    DepthDatasetMixin,
    DepthRecordConfig,
    OrbbecRGBDCamera,
    record_with_depth,
    serial_from_video_path,
)
from lerobot.scripts.lerobot_human_inloop_record import HumanInloopRecordConfig


def parse_config(*extra):
    return draccus.parse(
        HumanInloopRecordConfig,
        args=[
            "--robot.type=piper_follower",
            "--robot.port=FOLLOWER",
            "--robot.cameras={head: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 14, width: 640, height: 480, fps: 30}}",
            "--teleop.type=piper_leader",
            "--teleop.port=LEADER",
            "--dataset.repo_id=local/test",
            "--dataset.single_task=test",
            "--dataset.push_to_hub=false",
            *extra,
        ],
    )


def sample(value=1000):
    return {
        "color": np.zeros((2, 3, 3), dtype=np.uint8),
        "depth": np.array([[0, 1, 256], [value, 42000, 65535]], dtype=np.uint16),
        "metadata": {
            "serial": "SERIAL",
            "depth_scale_mm": 0.1,
            "color_timestamp_ms": 50.0,
            "depth_timestamp_ms": 49.0,
            "aligned_to_color": False,
        },
    }


def make_camera(value=1000):
    return SimpleNamespace(serial="SERIAL", calibration={}, consumed=sample(value))


def create_dataset(tmp_path, cameras, use_videos=False):
    class DepthDataset(DepthDatasetMixin, LeRobotDataset):
        depth_cameras = cameras

    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        **{
            f"observation.images.{name}": {
                "dtype": "video" if use_videos else "image",
                "shape": camera.consumed["color"].shape,
                "names": ["height", "width", "channels"],
            }
            for name, camera in cameras.items()
        },
    }
    return DepthDataset.create(
        "local/test",
        fps=30,
        root=tmp_path / "dataset",
        features=features,
        use_videos=use_videos,
        vcodec="h264",
    )


def add_frame(dataset):
    dataset.add_frame(
        {
            "observation.state": np.array([1], dtype=np.float32),
            "action": np.array([2], dtype=np.float32),
            "task": "test",
            **{
                f"observation.images.{name}": cam.consumed["color"]
                for name, cam in dataset.depth_cameras.items()
            },
        }
    )


def test_cli_depth_is_optional_and_parses_camera_selection():
    default_config = parse_config()
    assert not default_config.depth.enable
    assert default_config.depth.align_to_color
    assert (default_config.depth.width, default_config.depth.height, default_config.depth.fps) == (
        640,
        480,
        30,
    )
    assert default_config.depth.hole_filling_cameras == ["head"]
    assert default_config.depth.hole_filling_mode == "FURTHEST"
    assert not parse_config("--depth.align_to_color=false").depth.align_to_color
    config = parse_config("--depth.enable=true", "--depth.cameras=[wrist]", "--depth.serials={wrist: SERIAL}")
    assert config.depth.enable
    assert config.depth.cameras == ["wrist"]
    assert config.depth.serials == {"wrist": "SERIAL"}
    assert config.depth.hole_filling_cameras == []
    assert config.depth.hole_filling_mode == "FURTHEST"
    assert parse_config("--depth.hole_filling_cameras=[]").depth.hole_filling_cameras == []
    assert parse_config("--depth.hole_filling_cameras=[head]").depth.hole_filling_cameras == ["head"]


def test_disabled_depth_does_not_load_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)
    sentinel = object()
    configs = []

    def ordinary_record(config):
        configs.append(config)
        return sentinel

    monkeypatch.setattr(lerobot_human_inloop_record.record, "__wrapped__", ordinary_record)
    assert lerobot_human_inloop_record.human_inloop_record(parse_config()) is sentinel
    assert configs[0].enable_episode_outcome_labeling


def test_missing_sdk_has_install_instruction(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)
    with pytest.raises(ImportError, match="pip install pyorbbecsdk2"):
        record_with_depth(parse_config("--depth.enable=true"), lambda config: None)


@pytest.mark.parametrize(
    "options",
    [
        {"cameras": ["head", "head"]},
        {"cameras": ["../head"]},
        {"serials": {"unknown": "SERIAL"}},
        {"fps": 0},
        {"timeout_s": float("nan")},
        {"hole_filling_cameras": ["head", "head"]},
        {"hole_filling_cameras": ["../head"]},
        {"hole_filling_cameras": ["unknown"]},
        {"hole_filling_mode": "other"},
    ],
)
def test_invalid_depth_options(options):
    with pytest.raises(ValueError):
        DepthRecordConfig(**options)


def test_resolve_serial_from_rgb_usb_path(tmp_path):
    usb = tmp_path / "usb" / "camera"
    interface = usb / "interface"
    interface.mkdir(parents=True)
    (usb / "idVendor").write_text("2bc5\n")
    (usb / "serial").write_text("ORBBEC_SERIAL\n")
    sys_video = tmp_path / "video4linux"
    (sys_video / "video14").mkdir(parents=True)
    (sys_video / "video14" / "device").symlink_to(interface, target_is_directory=True)
    video_path = tmp_path / "by-path"
    video_path.symlink_to("/dev/video14")
    assert serial_from_video_path(video_path, sys_video) == "ORBBEC_SERIAL"
    assert serial_from_video_path(14, sys_video) == "ORBBEC_SERIAL"
    (usb / "idVendor").write_text("0000")
    with pytest.raises(ValueError, match="not an Orbbec"):
        serial_from_video_path(video_path, sys_video)


@pytest.mark.parametrize("names", [["wrist"], ["head", "wrist"]])
def test_lossless_depth_and_manifest_match_saved_episode(tmp_path, names):
    cameras = {name: make_camera() for name in names}
    dataset = create_dataset(tmp_path, cameras)
    try:
        add_frame(dataset)
        for camera in cameras.values():
            camera.consumed = sample(2000)
        add_frame(dataset)
        dataset.save_episode(extra_episode_metadata={"episode_success": "success"})
        episode = dataset.root / "depth" / "episode_000000"
        frames = [json.loads(line) for line in (episode / "frames.jsonl").read_text().splitlines()]
        assert [frame["frame_index"] for frame in frames] == [0, 1]
        assert frames[1]["timestamp"] == 1 / 30
        assert frames[0]["episode_index"] == 0
        assert set(frames[0]["cameras"]) == set(names)
        for name in names:
            for index, value in enumerate((1000, 2000)):
                metadata = frames[index]["cameras"][name]
                image = cv2.imread(str(episode / metadata["path"]), cv2.IMREAD_UNCHANGED)
                assert image.dtype == np.uint16
                np.testing.assert_array_equal(image, sample(value)["depth"])
                assert metadata["depth_scale_mm"] == 0.1
                assert metadata["rgb_key"] == f"observation.images.{name}"
        assert dataset.num_episodes == 1
        assert not list((dataset.root / "depth").glob(".pending_*"))
    finally:
        dataset.finalize()


def test_rerecord_discards_depth_and_restarts_frame_numbers(tmp_path):
    dataset = create_dataset(tmp_path, {"wrist": make_camera()})
    try:
        add_frame(dataset)
        add_frame(dataset)
        dataset.clear_episode_buffer()
        assert not list((dataset.root / "depth").iterdir())
        dataset.depth_cameras["wrist"].consumed = sample(3000)
        add_frame(dataset)
        dataset.save_episode()
        episode = dataset.root / "depth" / "episode_000000"
        assert len(list((episode / "wrist").glob("*.png"))) == 1
        image = cv2.imread(str(episode / "wrist/frame_000000.png"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(image, sample(3000)["depth"])
    finally:
        dataset.finalize()


def test_resume_preserves_previous_depth_episode(tmp_path):
    dataset = create_dataset(tmp_path, {"wrist": make_camera()})
    add_frame(dataset)
    dataset.save_episode()
    dataset.finalize()
    first = dataset.root / "depth/episode_000000/wrist/frame_000000.png"
    original = first.read_bytes()
    resumed = type(dataset)("local/test", root=dataset.root)
    try:
        resumed.depth_cameras["wrist"].consumed = sample(3000)
        add_frame(resumed)
        resumed.save_episode()
        assert first.read_bytes() == original
        manifest = resumed.root / "depth/episode_000001/frames.jsonl"
        assert json.loads(manifest.read_text())["episode_index"] == 1
        assert resumed.num_episodes == 2
    finally:
        resumed.finalize()


def test_finalize_discards_unsaved_depth(tmp_path):
    dataset = create_dataset(tmp_path, {"wrist": make_camera()})
    add_frame(dataset)
    dataset.finalize()
    assert not list((dataset.root / "depth").iterdir())


def test_depth_write_failure_prevents_episode_save(tmp_path, monkeypatch):
    dataset = create_dataset(tmp_path, {"wrist": make_camera()})

    def fail_write(path, depth):
        raise OSError("disk full")

    monkeypatch.setattr(human_inloop_depth, "_write_depth", fail_write)
    try:
        add_frame(dataset)
        with pytest.raises(OSError, match="disk full"):
            dataset.save_episode()
        assert dataset.num_episodes == 0
        assert not (dataset.root / "depth/episode_000000").exists()
    finally:
        dataset.finalize()


@pytest.mark.parametrize("camera_names", [["wrist"], ["head", "wrist"]])
@pytest.mark.parametrize("episode_count", [1, 2])
def test_h264_video_and_depth_episode_save(tmp_path, camera_names, episode_count):
    cameras = {name: make_camera() for name in camera_names}
    for camera in cameras.values():
        camera.consumed["color"] = np.full((32, 32, 3), 100, dtype=np.uint8)
    dataset = create_dataset(tmp_path, cameras, use_videos=True)
    expected_colors = []
    total_frames = sum(3 + episode_index for episode_index in range(episode_count))
    try:
        for episode_index in range(episode_count):
            color_value = 100 + 50 * episode_index
            for camera in cameras.values():
                camera.consumed = sample(1000 * (episode_index + 1))
                camera.consumed["color"] = np.full((32, 32, 3), color_value, dtype=np.uint8)
            for _ in range(3 + episode_index):
                add_frame(dataset)
                expected_colors.append(color_value)
            dataset.save_episode(extra_episode_metadata={"episode_success": "success"})
        assert dataset.num_episodes == episode_count
        assert dataset.num_frames == total_frames
        for name in camera_names:
            video_path = dataset.root / f"videos/observation.images.{name}/chunk-000/file-000.mp4"
            info = get_video_info(video_path)
            assert info["video.codec"] == "h264"
            assert info["video.width"] == info["video.height"] == 32
            assert info["video.fps"] == 30
            assert not info["has_audio"]
            with av.open(str(video_path)) as video:
                frames = list(video.decode(video=0))
                assert len(frames) == total_frames
                np.testing.assert_allclose(
                    [frame.time for frame in frames], np.arange(total_frames) / 30, atol=1e-4
                )
                np.testing.assert_allclose(
                    [frame.to_ndarray(format="rgb24").mean() for frame in frames], expected_colors, atol=5
                )
            for episode_index in range(episode_count):
                episode_path = dataset.root / "depth" / f"episode_{episode_index:06d}"
                assert len(list((episode_path / name).glob("*.png"))) == 3 + episode_index
                manifest = [
                    json.loads(line) for line in (episode_path / "frames.jsonl").read_text().splitlines()
                ]
                assert [frame["frame_index"] for frame in manifest] == list(range(3 + episode_index))
                assert all(frame["episode_index"] == episode_index for frame in manifest)
                depth = cv2.imread(str(episode_path / name / "frame_000000.png"), cv2.IMREAD_UNCHANGED)
                assert depth.dtype == np.uint16
                np.testing.assert_array_equal(depth, sample(1000 * (episode_index + 1))["depth"])
        assert not list((dataset.root / "depth").glob("failed_*"))
    finally:
        dataset.finalize()
    info = json.loads((dataset.root / "meta/info.json").read_text())
    assert info["total_episodes"] == episode_count
    assert info["total_frames"] == total_frames
    assert (dataset.root / "meta/episodes/chunk-000/file-000.parquet").is_file()


def test_video_save_failure_preserves_depth_on_finalize(tmp_path, monkeypatch):
    dataset = create_dataset(tmp_path, {"wrist": make_camera()})

    def fail_save(*args, **kwargs):
        raise AttributeError("canonical_name")

    monkeypatch.setattr(LeRobotDataset, "save_episode", fail_save)
    add_frame(dataset)
    with pytest.raises(AttributeError, match="canonical_name"):
        dataset.save_episode()
    dataset.finalize()
    assert dataset.num_episodes == 0
    assert not (dataset.root / "depth/episode_000000").exists()
    failed = list((dataset.root / "depth").glob("failed_episode_000000_*"))
    assert len(failed) == 1
    depth = cv2.imread(str(failed[0] / "wrist/frame_000000.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(depth, sample()["depth"])
    assert json.loads((failed[0] / "frames.jsonl").read_text())["frame_index"] == 0


@pytest.mark.parametrize("canonical_name", [None, "canonical_codec"])
def test_codec_metadata_with_old_and_new_pyav(monkeypatch, canonical_name):
    from unittest.mock import MagicMock

    codec = SimpleNamespace(name="h264")
    if canonical_name is not None:
        codec.canonical_name = canonical_name
    stream = SimpleNamespace(
        codec=codec,
        height=32,
        width=32,
        pix_fmt="yuv420p",
        base_rate=30,
        channels=1,
        bit_rate=128000,
        sample_rate=16000,
        format=SimpleNamespace(bits=16),
        layout=SimpleNamespace(name="mono"),
    )
    container = MagicMock()
    container.__enter__.return_value.streams = SimpleNamespace(video=[stream], audio=[stream])
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: container)
    assert get_video_info("video.mp4")["video.codec"] == (canonical_name or "h264")
    assert get_audio_info("video.mp4")["audio.codec"] == (canonical_name or "h264")


def test_existing_depth_is_never_overwritten(tmp_path):
    dataset = create_dataset(tmp_path, {"wrist": make_camera()})
    destination = dataset.root / "depth/episode_000000"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("existing")
    try:
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            add_frame(dataset)
        assert marker.read_text() == "existing"
    finally:
        dataset.finalize()


def fake_frames(value=1000):
    success, encoded = cv2.imencode(".jpg", sample()["color"])
    assert success
    profile = SimpleNamespace(
        get_intrinsic=lambda: SimpleNamespace(width=3, height=2, fx=2.0, fy=2.0, cx=1.0, cy=1.0)
    )
    frames = SimpleNamespace(
        get_color_frame=lambda: SimpleNamespace(get_data=lambda: encoded, get_timestamp=lambda: 50),
        get_depth_frame=lambda: SimpleNamespace(
            get_data=lambda: sample(value)["depth"].tobytes(),
            get_height=lambda: 2,
            get_width=lambda: 3,
            get_depth_scale=lambda: 0.1,
            get_timestamp=lambda: 49,
            get_stream_profile=lambda: SimpleNamespace(as_video_stream_profile=lambda: profile),
        ),
    )
    frames.as_frame_set = lambda: frames
    return frames


def test_callback_does_not_replace_depth_for_consumed_rgb():
    config = OpenCVCameraConfig(index_or_path=0, width=3, height=2, fps=30)
    camera = OrbbecRGBDCamera(None, None, config, DepthRecordConfig(align_to_color=False), "SERIAL")
    camera._on_frames(fake_frames(1000))
    camera.async_read()
    camera._on_frames(fake_frames(2000))
    np.testing.assert_array_equal(camera.consumed["depth"], sample(1000)["depth"])
    camera.async_read()
    np.testing.assert_array_equal(camera.consumed["depth"], sample(2000)["depth"])
    with pytest.raises(TimeoutError, match="no fresh RGB-D"):
        camera.async_read(timeout_ms=1)


def test_malformed_camera_frame_fails_explicitly():
    config = OpenCVCameraConfig(index_or_path=0, width=3, height=2, fps=30)
    camera = OrbbecRGBDCamera(None, None, config, DepthRecordConfig(align_to_color=False), "SERIAL")
    frame = fake_frames()
    frame.get_depth_frame = lambda: SimpleNamespace(
        get_data=lambda: b"bad",
        get_height=lambda: 2,
        get_width=lambda: 3,
    )
    camera._on_frames(frame)
    with pytest.raises(RuntimeError, match="capture failed"):
        camera.async_read()


def test_record_factories_are_local_and_cameras_close_on_error(monkeypatch):
    config = parse_config(
        "--depth.enable=true",
        "--depth.serials={head: HEAD, wrist: WRIST}",
    )
    devices = SimpleNamespace(get_device_by_serial_number=lambda serial: serial)
    monkeypatch.setitem(
        sys.modules,
        "pyorbbecsdk",
        SimpleNamespace(
            Context=lambda: SimpleNamespace(query_devices=lambda: devices),
        ),
    )
    closed = []
    monkeypatch.setattr(OrbbecRGBDCamera, "disconnect", lambda self: closed.append(self.serial))

    def fake_robot_factory(config):
        return SimpleNamespace(
            cameras={name: object() for name in config.cameras},
            is_connected=True,
            disconnect=lambda: closed.append("ROBOT"),
        )

    def fake_record(config):
        robot = make_robot_from_config(config.robot)
        assert isinstance(robot.cameras["head"], OrbbecRGBDCamera)
        assert robot.cameras["head"].hole_filling_mode == "FURTHEST"
        assert robot.cameras["wrist"].hole_filling_mode is None
        assert issubclass(LeRobotDataset, DepthDatasetMixin)
        raise RuntimeError("record failure")

    monkeypatch.setitem(fake_record.__globals__, "make_robot_from_config", fake_robot_factory)
    original_dataset_class = fake_record.__globals__["LeRobotDataset"]
    with pytest.raises(RuntimeError, match="record failure"):
        record_with_depth(config, fake_record)
    assert fake_record.__globals__["LeRobotDataset"] is original_dataset_class
    assert fake_record.__globals__["make_robot_from_config"] is fake_robot_factory
    assert sorted(closed) == ["HEAD", "ROBOT", "WRIST"]


def test_missing_device_does_not_fall_back_to_first_camera(monkeypatch):
    config = parse_config("--depth.enable=true", "--depth.serials={head: HEAD, wrist: WRIST}")
    devices = SimpleNamespace(get_device_by_serial_number=lambda serial: None)
    monkeypatch.setitem(
        sys.modules,
        "pyorbbecsdk",
        SimpleNamespace(
            Context=lambda: SimpleNamespace(query_devices=lambda: devices),
        ),
    )
    with pytest.raises(ValueError, match="head with serial HEAD was not found"):
        record_with_depth(config, lambda config: None)


@pytest.mark.parametrize("fail_start", [False, True])
@pytest.mark.parametrize("align_to_color", [False, True])
def test_sdk_stream_setup_and_cleanup(fail_start, align_to_color):
    from unittest.mock import Mock

    sdk = Mock()
    pipeline = sdk.Pipeline.return_value
    profile = pipeline.get_stream_profile_list.return_value.get_video_stream_profile.return_value
    profile.get_intrinsic.return_value = SimpleNamespace(width=3, height=2, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    sdk.AlignFilter.return_value.process.return_value = fake_frames(2000)

    def start(config, callback):
        if fail_start:
            raise RuntimeError("unsupported profile")
        callback(fake_frames())

    pipeline.start.side_effect = start
    config = OpenCVCameraConfig(index_or_path=0, width=3, height=2, fps=30)
    camera = OrbbecRGBDCamera(
        sdk, object(), config, DepthRecordConfig(width=3, height=2, align_to_color=align_to_color), "SERIAL"
    )
    if fail_start:
        with pytest.raises(RuntimeError, match="unsupported profile"):
            camera.connect()
    else:
        camera.connect()
        assert camera.is_connected
        np.testing.assert_array_equal(
            camera.consumed["depth"], sample(2000 if align_to_color else 1000)["depth"]
        )
        sdk.Config.return_value.disable_all_stream.assert_called_once()
        assert sdk.Config.return_value.enable_stream.call_count == 2
        sdk.Config.return_value.set_align_mode.assert_called_once_with(sdk.OBAlignMode.DISABLE)
        if align_to_color:
            sdk.AlignFilter.assert_called_once_with(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
            sdk.AlignFilter.return_value.set_match_target_resolution.assert_called_once_with(True)
            assert camera.calibration["depth_intrinsics"]["fx"] == 2.0
            assert camera.calibration["native_depth_intrinsics"]["fx"] == 1.0
        else:
            sdk.AlignFilter.assert_not_called()
        assert camera.consumed["metadata"]["aligned_to_color"] == align_to_color
        pipeline.enable_frame_sync.assert_called_once()
        assert camera.calibration["depth_intrinsics"]["width"] == 3
        camera.disconnect()
    pipeline.stop.assert_called_once()
    assert not camera.is_connected
    assert camera.align_filter is None


@pytest.mark.parametrize("align_to_color", [False, True])
@pytest.mark.parametrize("name", ["head", "wrist"])
def test_hole_filling_only_applies_to_selected_camera(tmp_path, align_to_color, name):
    from unittest.mock import Mock

    sdk = Mock()
    pipeline = sdk.Pipeline.return_value
    profile = pipeline.get_stream_profile_list.return_value.get_video_stream_profile.return_value
    profile.get_intrinsic.return_value = SimpleNamespace(width=3, height=2, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
    frames = fake_frames(2000 if align_to_color else 1000)
    depth = frames.get_depth_frame()
    frames.get_depth_frame = lambda: depth
    sdk.AlignFilter.return_value.process.return_value = frames
    sdk.HoleFillingFilter.return_value.process.return_value.as_depth_frame.return_value = fake_frames(
        3000
    ).get_depth_frame()
    pipeline.start.side_effect = lambda config, callback: callback(frames)
    camera = OrbbecRGBDCamera(
        sdk,
        object(),
        OpenCVCameraConfig(index_or_path=0, width=3, height=2, fps=30),
        DepthRecordConfig(align_to_color=align_to_color, hole_filling_cameras=["head"]),
        "SERIAL",
        name=name,
    )
    try:
        camera.connect()
        expected = sample(3000 if name == "head" else (2000 if align_to_color else 1000))["depth"]
        np.testing.assert_array_equal(camera.consumed["depth"], expected)
        if name == "head":
            sdk.HoleFillingFilter.return_value.set_filling_mode.assert_called_once_with(
                sdk.OBHoleFillingMode.FURTHEST
            )
            sdk.HoleFillingFilter.return_value.enable.assert_called_once_with(True)
            sdk.HoleFillingFilter.return_value.process.assert_called_once_with(depth)
        else:
            sdk.HoleFillingFilter.assert_not_called()
        assert camera.consumed["metadata"]["hole_filling_mode"] == camera.hole_filling_mode
        dataset = create_dataset(tmp_path, {name: camera})
        try:
            add_frame(dataset)
            dataset.save_episode()
            episode = dataset.root / "depth/episode_000000"
            saved = cv2.imread(str(episode / name / "frame_000000.png"), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(saved, expected)
            metadata = json.loads((episode / "frames.jsonl").read_text())["cameras"][name]
            assert metadata["hole_filling_mode"] == camera.hole_filling_mode
            assert (
                json.loads((episode / "cameras.json").read_text())[name]["hole_filling_mode"]
                == camera.hole_filling_mode
            )
        finally:
            dataset.finalize()
    finally:
        camera.disconnect()
    assert camera.hole_filling_filter is None


def test_missing_filled_frame_does_not_record_unfiltered_depth():
    from unittest.mock import Mock

    camera = OrbbecRGBDCamera(
        None,
        None,
        OpenCVCameraConfig(index_or_path=0, width=3, height=2, fps=30),
        DepthRecordConfig(align_to_color=False, hole_filling_cameras=["head"]),
        "SERIAL",
        name="head",
    )
    camera.hole_filling_filter = Mock()
    camera.hole_filling_filter.process.return_value = None
    camera._on_frames(fake_frames())
    with pytest.raises(TimeoutError, match="no fresh RGB-D"):
        camera.async_read(timeout_ms=1)
    assert camera.consumed is None


@pytest.mark.parametrize("rotation", list(Cv2Rotation))
def test_software_aligned_depth_rotates_with_saved_rgb(tmp_path, rotation):
    from unittest.mock import Mock

    width, height = (2, 3) if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270) else (3, 2)
    config = OpenCVCameraConfig(index_or_path=0, width=width, height=height, fps=30, rotation=rotation)
    camera = OrbbecRGBDCamera(None, None, config, DepthRecordConfig(), "SERIAL")
    camera.align_filter = Mock()
    camera.align_filter.process.return_value = fake_frames(2000)
    raw_frames = fake_frames(1000)
    camera._on_frames(raw_frames)
    camera.async_read()
    camera.align_filter.process.assert_called_once_with(raw_frames)
    expected = sample(2000)["depth"]
    cv_rotation = get_cv2_rotation(rotation)
    if cv_rotation is not None:
        expected = cv2.rotate(expected, cv_rotation)
    np.testing.assert_array_equal(camera.consumed["depth"], expected)
    assert camera.consumed["depth"].shape == camera.consumed["color"].shape[:2]
    dataset = create_dataset(tmp_path, {"wrist": camera})
    try:
        add_frame(dataset)
        dataset.save_episode()
        episode = dataset.root / "depth/episode_000000"
        saved = cv2.imread(str(episode / "wrist/frame_000000.png"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(saved, expected)
        metadata = json.loads((episode / "frames.jsonl").read_text())["cameras"]["wrist"]
        assert metadata["aligned_to_color"]
        assert metadata["alignment_mode"] == "software"
        assert (metadata["width"], metadata["height"]) == (width, height)
    finally:
        dataset.finalize()


def test_missing_aligned_frame_does_not_save_unaligned_depth():
    from unittest.mock import Mock

    config = OpenCVCameraConfig(index_or_path=0, width=3, height=2, fps=30)
    camera = OrbbecRGBDCamera(None, None, config, DepthRecordConfig(), "SERIAL")
    camera.align_filter = Mock()
    camera.align_filter.process.return_value = None
    camera._on_frames(fake_frames())
    with pytest.raises(TimeoutError, match="no fresh RGB-D"):
        camera.async_read(timeout_ms=1)
    assert camera.consumed is None


def test_aligned_depth_dimension_mismatch_fails_explicitly():
    from unittest.mock import Mock

    config = OpenCVCameraConfig(index_or_path=0, width=3, height=2, fps=30)
    camera = OrbbecRGBDCamera(None, None, config, DepthRecordConfig(), "SERIAL")
    camera.align_filter = Mock()
    aligned = fake_frames()
    original_depth = aligned.get_depth_frame()
    original_depth.get_width = lambda: 2
    original_depth.get_height = lambda: 3
    aligned.get_depth_frame = lambda: original_depth
    camera.align_filter.process.return_value = aligned
    camera._on_frames(fake_frames())
    with pytest.raises(RuntimeError, match="capture failed") as error:
        camera.async_read()
    assert "does not match RGB" in str(error.value.__cause__)


def test_real_sdk_software_projection_at_640_by_480(tmp_path):
    sdk = pytest.importorskip("pyorbbecsdk")
    color = sdk.Frame.create_video_frame(sdk.OBFrameType.COLOR_FRAME, sdk.OBFormat.MJPG, 640, 480)
    success, encoded = cv2.imencode(".jpg", np.full((480, 640, 3), 100, dtype=np.uint8))
    assert success
    color_buffer = np.zeros(color.get_data_size(), dtype=np.uint8)
    color_buffer[: encoded.size] = encoded.ravel()
    color.update_data(color_buffer)
    depth = sdk.Frame.create_video_frame(
        sdk.OBFrameType.DEPTH_FRAME, sdk.OBFormat.Y16, 640, 480
    ).as_depth_frame()
    native_depth = np.zeros((480, 640), dtype=np.uint16)
    native_depth[200:240, 260:300] = 1000
    depth.update_data(native_depth.tobytes())
    depth.set_value_scale(1.0)
    profiles = []
    for frame in (color, depth):
        profile = frame.get_stream_profile().as_video_stream_profile()
        intrinsic = sdk.OBCameraIntrinsic()
        intrinsic.width, intrinsic.height = 640, 480
        intrinsic.fx = intrinsic.fy = 500
        intrinsic.cx, intrinsic.cy = 320, 240
        profile.set_intrinsic(intrinsic)
        distortion = sdk.OBCameraDistortion()
        distortion.model = sdk.OBCameraDistortionModel.NONE
        profile.set_distortion(distortion)
        profiles.append(profile)
    extrinsic = sdk.OBExtrinsic()
    extrinsic.rot = np.eye(3, dtype=np.float32)
    extrinsic.transform = np.array([20, 0, 0], dtype=np.float32)
    profiles[1].bind_extrinsic_to(profiles[0], extrinsic)
    frames = sdk.Frame.create_frame_set()
    frames.push_frame(color)
    frames.push_frame(depth)
    config = OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)
    camera = OrbbecRGBDCamera(sdk, None, config, DepthRecordConfig(), "SYNTHETIC")
    camera.align_filter = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
    camera.align_filter.set_match_target_resolution(True)
    camera._on_frames(frames)
    camera.async_read()
    projected = camera.consumed["depth"]
    assert projected.shape == (480, 640)
    assert projected.dtype == np.uint16
    assert native_depth[220, 265] == 1000
    assert projected[220, 265] == 0
    assert projected[220, 280] == 1000
    assert camera.calibration["depth_intrinsics"]["fx"] == 500
    dataset = create_dataset(tmp_path, {"head": camera}, use_videos=True)
    try:
        add_frame(dataset)
        dataset.save_episode()
        episode = dataset.root / "depth/episode_000000"
        saved = cv2.imread(str(episode / "head/frame_000000.png"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(saved, projected)
        metadata = json.loads((episode / "frames.jsonl").read_text())["cameras"]["head"]
        assert metadata["aligned_to_color"]
        assert metadata["alignment_mode"] == "software"
        assert metadata["depth_scale_mm"] == 1.0
    finally:
        dataset.finalize()

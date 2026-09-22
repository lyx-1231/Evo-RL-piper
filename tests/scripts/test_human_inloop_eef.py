import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import draccus
import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.scripts import human_inloop_depth, human_inloop_eef, lerobot_record
from lerobot.scripts.human_inloop_depth import DepthRecordConfig
from lerobot.scripts.human_inloop_eef import (
    EefRecordConfig,
    PiperActionEefConverter,
    PiperEefReader,
    record_with_eef,
)
from lerobot.scripts.lerobot_human_inloop_record import HumanInloopRecordConfig, human_inloop_record
from lerobot.scripts.lerobot_record import DatasetRecordConfig
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.piper_leader.config_piper_leader import PiperLeaderConfig
from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS


def feedback(timestamp=1.0, frame_index=0):
    return SimpleNamespace(
        time_stamp=timestamp,
        end_pose=SimpleNamespace(
            X_axis=100000 + frame_index * 1000,
            Y_axis=-20000,
            Z_axis=300000,
            RX_axis=90000,
            RY_axis=-45000,
            RZ_axis=180000,
        ),
    )


def test_cli_eef_is_optional():
    arguments = [
        "--robot.type=piper_follower",
        "--robot.port=FOLLOWER",
        "--teleop.type=piper_leader",
        "--teleop.port=LEADER",
        "--dataset.repo_id=local/test",
        "--dataset.single_task=test",
    ]
    assert not draccus.parse(HumanInloopRecordConfig, args=arguments).eef.enable
    config = draccus.parse(
        HumanInloopRecordConfig,
        args=[*arguments, "--eef.enable=true", "--eef.timeout_s=2", "--eef.max_age_s=0.2"],
    )
    assert config.eef == EefRecordConfig(enable=True, timeout_s=2, max_age_s=0.2)


@pytest.mark.parametrize("field", ["timeout_s", "max_age_s"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_eef_timeouts(field, value):
    with pytest.raises(ValueError, match="positive and finite"):
        EefRecordConfig(**{field: value})


def test_sdk_pose_conversion_preserves_signed_euler_angles_and_zero_values():
    message = feedback()
    message.end_pose.Y_axis = 0
    reader = PiperEefReader(SimpleNamespace(GetArmEndPoseMsgs=lambda: message), EefRecordConfig())
    assert list(reader.read().values()) == [100, 0, 300, 90, -45, 180]


def test_action_pose_uses_piper_fk_in_robot_base_frame():
    action = dict.fromkeys(PIPER_ACTION_KEYS, 0.0)
    pose = PiperActionEefConverter().convert(action)
    np.testing.assert_allclose(
        list(pose.values()),
        [56.1275121647, 0.0, 213.266268102, 0.0, 85.0, 0.0],
        atol=1e-9,
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), None])
def test_bad_pose_feedback_is_not_saved_as_zero(bad_value):
    message = feedback()
    message.end_pose.X_axis = bad_value
    reader = PiperEefReader(SimpleNamespace(GetArmEndPoseMsgs=lambda: message), EefRecordConfig())
    with pytest.raises(RuntimeError, match="feedback"):
        reader.read()


def fake_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)

    def sleep(duration):
        clock.now += duration

    monkeypatch.setattr(human_inloop_eef, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    return clock


def test_missing_and_stale_feedback_time_out(monkeypatch):
    clock = fake_clock(monkeypatch)
    message = feedback(timestamp=0)
    config = EefRecordConfig(timeout_s=0.02, max_age_s=0.1)
    reader = PiperEefReader(SimpleNamespace(GetArmEndPoseMsgs=lambda: message), config)
    with pytest.raises(TimeoutError, match="missing or stopped updating"):
        reader.read()
    message.time_stamp = 1.0
    reader.read()
    clock.now += 0.2
    with pytest.raises(TimeoutError, match="missing or stopped updating"):
        reader.read()
    message.time_stamp = 2.0
    assert reader.read()["eef.x_mm"] == 100


def test_backwards_feedback_timestamp_is_rejected():
    message = feedback(timestamp=2.0)
    reader = PiperEefReader(SimpleNamespace(GetArmEndPoseMsgs=lambda: message), EefRecordConfig())
    reader.read()
    message.time_stamp = 1.0
    with pytest.raises(RuntimeError, match="backwards"):
        reader.read()


class FakeCamera:
    def __init__(self, sdk, device, config, depth_config, serial):
        self.serial = serial
        self.calibration = {"aligned_to_color": True, "alignment_mode": "software"}
        self.is_connected = False
        self.consumed = None

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def async_read(self):
        self.consumed = {
            "color": np.full((32, 32, 3), 100, dtype=np.uint8),
            "depth": np.full((32, 32), 1500, dtype=np.uint16),
            "metadata": {"serial": self.serial, "depth_scale_mm": 1.0, "aligned_to_color": True},
        }
        return self.consumed["color"]


class FakePiper:
    name = robot_type = "piper_follower"
    action_features = dict.fromkeys(PIPER_ACTION_KEYS, float)

    def __init__(self, events, camera_names, eef_enabled):
        self.events = events
        self.cameras = {}
        self.observation_features = {
            **self.action_features,
            **dict.fromkeys(camera_names, (32, 32, 3)),
        }
        self.is_connected = False
        self.observations = 0
        self.arm = SimpleNamespace(GetArmEndPoseMsgs=Mock(side_effect=self.read_eef))
        self.eef_enabled = eef_enabled

    def read_eef(self):
        assert self.eef_enabled
        return feedback(timestamp=float(self.observations), frame_index=self.observations)

    def connect(self):
        self.is_connected = True
        for camera in self.cameras.values():
            camera.connect()

    def disconnect(self):
        self.is_connected = False
        for camera in self.cameras.values():
            camera.disconnect()

    def get_observation(self):
        self.observations += 1
        return {
            **{key: float(index + 1) for index, key in enumerate(PIPER_ACTION_KEYS)},
            **{name: camera.async_read() for name, camera in self.cameras.items()},
        }

    def send_action(self, action):
        assert list(action) == list(PIPER_ACTION_KEYS)
        if self.observations % 3 == 0:
            self.events["exit_early"] = True
            self.events["episode_outcome"] = "success"
        return action


def recording_setup(tmp_path, monkeypatch, eef_enabled=True, with_depth=False):
    events = {
        "exit_early": False,
        "episode_outcome": None,
        "rerecord_episode": False,
        "stop_recording": False,
        "toggle_intervention": False,
    }
    cameras = {
        name: OpenCVCameraConfig(index_or_path=index, width=32, height=32, fps=30)
        for index, name in enumerate(("head", "wrist") if with_depth else ())
    }
    config = HumanInloopRecordConfig(
        robot=PiperFollowerConfig(port="FOLLOWER", cameras=cameras),
        teleop=PiperLeaderConfig(port="LEADER"),
        dataset=DatasetRecordConfig(
            repo_id="local/test",
            single_task="test",
            root=tmp_path / "dataset",
            fps=30,
            num_episodes=2,
            episode_time_s=1,
            reset_time_s=0,
            video=with_depth,
            vcodec="h264",
            push_to_hub=False,
            num_image_writer_threads_per_camera=1,
        ),
        eef=EefRecordConfig(enable=eef_enabled),
        depth=DepthRecordConfig(enable=with_depth, serials={name: name.upper() for name in cameras}),
        play_sounds=False,
    )
    robot = FakePiper(events, cameras, eef_enabled)
    teleop = Mock(spec=Teleoperator)
    teleop.is_connected = True
    teleop.get_action.return_value = dict.fromkeys(PIPER_ACTION_KEYS, 10.0)
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda cfg: teleop)
    monkeypatch.setattr(lerobot_record, "init_keyboard_listener", lambda **kwargs: (None, events))
    if with_depth:
        devices = SimpleNamespace(get_device_by_serial_number=lambda serial: serial)
        sdk = SimpleNamespace(Context=lambda: SimpleNamespace(query_devices=lambda: devices))
        monkeypatch.setitem(sys.modules, "pyorbbecsdk", sdk)
        monkeypatch.setattr(human_inloop_depth, "OrbbecRGBDCamera", FakeCamera)
    return config, robot


@pytest.mark.parametrize("eef_enabled", [False, True])
@pytest.mark.parametrize("with_depth", [False, True])
def test_record_and_resume_eef_with_unchanged_joint_data(tmp_path, monkeypatch, eef_enabled, with_depth):
    config, robot = recording_setup(tmp_path, monkeypatch, eef_enabled, with_depth)
    original_observation = robot.get_observation
    original_factory = lerobot_record.make_robot_from_config
    original_features = lerobot_record.combine_feature_dicts
    dataset = human_inloop_record(config)
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 6
    config.resume = True
    config.dataset.num_episodes = 1
    resumed = human_inloop_record(config)
    assert resumed.num_episodes == 3
    assert resumed.num_frames == 9
    table = pq.read_table(config.dataset.root / "data").sort_by("index")
    np.testing.assert_array_equal(table["observation.state"].to_pylist(), [list(range(1, 8))] * 9)
    np.testing.assert_array_equal(table["action"].to_pylist(), [[10.0] * 7] * 9)
    info = json.loads((config.dataset.root / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [7]
    assert info["features"]["action"]["shape"] == [7]
    assert ("observation.eef_pose" in table.column_names) == eef_enabled
    assert ("action.eef_pose" in table.column_names) == eef_enabled
    if eef_enabled:
        np.testing.assert_allclose(
            table["observation.eef_pose"].to_pylist(),
            [[100 + index, -20, 300, 90, -45, 180] for index in range(1, 10)],
        )
        expected_action_eef = list(
            PiperActionEefConverter().convert(dict.fromkeys(PIPER_ACTION_KEYS, 10.0)).values()
        )
        np.testing.assert_allclose(table["action.eef_pose"].to_pylist(), [expected_action_eef] * 9)
        for key in ("observation.eef_pose", "action.eef_pose"):
            assert info["features"][key]["info"]["reference_frame"] == "robot_base"
            assert info["features"][key]["info"]["tcp_offset_applied"] is False
        assert info["features"]["observation.eef_pose"]["info"]["value_kind"] == "measured_state"
        assert info["features"]["action.eef_pose"]["info"]["value_kind"] == "commanded_target"
        loaded = LeRobotDataset("local/test", root=config.dataset.root, video_backend="pyav")
        np.testing.assert_allclose(loaded[8]["observation.eef_pose"].numpy(), [109, -20, 300, 90, -45, 180])
        np.testing.assert_allclose(loaded[8]["action.eef_pose"].numpy(), expected_action_eef)
    else:
        robot.arm.GetArmEndPoseMsgs.assert_not_called()
    if with_depth:
        for episode in range(3):
            path = config.dataset.root / "depth" / f"episode_{episode:06d}"
            manifest = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
            assert len(manifest) == 3
            for name in ("head", "wrist"):
                image = cv2.imread(str(path / name / "frame_000002.png"), cv2.IMREAD_UNCHANGED)
                assert image.dtype == np.uint16
                np.testing.assert_array_equal(image, np.full((32, 32), 1500, dtype=np.uint16))
    assert robot.get_observation == original_observation
    assert lerobot_record.make_robot_from_config is original_factory
    assert lerobot_record.combine_feature_dicts is original_features
    assert not robot.is_connected


def test_adding_eef_to_existing_joint_only_dataset_requires_new_dataset(tmp_path, monkeypatch):
    config, robot = recording_setup(tmp_path, monkeypatch, eef_enabled=False)
    human_inloop_record(config)
    config.resume = True
    config.eef.enable = True
    with pytest.raises(ValueError, match="metadata compatibility"):
        human_inloop_record(config)
    assert not robot.is_connected


def test_eef_failure_restores_observation_method_and_disconnects(tmp_path, monkeypatch):
    config, robot = recording_setup(tmp_path, monkeypatch)
    original_observation = robot.get_observation
    robot.arm.GetArmEndPoseMsgs.side_effect = RuntimeError("CAN disconnected")
    with pytest.raises(RuntimeError, match="CAN disconnected"):
        human_inloop_record(config)
    assert robot.get_observation == original_observation
    assert not robot.is_connected


def test_unsupported_robot_rejected_before_hardware_access():
    config = SimpleNamespace(robot=SimpleNamespace(type="so100_follower"))
    with pytest.raises(ValueError, match="requires a Piper"):
        record_with_eef(config, None)

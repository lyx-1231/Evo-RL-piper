import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.scripts import lerobot_record
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig
from lerobot.teleoperators.piper_leader.config_piper_leader import PiperLeaderConfig


class FakeHardware:
    name = "piper_follower"
    action_features = {"joint.pos": float}
    observation_features = {"joint.pos": float}

    def __init__(self):
        self.cameras = {}
        self.is_connected = False

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False


def config_for_test(tmp_path, num_episodes=2):
    return RecordConfig(
        robot=PiperFollowerConfig(port="FOLLOWER"),
        teleop=PiperLeaderConfig(port="LEADER"),
        dataset=DatasetRecordConfig(
            repo_id="local/test",
            single_task="test",
            root=tmp_path / "dataset",
            fps=30,
            num_episodes=num_episodes,
            episode_time_s=1,
            reset_time_s=1,
            video=False,
            push_to_hub=False,
        ),
        enable_episode_outcome_labeling=True,
        default_episode_success="failure",
        play_sounds=False,
    )


def setup_recording(monkeypatch):
    events = {
        "exit_early": False,
        "episode_outcome": None,
        "rerecord_episode": False,
        "stop_recording": False,
        "toggle_intervention": False,
    }
    hardware = [FakeHardware(), FakeHardware()]
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda config: hardware[0])
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda config: hardware[1])
    monkeypatch.setattr(lerobot_record, "init_keyboard_listener", lambda **kwargs: (None, events))
    return events, hardware


def add_test_frame(dataset):
    frame = {
        name: np.zeros(feature["shape"], dtype=np.float32)
        for name, feature in dataset.features.items()
        if name in {"action", "observation.state"} or name.startswith("complementary_info.")
    }
    dataset.add_frame({**frame, "task": "test"})


def test_success_pressed_during_save_does_not_skip_next_episode(tmp_path, monkeypatch):
    events, hardware = setup_recording(monkeypatch)
    original_save = LeRobotDataset.save_episode
    recording_calls = []

    def save_with_late_keypress(dataset, **kwargs):
        result = original_save(dataset, **kwargs)
        events["episode_outcome"] = "success"
        events["exit_early"] = True
        return result

    def record_loop(**kwargs):
        dataset = kwargs.get("dataset")
        if dataset is None:
            return
        recording_calls.append(dataset.num_episodes)
        assert not events["exit_early"]
        assert events["episode_outcome"] is None
        add_test_frame(dataset)
        events["episode_outcome"] = "success"

    monkeypatch.setattr(LeRobotDataset, "save_episode", save_with_late_keypress)
    monkeypatch.setattr(lerobot_record, "record_loop", record_loop)
    dataset = lerobot_record.record(config_for_test(tmp_path))
    assert recording_calls == [0, 1]
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 2
    assert all(not device.is_connected for device in hardware)


@pytest.mark.parametrize("stop_on_empty", [False, True])
def test_empty_episode_is_not_saved_or_counted(tmp_path, monkeypatch, stop_on_empty):
    events, hardware = setup_recording(monkeypatch)
    original_save = LeRobotDataset.save_episode
    recording_calls = []
    saved_sizes = []

    def save_nonempty(dataset, **kwargs):
        saved_sizes.append(dataset.episode_buffer["size"])
        assert dataset.episode_buffer["size"] > 0
        return original_save(dataset, **kwargs)

    def record_loop(**kwargs):
        dataset = kwargs.get("dataset")
        if dataset is None:
            return
        recording_calls.append(dataset.num_episodes)
        if len(recording_calls) == 1:
            events["stop_recording"] = stop_on_empty
            return
        add_test_frame(dataset)
        events["episode_outcome"] = "success"

    monkeypatch.setattr(LeRobotDataset, "save_episode", save_nonempty)
    monkeypatch.setattr(lerobot_record, "record_loop", record_loop)
    dataset = lerobot_record.record(config_for_test(tmp_path, num_episodes=1))
    assert dataset.num_episodes == (0 if stop_on_empty else 1)
    assert saved_sizes == ([] if stop_on_empty else [1])
    assert recording_calls == ([0] if stop_on_empty else [0, 0])
    assert all(not device.is_connected for device in hardware)


def test_escape_during_save_is_not_cleared(tmp_path, monkeypatch):
    events, _ = setup_recording(monkeypatch)
    original_save = LeRobotDataset.save_episode

    def save_and_stop(dataset, **kwargs):
        result = original_save(dataset, **kwargs)
        events["stop_recording"] = True
        events["exit_early"] = True
        return result

    def record_loop(**kwargs):
        dataset = kwargs.get("dataset")
        if dataset is not None:
            add_test_frame(dataset)
            events["episode_outcome"] = "success"

    monkeypatch.setattr(LeRobotDataset, "save_episode", save_and_stop)
    monkeypatch.setattr(lerobot_record, "record_loop", record_loop)
    dataset = lerobot_record.record(config_for_test(tmp_path))
    assert dataset.num_episodes == 1
    assert events["stop_recording"]


def test_zero_duration_rejected_before_hardware_connection(tmp_path, monkeypatch):
    _, hardware = setup_recording(monkeypatch)
    config = config_for_test(tmp_path)
    config.dataset.episode_time_s = 0
    with pytest.raises(ValueError, match="episode_time_s"):
        lerobot_record.record(config)
    assert all(not device.is_connected for device in hardware)

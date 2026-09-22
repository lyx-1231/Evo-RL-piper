"""Optional Piper end-effector feedback for human-in-loop datasets."""

import logging
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from types import FunctionType

from lerobot.scripts.human_inloop_depth import record_with_depth
from lerobot.utils.piper_sdk import PIPER_JOINT_ACTION_KEYS

EEF_COMPONENTS = (
    ("X_axis", "eef.x_mm"),
    ("Y_axis", "eef.y_mm"),
    ("Z_axis", "eef.z_mm"),
    ("RX_axis", "eef.rx_deg"),
    ("RY_axis", "eef.ry_deg"),
    ("RZ_axis", "eef.rz_deg"),
)


@dataclass
class EefRecordConfig:
    enable: bool = False
    timeout_s: float = 3.0
    max_age_s: float = 0.5

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in (self.timeout_s, self.max_age_s)):
            raise ValueError("EEF timeout_s and max_age_s must be positive and finite.")


def eef_feature(*, source: str, value_kind: str):
    return {
        "dtype": "float32",
        "shape": (6,),
        "names": [name for _, name in EEF_COMPONENTS],
        "info": {
            "source": source,
            "value_kind": value_kind,
            "reference_frame": "robot_base",
            "reference_point": "sdk_end_effector",
            "position_unit": "mm",
            "orientation_unit": "degree",
            "orientation_representation": "SDK RX/RY/RZ Euler angles",
            "tcp_offset_applied": False,
        },
    }


class PiperActionEefConverter:
    def __init__(self):
        from piper_sdk import C_PiperForwardKinematics

        self.kinematics = C_PiperForwardKinematics(dh_is_offset=1)

    def convert(self, action):
        missing = [key for key in PIPER_JOINT_ACTION_KEYS if key not in action]
        if missing:
            raise RuntimeError(f"Cannot compute action EEF pose; missing Piper joint targets: {missing}")
        try:
            joints_rad = [math.radians(float(action[key])) for key in PIPER_JOINT_ACTION_KEYS]
            pose = self.kinematics.CalFK(joints_rad)[-1]
            values = {name: float(value) for (_, name), value in zip(EEF_COMPONENTS, pose, strict=True)}
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError("Piper action EEF forward kinematics returned an invalid pose.") from exc
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError("Piper action EEF forward kinematics returned NaN or infinite values.")
        return values


class EefDatasetMixin:
    action_eef_converter: PiperActionEefConverter

    def prepare_action_for_dataset(self, action):
        return {**action, **self.action_eef_converter.convert(action)}


class PiperEefReader:
    def __init__(self, arm, config: EefRecordConfig):
        self.arm = arm
        self.config = config
        self.last_timestamp = None
        self.last_update = None

    def read(self):
        deadline = time.monotonic() + self.config.timeout_s
        while True:
            message = deepcopy(self.arm.GetArmEndPoseMsgs())
            timestamp = getattr(message, "time_stamp", None)
            pose = getattr(message, "end_pose", None)
            now = time.monotonic()
            if timestamp is not None and math.isfinite(timestamp) and timestamp > 0 and pose is not None:
                try:
                    values = {
                        name: float(getattr(pose, attribute)) * 1e-3 for attribute, name in EEF_COMPONENTS
                    }
                except (AttributeError, TypeError, ValueError) as exc:
                    raise RuntimeError("Piper EEF feedback is missing valid XYZ/RX/RY/RZ fields.") from exc
                if not all(math.isfinite(value) for value in values.values()):
                    raise RuntimeError("Piper EEF feedback contains NaN or infinite values.")
                if self.last_timestamp is not None and timestamp < self.last_timestamp:
                    raise RuntimeError("Piper EEF feedback timestamp moved backwards.")
                if self.last_timestamp is None or timestamp > self.last_timestamp:
                    self.last_timestamp = timestamp
                    self.last_update = now
                if now - self.last_update <= self.config.max_age_s:
                    return values
            if now >= deadline:
                raise TimeoutError(
                    "Piper EEF feedback is missing or stopped updating. "
                    "Check the follower CAN feedback; no placeholder EEF pose was saved."
                )
            time.sleep(min(0.005, max(0.0, deadline - now)))


def record_with_eef(cfg, record_function):
    if cfg.robot.type not in {"piper_follower", "piperx_follower"}:
        raise ValueError("`eef.enable=true` requires a Piper follower robot.")

    robot_factory = record_function.__globals__["make_robot_from_config"]
    combine_features = record_function.__globals__["combine_feature_dicts"]
    dataset_class = record_function.__globals__["LeRobotDataset"]
    action_eef_converter = PiperActionEefConverter()
    wrapped_robots = []

    class EefDataset(EefDatasetMixin, dataset_class):
        pass

    EefDataset.action_eef_converter = action_eef_converter

    def make_eef_robot(robot_config):
        robot = robot_factory(robot_config)
        if not callable(getattr(robot.arm, "GetArmEndPoseMsgs", None)):
            raise RuntimeError("The installed Piper SDK does not provide GetArmEndPoseMsgs().")
        reader = PiperEefReader(robot.arm, cfg.eef)
        original_observation = robot.get_observation

        def get_observation_with_eef():
            observation = original_observation()
            return {**observation, **reader.read()}

        wrapped_robots.append((robot, original_observation))
        robot.get_observation = get_observation_with_eef
        return robot

    def combine_features_with_eef(*features):
        return {
            **combine_features(*features),
            "observation.eef_pose": eef_feature(
                source="Piper.GetArmEndPoseMsgs",
                value_kind="measured_state",
            ),
            "action.eef_pose": eef_feature(
                source="Piper.C_PiperForwardKinematics(dh_is_offset=1)",
                value_kind="commanded_target",
            ),
        }

    local_globals = {
        **record_function.__globals__,
        "LeRobotDataset": EefDataset,
        "make_robot_from_config": make_eef_robot,
        "combine_feature_dicts": combine_features_with_eef,
    }
    eef_record = FunctionType(
        record_function.__code__,
        local_globals,
        record_function.__name__,
        record_function.__defaults__,
        record_function.__closure__,
    )
    eef_record.__kwdefaults__ = record_function.__kwdefaults__
    logging.info(
        "Saving Piper state and action EEF poses in the robot base frame: "
        "XYZ in mm, RX/RY/RZ in degrees; no TCP offset."
    )
    try:
        if cfg.depth.enable:
            return record_with_depth(cfg, eef_record)
        return eef_record(cfg)
    finally:
        for robot, original_observation in wrapped_robots:
            robot.get_observation = original_observation

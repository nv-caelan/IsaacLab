# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visuomotor cube packing where the blue cube sometimes blocks a shrunk platform."""

import random

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from .pack_ik_rel_visuomotor_env_cfg import (
    CUBE_HEIGHT,
    LEFT_CUBE_POSE_RANGE,
    PLATFORM_POSITION,
    RIGHT_CUBE_POSE_RANGE,
    FrankaCubePackTwoVisuomotorEnvCfg,
    cube_on_box,
    gripper_open,
)

BLOCKED_PLATFORM_SIZE = (0.05, 0.05, 0.01)
_BLOCKED_PLATFORM_HALF_X = BLOCKED_PLATFORM_SIZE[0] / 2
_BLOCKED_PLATFORM_HALF_Y = BLOCKED_PLATFORM_SIZE[1] / 2
BLUE_ON_PLATFORM_POSE_RANGE = {
    "x": (PLATFORM_POSITION[0] - _BLOCKED_PLATFORM_HALF_X, PLATFORM_POSITION[0] + _BLOCKED_PLATFORM_HALF_X),
    "y": (PLATFORM_POSITION[1] - _BLOCKED_PLATFORM_HALF_Y, PLATFORM_POSITION[1] + _BLOCKED_PLATFORM_HALF_Y),
    "z": (
        PLATFORM_POSITION[2] + BLOCKED_PLATFORM_SIZE[2] / 2 + CUBE_HEIGHT / 2,
        PLATFORM_POSITION[2] + BLOCKED_PLATFORM_SIZE[2] / 2 + CUBE_HEIGHT / 2,
    ),
    "yaw": (-1.0, 1.0),
}


def randomize_cube_position_mixture(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range_a: dict,
    pose_range_b: dict,
    probability_a: float = 0.5,
):
    """Reset a cube to ``pose_range_a`` with the given probability, else ``pose_range_b`` (per env)."""
    if env_ids is None:
        return
    for cur_env in env_ids.tolist():
        pose_range = pose_range_a if random.random() < probability_a else pose_range_b
        franka_stack_events.randomize_object_pose(
            env, torch.tensor([cur_env], device=env.device), asset_cfgs=[asset_cfg], pose_range=pose_range
        )


def red_block_on_platform(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Return whether the red cube is on the (shrunk) platform and the gripper is open."""
    robot = env.scene[robot_cfg.name]
    on_platform = cube_on_box(env, cube_2_cfg, PLATFORM_POSITION, BLOCKED_PLATFORM_SIZE)
    return on_platform & gripper_open(env, robot)


@configclass
class FrankaCubeBlockedVisuomotorEnvCfg(FrankaCubePackTwoVisuomotorEnvCfg):
    """Visuomotor packing where the blue cube sometimes blocks the (shrunk) platform.

    The blue cube resets directly on the platform with probability :attr:`blocked_probability`
    and in its usual left-side reset box otherwise; the red cube always resets to its usual
    right-side reset box (both from FrankaCubePackTwoVisuomotorEnvCfg). The goal is only for the
    red cube to reach the platform, so the robot must sometimes clear the blue cube off first.
    """

    blocked_probability: float = 0.5
    """Probability the blue cube resets directly on the platform instead of its usual reset box."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Shrink the platform
        self.scene.platform.spawn.size = BLOCKED_PLATFORM_SIZE

        # Blue starts on the platform (blocking it) with probability blocked_probability, and in
        # its usual left reset box otherwise; red always resets to its usual right reset box.
        self.scene.cube_1.init_state.pos = [PLATFORM_POSITION[0], PLATFORM_POSITION[1], CUBE_HEIGHT / 2]
        right_box_center_y = sum(RIGHT_CUBE_POSE_RANGE["y"]) / 2
        self.scene.cube_2.init_state.pos = [PLATFORM_POSITION[0], right_box_center_y, CUBE_HEIGHT / 2]
        self.events.randomize_cube_1_position = EventTerm(
            func=randomize_cube_position_mixture,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube_1"),
                "pose_range_a": BLUE_ON_PLATFORM_POSE_RANGE,
                "pose_range_b": LEFT_CUBE_POSE_RANGE,
                "probability_a": self.blocked_probability,
            },
        )
        self.events.randomize_cube_2_position = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={"pose_range": RIGHT_CUBE_POSE_RANGE, "asset_cfgs": [SceneEntityCfg("cube_2")]},
        )

        # Only the red cube needs to reach the platform
        self.terminations.success = DoneTerm(func=red_block_on_platform)


@configclass
class FrankaCubeAlwaysBlockedVisuomotorEnvCfg(FrankaCubeBlockedVisuomotorEnvCfg):
    """Visuomotor packing where the blue cube always starts blocking the (shrunk) platform."""

    blocked_probability: float = 1.0

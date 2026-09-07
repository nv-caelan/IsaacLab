# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visuomotor cube packing where either the red or blue cube reaching the platform succeeds."""

import random

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from .pack_ik_rel_visuomotor_env_cfg import (
    LEFT_CUBE_POSE_RANGE,
    PLATFORM_POSITION,
    PLATFORM_SIZE,
    RIGHT_CUBE_POSE_RANGE,
    FrankaCubePackTwoVisuomotorEnvCfg,
    cube_on_box,
    gripper_open,
)

# How far (in x) the "far" reset range sits from the cube's usual ("near") reset box.
FAR_OFFSET_X = 0.4


def _shift_xy(pose_range: dict, x: float = 0.0, y: float = 0.0) -> dict:
    """Return a copy of ``pose_range`` with its ``x``/``y`` bounds shifted by ``x``/``y``."""
    x_min, x_max = pose_range["x"]
    y_min, y_max = pose_range["y"]
    return {**pose_range, "x": (x_min + x, x_max + x), "y": (y_min + y, y_max + y)}


LEFT_CUBE_FAR_POSE_RANGE = _shift_xy(LEFT_CUBE_POSE_RANGE, x=FAR_OFFSET_X)
RIGHT_CUBE_FAR_POSE_RANGE = _shift_xy(RIGHT_CUBE_POSE_RANGE, x=FAR_OFFSET_X)


def randomize_cubes_near_far(
    env,
    env_ids: torch.Tensor,
    cube_1_cfg: SceneEntityCfg,
    cube_2_cfg: SceneEntityCfg,
    cube_1_near_range: dict,
    cube_1_far_range: dict,
    cube_2_near_range: dict,
    cube_2_far_range: dict,
    near_probability: float = 0.5,
):
    """Per env, put cube_1 in its near range and cube_2 in its far range with ``near_probability``;
    otherwise put cube_1 in its far range and cube_2 in its near range."""
    if env_ids is None:
        return
    for cur_env in env_ids.tolist():
        cur_env_ids = torch.tensor([cur_env], device=env.device)
        if random.random() < near_probability:
            cube_1_range, cube_2_range = cube_1_near_range, cube_2_far_range
        else:
            cube_1_range, cube_2_range = cube_1_far_range, cube_2_near_range
        franka_stack_events.randomize_object_pose(env, cur_env_ids, asset_cfgs=[cube_1_cfg], pose_range=cube_1_range)
        franka_stack_events.randomize_object_pose(env, cur_env_ids, asset_cfgs=[cube_2_cfg], pose_range=cube_2_range)


def either_cube_packed(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Return whether either cube is on the platform and the gripper is open."""
    robot = env.scene[robot_cfg.name]
    on_platform = cube_on_box(env, cube_1_cfg, PLATFORM_POSITION, PLATFORM_SIZE) | cube_on_box(
        env, cube_2_cfg, PLATFORM_POSITION, PLATFORM_SIZE
    )
    return on_platform & gripper_open(env, robot)


@configclass
class FrankaCubeEitherVisuomotorEnvCfg(FrankaCubePackTwoVisuomotorEnvCfg):
    """Visuomotor cube packing where success requires either cube (not both) on the platform.

    Each cube has a "near" reset range (its usual left/right box) and a "far" one (the same box
    shifted :data:`FAR_OFFSET_X` further away in x). With probability :attr:`near_probability`,
    cube_1 resets near and cube_2 resets far; otherwise cube_1 resets far and cube_2 resets near.
    """

    near_probability: float = 0.5
    """Probability cube_1 is the "near" cube (and cube_2 the "far" one) on a given reset."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # One cube resets near the platform and the other far from it, chosen per env
        self.events.randomize_cube_1_position = None
        self.events.randomize_cube_2_position = None
        self.events.randomize_cubes_near_far = EventTerm(
            func=randomize_cubes_near_far,
            mode="reset",
            params={
                "cube_1_cfg": SceneEntityCfg("cube_1"),
                "cube_2_cfg": SceneEntityCfg("cube_2"),
                "cube_1_near_range": LEFT_CUBE_POSE_RANGE,
                "cube_1_far_range": LEFT_CUBE_FAR_POSE_RANGE,
                "cube_2_near_range": RIGHT_CUBE_POSE_RANGE,
                "cube_2_far_range": RIGHT_CUBE_FAR_POSE_RANGE,
                "near_probability": self.near_probability,
            },
        )

        # Success requires just one cube on the platform, not both
        self.terminations.success = DoneTerm(func=either_cube_packed)

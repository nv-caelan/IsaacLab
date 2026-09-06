# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visuomotor cube packing: place the red and blue cubes on a fixed platform (no stacking)."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from .stack_ik_rel_visuomotor_env_cfg import FrankaCubeStackVisuomotorEnvCfg

# Cubes rest with their center this far above whatever surface they sit on (matches the
# blue/red/green block convention used throughout this config family).
CUBE_REST_HEIGHT_OFFSET = 0.0203

PLATFORM_SIZE = (0.2, 0.2, 0.01)
PLATFORM_POSITION = (0.5, 0.0, PLATFORM_SIZE[2] / 2)

# 0.1 m x 0.1 m reset boxes for the left (+y) and right (-y) cubes, right next to the platform
_RESET_BOX_SIZE = 0.1
_PLATFORM_HALF_Y = PLATFORM_SIZE[1] / 2
LEFT_CUBE_POSE_RANGE = {
    "x": (PLATFORM_POSITION[0] - _RESET_BOX_SIZE / 2, PLATFORM_POSITION[0] + _RESET_BOX_SIZE / 2),
    "y": (PLATFORM_POSITION[1] + _PLATFORM_HALF_Y, PLATFORM_POSITION[1] + _PLATFORM_HALF_Y + _RESET_BOX_SIZE),
    "z": (0.0203, 0.0203),
    "yaw": (-1.0, 1.0),
}
RIGHT_CUBE_POSE_RANGE = {
    "x": (PLATFORM_POSITION[0] - _RESET_BOX_SIZE / 2, PLATFORM_POSITION[0] + _RESET_BOX_SIZE / 2),
    "y": (PLATFORM_POSITION[1] - _PLATFORM_HALF_Y - _RESET_BOX_SIZE, PLATFORM_POSITION[1] - _PLATFORM_HALF_Y),
    "z": (0.0203, 0.0203),
    "yaw": (-1.0, 1.0),
}


def cube_on_platform(
    env, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"), height_tolerance: float = 0.01
) -> torch.Tensor:
    """Check whether a cube's center is within the platform footprint and resting on its surface."""
    cube_pos = env.scene[cube_cfg.name].data.root_pos_w.torch - env.scene.env_origins

    dx = torch.abs(cube_pos[:, 0] - PLATFORM_POSITION[0])
    dy = torch.abs(cube_pos[:, 1] - PLATFORM_POSITION[1])
    within_xy = (dx < PLATFORM_SIZE[0] / 2) & (dy < PLATFORM_SIZE[1] / 2)

    expected_z = PLATFORM_POSITION[2] + PLATFORM_SIZE[2] / 2 + CUBE_REST_HEIGHT_OFFSET
    height_ok = torch.abs(cube_pos[:, 2] - expected_z) < height_tolerance

    return within_xy & height_ok


def gripper_open(env, robot) -> torch.Tensor:
    """Check whether the robot's parallel gripper joints are at the open position."""
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"
    gripper_open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
    is_open = None
    for joint_id in gripper_joint_ids:
        joint_is_open = torch.isclose(robot.data.joint_pos.torch[:, joint_id], gripper_open_val, atol=1e-3)
        is_open = joint_is_open if is_open is None else is_open & joint_is_open
    return is_open


def cubes_packed(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Return whether both cubes are on the platform and the gripper is open."""
    robot = env.scene[robot_cfg.name]
    packed = cube_on_platform(env, cube_1_cfg) & cube_on_platform(env, cube_2_cfg)
    return packed & gripper_open(env, robot)


@configclass
class FrankaCubePackTwoVisuomotorEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    """Visuomotor cube packing with the red and blue cubes and a fixed platform (no green cube, no stacking)."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Drop the green cube from the scene entirely
        self.scene.cube_3 = None

        # Move the cubes' default spawn poses into their left/right reset boxes below; the reset
        # event re-randomizes them every episode, but the very first frame uses these poses.
        left_box_center_y = PLATFORM_POSITION[1] + _PLATFORM_HALF_Y + _RESET_BOX_SIZE / 2
        right_box_center_y = PLATFORM_POSITION[1] - _PLATFORM_HALF_Y - _RESET_BOX_SIZE / 2
        self.scene.cube_1.init_state.pos = [PLATFORM_POSITION[0], left_box_center_y, 0.0203]
        self.scene.cube_2.init_state.pos = [PLATFORM_POSITION[0], right_box_center_y, 0.0203]

        # Add a fixed rectangular platform in front of the robot. A kinematic RigidObjectCfg
        # (not AssetBaseCfg) so it's a named, non-graspable object in the scene's rigid_objects
        # (TAMP planners resolve goal targets by that name; a plain AssetBaseCfg prim would be an
        # unnamed collision obstacle instead).
        self.scene.platform = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Platform",
            init_state=RigidObjectCfg.InitialStateCfg(pos=PLATFORM_POSITION),
            spawn=sim_utils.CuboidCfg(
                size=PLATFORM_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                semantic_tags=[("class", "platform")],
            ),
        )

        # Reset cube_1 (left, +y) and cube_2 (right, -y) independently, each within its own
        # 0.1 m x 0.1 m box on its side of the platform, instead of the shared 3-cube range
        self.events.randomize_cube_positions = None
        self.events.randomize_cube_1_position = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={"pose_range": LEFT_CUBE_POSE_RANGE, "asset_cfgs": [SceneEntityCfg("cube_1")]},
        )
        self.events.randomize_cube_2_position = EventTerm(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={"pose_range": RIGHT_CUBE_POSE_RANGE, "asset_cfgs": [SceneEntityCfg("cube_2")]},
        )

        # Drop the low-dimensional object observations and subtask terms
        self.observations.policy.object = None
        self.observations.policy.cube_positions = None
        self.observations.policy.cube_orientations = None
        self.observations.subtask_terms = None

        # Drop the termination that references the removed cube, and check both cubes are packed
        self.terminations.cube_3_dropping = None
        self.terminations.success = DoneTerm(func=cubes_packed)

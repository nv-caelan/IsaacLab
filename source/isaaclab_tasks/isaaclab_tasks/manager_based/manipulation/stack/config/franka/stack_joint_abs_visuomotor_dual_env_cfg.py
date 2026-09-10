# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visuomotor cube stacking with two Franka arms side by side, modeled on
:class:`~isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_abs_visuomotor_env_cfg.FrankaCubeStackJointAbsVisuomotorEnvCfg`.

Standalone (not built on :class:`StackEnvCfg`): that base's ``ObjectTableSceneCfg`` and the shared
stack ``mdp`` helpers (``object_grasped``, ``object_stacked``, single ``ee_frame``/``object_obs``
observations) are built around exactly one ``robot`` field, so a second arm needs its own scene
entities, actions, and observations rather than fitting into the existing single-robot slots.
"""

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from .stack_ik_rel_visuomotor_env_cfg import _FRANKA_STACK_IK_REL_INIT_JOINT_POS

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


def _make_franka(prim_path: str, y_position: float = 0.0) -> ArticulationCfg:
    cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=prim_path)
    cfg.init_state.pos = (0.0, y_position, 0.0)
    cfg.init_state.joint_pos = _FRANKA_STACK_IK_REL_INIT_JOINT_POS
    return cfg


def _make_ee_frame(robot_prim: str) -> FrameTransformerCfg:
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    return FrameTransformerCfg(
        prim_path=f"{robot_prim}/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{robot_prim}/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{robot_prim}/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{robot_prim}/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )


def _make_wrist_cam(prim_path: str) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=200,
        width=200,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.13, 0.0, -0.15), rot=(0.03701, 0.03701, -0.70614, -0.70614), convention="ros"
        ),
    )


_ROBOT_Y_POSITION = 0.25

_CUBE_RIGID_PROPERTIES = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


@configclass
class DualFrankaCubeStackSceneCfg(InteractiveSceneCfg):
    """Two Franka arms side by side (offset in y), a shared table, three cubes, and cameras."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0, 0, 0.707, 0.707)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            # scale=(1.0, 1.0, 1.0),
            scale=(1.5, 1.0, 1.0),
            semantic_tags=[("class", "table")],
        ),
    )

    left_robot = _make_franka("{ENV_REGEX_NS}/LeftRobot", _ROBOT_Y_POSITION)
    right_robot = _make_franka("{ENV_REGEX_NS}/RightRobot", -_ROBOT_Y_POSITION)
    left_ee_frame = _make_ee_frame("{ENV_REGEX_NS}/LeftRobot")
    right_ee_frame = _make_ee_frame("{ENV_REGEX_NS}/RightRobot")

    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=_CUBE_RIGID_PROPERTIES,
            semantic_tags=[("class", "cube_1")],
        ),
    )
    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=_CUBE_RIGID_PROPERTIES,
            semantic_tags=[("class", "cube_2")],
        ),
    )
    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=_CUBE_RIGID_PROPERTIES,
            semantic_tags=[("class", "cube_3")],
        ),
    )

    table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=200,
        width=200,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.4), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"),
    )
    left_wrist_cam = _make_wrist_cam("{ENV_REGEX_NS}/LeftRobot/panda_hand/wrist_cam")
    right_wrist_cam = _make_wrist_cam("{ENV_REGEX_NS}/RightRobot/panda_hand/wrist_cam")


@configclass
class ActionsCfg:
    """Absolute (stiff-PD-tracked) joint-position commands for both arms and grippers."""

    left_arm_action = mdp.JointPositionActionCfg(
        asset_name="left_robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    left_gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="left_robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )
    right_arm_action = mdp.JointPositionActionCfg(
        asset_name="right_robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    right_gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="right_robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        left_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("left_robot")})
        left_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("left_robot")})
        right_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("right_robot")})
        right_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("right_robot")})
        cube_positions = ObsTerm(func=stack_mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=stack_mdp.cube_orientations_in_world_frame)
        left_eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos, params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame")})
        left_eef_quat = ObsTerm(func=stack_mdp.ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("left_ee_frame")})
        left_gripper_pos = ObsTerm(func=stack_mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg("left_robot")})
        right_eef_pos = ObsTerm(func=stack_mdp.ee_frame_pos, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")})
        right_eef_quat = ObsTerm(
            func=stack_mdp.ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")}
        )
        right_gripper_pos = ObsTerm(func=stack_mdp.gripper_pos, params={"robot_cfg": SceneEntityCfg("right_robot")})
        table_cam = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        left_wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    randomize_left_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("left_robot")},
    )
    randomize_right_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("right_robot")},
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


def cubes_stacked_dual_arm(
    env,
    left_robot_cfg: SceneEntityCfg = SceneEntityCfg("left_robot"),
    right_robot_cfg: SceneEntityCfg = SceneEntityCfg("right_robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
):
    """Cubes stacked (per :func:`stack_mdp.cubes_stacked`'s geometry check) and *both* grippers open.

    Either arm may have done the stacking; checking both grippers avoids reporting success while
    the arm that didn't stack is still mid-reach with its gripper closed on nothing.
    """
    left = stack_mdp.cubes_stacked(env, left_robot_cfg, cube_1_cfg, cube_2_cfg, cube_3_cfg)
    right = stack_mdp.cubes_stacked(env, right_robot_cfg, cube_1_cfg, cube_2_cfg, cube_3_cfg)
    return left & right


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_1_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")},
    )
    cube_2_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")},
    )
    cube_3_dropping = DoneTerm(
        func=stack_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")},
    )
    success = DoneTerm(func=cubes_stacked_dual_arm)


@configclass
class DualFrankaCubeStackJointAbsVisuomotorEnvCfg(ManagerBasedRLEnvCfg):
    """Visuomotor cube stacking with two Franka arms (side by side) instead of one."""

    scene: DualFrankaCubeStackSceneCfg = DualFrankaCubeStackSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Evaluation settings
    eval_mode = False
    eval_type = None

    def __post_init__(self):
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=2**21,
            friction_correlation_distance=0.00625,
        )
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"  # Use DLAA for higher quality rendering

        # utilities for gripper status check (shared: both arms use the same Franka gripper)
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "left_wrist_cam", "right_wrist_cam"]

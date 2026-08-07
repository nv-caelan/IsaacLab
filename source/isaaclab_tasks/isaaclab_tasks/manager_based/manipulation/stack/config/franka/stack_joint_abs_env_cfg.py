# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


def set_stiff_franka(cfg) -> None:
    """Swap in the gravity-disabled 400/80 Franka that the IK-relative envs use for tracking."""
    cfg.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.spawn.semantic_tags = [("class", "robot")]


@configclass
class FrankaCubeStackJointAbsEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    """State-based cube stacking with absolute joint-position control on the stiff Franka.

    The action is the parent's ``0.5 * a + default_joint_pos``; only the robot differs, so trajectories track
    without the gravity droop that the soft-gain :class:`FrankaCubeStackEnvCfg` shows.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        set_stiff_franka(self)

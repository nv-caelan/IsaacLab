# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from . import stack_ik_rel_visuomotor_env_cfg
from .stack_joint_rel_env_cfg import franka_rel_joint_action


@configclass
class FrankaCubeStackJointRelVisuomotorEnvCfg(stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg):
    """Visuomotor counterpart of :class:`FrankaCubeStackJointRelEnvCfg`.

    Inherits the cameras and the gravity-disabled stiff Franka from the IK-relative visuomotor env, since both
    command relative to the measured state; only the arm action differs.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.actions.arm_action = franka_rel_joint_action()

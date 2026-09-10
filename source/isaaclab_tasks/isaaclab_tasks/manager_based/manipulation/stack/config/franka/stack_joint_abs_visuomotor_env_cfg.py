# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from . import stack_joint_pos_visuomotor_env_cfg
from .stack_joint_abs_env_cfg import set_stiff_franka


@configclass
class FrankaCubeStackJointAbsVisuomotorEnvCfg(
    stack_joint_pos_visuomotor_env_cfg.FrankaCubeStackJointPosVisuomotorEnvCfg
):
    """Visuomotor counterpart of :class:`FrankaCubeStackJointAbsEnvCfg`.

    Inherits the absolute joint action and the table and wrist cameras; only the robot differs.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        set_stiff_franka(self)

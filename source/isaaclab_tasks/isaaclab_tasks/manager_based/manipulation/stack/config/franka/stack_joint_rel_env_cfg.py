# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from . import stack_joint_pos_env_cfg
from .stack_joint_abs_env_cfg import set_stiff_franka


def franka_rel_joint_action() -> mdp.RelativeJointPositionActionCfg:
    """Per-step joint deltas added to the measured joint positions.

    Re-basing on the measurement every step discards the untracked residual, so the delta acts as a velocity
    rather than a position command: the PD settles where ``stiffness * delta == damping * velocity``. At the
    stiff Franka's 400/80 that is ``5 * scale`` rad/s per unit action, so 0.2 gives 1 rad/s (measured 0.97 on
    both a shoulder and a forearm joint). The effort limits only bind during the transient, since the two PD
    terms cancel once the joint is up to speed.
    """
    return mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.2)


@configclass
class FrankaCubeStackJointRelEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    """State-based cube stacking with relative joint-position arm control."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Relative targets follow the measured joint positions, so a gravity-loaded arm sags with no restoring
        # error; the stiff gravity-disabled Franka is required here, not merely preferable.
        set_stiff_franka(self)

        self.actions.arm_action = franka_rel_joint_action()

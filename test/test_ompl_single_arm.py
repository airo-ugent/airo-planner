import numpy as np

from airo_planner import SingleArmOmplPlanner


def test_single_arm_scene():
    joint_bounds_lower = np.deg2rad([-360, -180, -160, -360, -360, -360])
    joint_bounds_upper = np.deg2rad([360, 0, 160, 360, 360, 360])
    joint_bounds = (joint_bounds_lower, joint_bounds_upper)

    def always_true_fn():
        return True

    SingleArmOmplPlanner(is_state_valid_fn=always_true_fn, joint_bounds=joint_bounds)

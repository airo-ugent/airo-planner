import numpy as np
import pytest

from airo_planner import NoPathFoundError, SingleArmOmplPlanner


def test_obstacle_free():
    joint_bounds_lower = np.deg2rad([-360, -180, -160, -360, -360, -360])
    joint_bounds_upper = np.deg2rad([360, 0, 160, 360, 360, 360])
    joint_bounds = (joint_bounds_lower, joint_bounds_upper)

    def always_true_fn(_):
        return True

    start_configuration = np.zeros(6)
    goal_configuration = np.deg2rad([270, 0, -45, 350, -350, 0.00001])

    planner = SingleArmOmplPlanner(is_state_valid_fn=always_true_fn, joint_bounds=joint_bounds)
    planner.plan_to_joint_configuration(start_configuration, goal_configuration)


def test_joint_space_tunnel():
    joint_bounds_lower = np.deg2rad([-360, -180, -160, -360, -360, -360])
    joint_bounds_upper = np.deg2rad([360, 0, 160, 360, 360, 360])
    joint_bounds = (joint_bounds_lower, joint_bounds_upper)

    def create_joint_space_tunnel_fn(width_in_degrees: float):
        def joint_space_tunnel_fn(joint_configuration: np.ndarray):
            joint_0 = joint_configuration[0]
            joint_1 = joint_configuration[1]

            if np.deg2rad(-90) <= joint_0 <= np.deg2rad(-10):
                if np.deg2rad(-50) <= joint_1 <= np.deg2rad(-50 + width_in_degrees):
                    return True
                return False

            return True

        return joint_space_tunnel_fn

    joint_space_wall_fn = create_joint_space_tunnel_fn(0.0)
    joint_space_tunnel_fn = create_joint_space_tunnel_fn(10.0)

    # Create an unreasonable narrow tunnel as well
    joint_space_narrow_tunnel_fn = create_joint_space_tunnel_fn(0.00000001)

    start_configuration = np.deg2rad([0, 0, 0, 0, 0, 0])
    goal_configuration = np.deg2rad([-100, -100, -45, 350, -350, 0.00001])

    with pytest.raises(NoPathFoundError):
        planner = SingleArmOmplPlanner(is_state_valid_fn=joint_space_wall_fn, joint_bounds=joint_bounds)
        planner.plan_to_joint_configuration(start_configuration, goal_configuration)

    planner2 = SingleArmOmplPlanner(is_state_valid_fn=joint_space_tunnel_fn, joint_bounds=joint_bounds)
    planner2.plan_to_joint_configuration(start_configuration, goal_configuration)

    with pytest.raises(NoPathFoundError):
        planner3 = SingleArmOmplPlanner(is_state_valid_fn=joint_space_narrow_tunnel_fn, joint_bounds=joint_bounds)
        planner3.plan_to_joint_configuration(start_configuration, goal_configuration)

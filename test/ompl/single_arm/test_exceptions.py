import numpy as np
import pytest
from airo_typing import HomogeneousMatrixType, JointConfigurationType

from airo_planner import (
    AllGoalConfigurationsRemovedError,
    InvalidGoalConfigurationError,
    NoInverseKinematicsSolutionsError,
    NoInverseKinematicsSolutionsWithinBoundsError,
    NoPathFoundError,
    NoValidInverseKinematicsSolutionsError,
    StartConfigurationOutOfBoundsError,
    create_simple_setup,
    uniform_symmetric_joint_bounds,
)
from airo_planner.exceptions import GoalConfigurationOutOfBoundsError, InvalidStartConfigurationError
from airo_planner.ompl.single_arm_ompl_planner import SingleArmOmplPlanner


def test_out_of_bounds():
    joint_bounds = uniform_symmetric_joint_bounds(6, np.pi)

    planner = SingleArmOmplPlanner(lambda _: True, joint_bounds)

    configuration_out_of_bounds = np.array([np.pi + 0.1, 0, 0, 0, 0, 0])
    configuration_in_of_bounds = np.array([0.0, 0, 0, 0, 0, 0])

    with pytest.raises(StartConfigurationOutOfBoundsError):
        planner.plan_to_joint_configuration(configuration_out_of_bounds, configuration_in_of_bounds)

    with pytest.raises(GoalConfigurationOutOfBoundsError):
        planner.plan_to_joint_configuration(configuration_in_of_bounds, configuration_out_of_bounds)

    with pytest.raises(StartConfigurationOutOfBoundsError):
        planner.plan_to_joint_configuration(configuration_out_of_bounds, configuration_out_of_bounds)

    path = planner.plan_to_joint_configuration(
        configuration_in_of_bounds, configuration_in_of_bounds
    )  # Should not raise
    assert path is not None  # Basic check for successful planning


def test_invalid_configurations():
    joint_bounds = uniform_symmetric_joint_bounds(6, np.pi)

    def only_positive_joints_allowed_fn(
        joint_configuration: JointConfigurationType,
    ) -> bool:
        return all(joint_configuration >= 0)

    planner = SingleArmOmplPlanner(only_positive_joints_allowed_fn, joint_bounds)

    positive_configuration = np.array([np.pi / 2] * 6)
    negative_configuration = np.array([-np.pi / 2] * 6)
    zero_configuration = np.zeros(6)

    with pytest.raises(InvalidStartConfigurationError):
        planner.plan_to_joint_configuration(negative_configuration, positive_configuration)

    with pytest.raises(InvalidGoalConfigurationError):
        planner.plan_to_joint_configuration(positive_configuration, negative_configuration)

    with pytest.raises(InvalidStartConfigurationError):
        planner.plan_to_joint_configuration(negative_configuration, negative_configuration)

    planner.plan_to_joint_configuration(positive_configuration, positive_configuration)
    planner.plan_to_joint_configuration(zero_configuration, zero_configuration)
    planner.plan_to_joint_configuration(zero_configuration, positive_configuration)
    planner.plan_to_joint_configuration(positive_configuration, zero_configuration)


def test_all_goals_removed():
    joint_bounds = uniform_symmetric_joint_bounds(6, np.pi)

    def dummy_inverse_kinematics_fn(tcp_pose: HomogeneousMatrixType) -> list[JointConfigurationType]:
        return [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]

    def filter_all_fn(goal_configurations: list[JointConfigurationType]) -> list[JointConfigurationType]:
        return []

    planner = SingleArmOmplPlanner(lambda _: True, joint_bounds, dummy_inverse_kinematics_fn, filter_all_fn)

    start_configuration = np.array([0.0, 0, 0, 0, 0, 0])
    goal_configuration = np.array([np.pi / 2] * 6)

    with pytest.raises(AllGoalConfigurationsRemovedError):
        planner.plan_to_tcp_pose(start_configuration, goal_configuration)

    planner.filter_goal_configurations_fn = None
    planner.rank_goal_configurations_fn = filter_all_fn

    with pytest.raises(RuntimeError):
        planner.plan_to_tcp_pose(start_configuration, goal_configuration)


def test_ik_problems():
    joint_bounds = uniform_symmetric_joint_bounds(6, np.pi)

    valid_ik_solutions = [np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]
    invalid_ik_solutions = [np.array([-0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]

    def valid_inverse_kinematics_fn(tcp_pose: HomogeneousMatrixType) -> list[JointConfigurationType]:
        return valid_ik_solutions

    def invalid_inverse_kinematics_fn(tcp_pose: HomogeneousMatrixType) -> list[JointConfigurationType]:
        return invalid_ik_solutions

    def out_of_bounds_inverse_kinematics_fn(tcp_pose: HomogeneousMatrixType) -> list[JointConfigurationType]:
        return [np.array([0, 0, 0, 0, 0, np.pi + 0.1])]

    def bad_inverse_kinematics_fn(tcp_pose: HomogeneousMatrixType) -> list[JointConfigurationType]:
        return []

    start_configuration = np.zeros(6)
    TCP_pose = np.identity(4)

    def is_state_valid_fn(joint_configuration: JointConfigurationType) -> bool:
        allowed_solutions = valid_ik_solutions + [start_configuration]
        for solution in allowed_solutions:
            if np.allclose(joint_configuration, solution):
                return True
        return False

    planner = SingleArmOmplPlanner(is_state_valid_fn, joint_bounds)

    with pytest.raises(AttributeError):
        planner.plan_to_tcp_pose(start_configuration, TCP_pose)

    planner.inverse_kinematics_fn = bad_inverse_kinematics_fn
    with pytest.raises(NoInverseKinematicsSolutionsError):
        planner.plan_to_tcp_pose(start_configuration, TCP_pose)

    planner.inverse_kinematics_fn = invalid_inverse_kinematics_fn
    with pytest.raises(NoValidInverseKinematicsSolutionsError):
        planner.plan_to_tcp_pose(start_configuration, TCP_pose)

    planner.inverse_kinematics_fn = out_of_bounds_inverse_kinematics_fn
    with pytest.raises(NoInverseKinematicsSolutionsWithinBoundsError):
        planner.plan_to_tcp_pose(start_configuration, TCP_pose)

    planner.inverse_kinematics_fn = valid_inverse_kinematics_fn
    with pytest.raises(NoPathFoundError):
        planner.plan_to_tcp_pose(start_configuration, TCP_pose)

    # TODO is the is_state_valid_fn was a property, the user wouldn't have to recreate the simple setup
    planner.is_state_valid_fn = lambda _: True
    planner._simple_setup = create_simple_setup(planner.is_state_valid_fn, planner.joint_bounds)
    planner.plan_to_tcp_pose(start_configuration, TCP_pose)

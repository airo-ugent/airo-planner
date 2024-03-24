import time
from typing import Callable

import numpy as np
from airo_typing import (
    HomogeneousMatrixType,
    InverseKinematicsFunctionType,
    JointConfigurationCheckerType,
    JointConfigurationType,
    JointPathType,
)
from loguru import logger

from airo_planner import (
    AllGoalConfigurationsRemovedError,
    GoalConfigurationOutOfBoundsError,
    InvalidGoalConfigurationError,
    InvalidStartConfigurationError,
    JointBoundsType,
    NoInverseKinematicsSolutionsError,
    NoInverseKinematicsSolutionsWithinBoundsError,
    NoPathFoundError,
    NoValidInverseKinematicsSolutionsError,
    SingleArmPlanner,
    StartConfigurationOutOfBoundsError,
    choose_shortest_path,
    create_simple_setup,
    is_within_bounds,
    solve_and_smooth_path,
    state_to_ompl,
    uniform_symmetric_joint_bounds,
)

# TODO move this to airo_typing?
JointConfigurationsModifierType = Callable[[list[JointConfigurationType]], list[JointConfigurationType]]
JointPathChooserType = Callable[[list[JointPathType]], JointPathType]


class SingleArmOmplPlanner(SingleArmPlanner):
    """Utility class for single-arm motion planning using OMPL.

    This class only plans in joint space.

    The purpose of this class is to make working with OMPL easier. It basically
    just handles the creation of OMPL objects and the conversion between numpy
    arrays and OMPL states and paths. After creating an instance of this class,
    you can also extract the SimpleSetup object and use it directly if you want.
    This can be useful for benchmarking with the OMPL benchmarking tools.
    """

    def __init__(
        self,
        is_state_valid_fn: JointConfigurationCheckerType,
        joint_bounds: JointBoundsType | None = None,
        inverse_kinematics_fn: InverseKinematicsFunctionType | None = None,
        filter_goal_configurations_fn: JointConfigurationsModifierType | None = None,
        rank_goal_configurations_fn: JointConfigurationsModifierType | None = None,
        choose_path_fn: JointPathChooserType = choose_shortest_path,
        degrees_of_freedom: int = 6,
    ):
        """Instiatiate a single-arm motion planner that uses OMPL. This creates
        a SimpleSetup object. Note that planning to TCP poses is only possible
        if the inverse kinematics function is provided.

        Note: you are free to change many of these atttributes after creation,
        but you might have to call `create_simple_setup` again to update the
        SimpleSetup object.

        Args:
            is_state_valid_fn: A function that checks if a given joint configuration is valid.
            inverse_kinematics_fn: A function that computes the inverse kinematics of a given TCP pose.
        """
        self.is_state_valid_fn = is_state_valid_fn

        # Functions for planning to TCP poses
        self.inverse_kinematics_fn = inverse_kinematics_fn
        self.filter_goal_configurations_fn = filter_goal_configurations_fn
        self.rank_goal_configurations_fn = rank_goal_configurations_fn
        self.choose_path_fn = choose_path_fn

        # Planning parameters
        self._degrees_of_freedom = degrees_of_freedom
        if joint_bounds is None:
            self.joint_bounds = uniform_symmetric_joint_bounds(self._degrees_of_freedom)
        else:
            self.joint_bounds = joint_bounds

        self._simple_setup = create_simple_setup(self.is_state_valid_fn, self.joint_bounds)

        # Debug attributes
        self._ik_solutions: list[JointConfigurationType] | None = None
        self._all_paths: list[JointPathType] | None = None

    def _set_start_and_goal_configurations(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> None:
        if not is_within_bounds(start_configuration, self.joint_bounds):
            raise StartConfigurationOutOfBoundsError(start_configuration, self.joint_bounds)

        if not is_within_bounds(goal_configuration, self.joint_bounds):
            raise GoalConfigurationOutOfBoundsError(goal_configuration, self.joint_bounds)

        if not self.is_state_valid_fn(start_configuration):
            raise InvalidStartConfigurationError(start_configuration)

        if not self.is_state_valid_fn(goal_configuration):
            raise InvalidGoalConfigurationError(goal_configuration)

        space = self._simple_setup.getStateSpace()
        start_state = state_to_ompl(start_configuration, space)
        goal_state = state_to_ompl(goal_configuration, space)
        self._simple_setup.setStartAndGoalStates(start_state, goal_state)

    def plan_to_joint_configuration(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> JointPathType:

        # Set start and goal
        self._simple_setup.clear()  # Needed to support multiple calls with different start/goal configurations
        self._set_start_and_goal_configurations(start_configuration, goal_configuration)
        simple_setup = self._simple_setup

        path = solve_and_smooth_path(simple_setup)

        if path is None:
            raise NoPathFoundError(start_configuration, goal_configuration)

        return path

    def _calculate_ik_solutions(self, tcp_pose: HomogeneousMatrixType) -> list[JointConfigurationType]:
        """Used by plan_to_tcp_pose() to calculate IK solutions."""

        if self.inverse_kinematics_fn is None:
            raise AttributeError("Inverse kinematics function is required for planning to TCP poses.")

        ik_solutions = self.inverse_kinematics_fn(tcp_pose)
        ik_solutions = [solution.squeeze() for solution in ik_solutions]
        self._ik_solutions = ik_solutions  # Saved for debugging

        if ik_solutions is None or len(ik_solutions) == 0:
            raise NoInverseKinematicsSolutionsError(tcp_pose)

        logger.info(f"IK returned {len(ik_solutions)} solutions.")
        return ik_solutions

    def _check_ik_solutions_bounds(self, ik_solutions: list[JointConfigurationType]) -> list[JointConfigurationType]:
        """Used by plan_to_tcp_pose() to check which IK solutions are within the joint bounds."""
        ik_solutions_within_bounds = []
        joints_lower, joints_upper = self.joint_bounds
        for ik_solution in ik_solutions:
            if np.all(ik_solution >= joints_lower) and np.all(ik_solution <= joints_upper):
                ik_solutions_within_bounds.append(ik_solution)

        if len(ik_solutions_within_bounds) == 0:
            raise NoInverseKinematicsSolutionsWithinBoundsError(ik_solutions, self.joint_bounds)

        logger.info(f"Found {len(ik_solutions_within_bounds)}/{len(ik_solutions)} solutions within joint bounds.")
        return ik_solutions_within_bounds

    def _check_ik_solutions_validity(self, ik_solutions: list[JointConfigurationType]) -> list[JointConfigurationType]:
        """Used by plan_to_tcp_pose() to check which IK solutions are valid."""
        ik_solutions_valid = [s for s in ik_solutions if self.is_state_valid_fn(s)]
        if len(ik_solutions_valid) == 0:
            raise NoValidInverseKinematicsSolutionsError(ik_solutions)

        logger.info(f"Found {len(ik_solutions_valid)}/{len(ik_solutions)} valid solutions.")
        return ik_solutions_valid

    def _filter_ik_solutions(self, ik_solutions: list[JointConfigurationType]) -> list[JointConfigurationType]:
        """Used by plan_to_tcp_pose() to filter IK solutions."""
        if self.filter_goal_configurations_fn is None:
            return ik_solutions

        ik_solutions_filtered = self.filter_goal_configurations_fn(ik_solutions)
        if len(ik_solutions_filtered) == 0:
            raise AllGoalConfigurationsRemovedError(ik_solutions)

        logger.info(f"Filtered IK solutions to {len(ik_solutions_filtered)}/{len(ik_solutions)} solutions.")
        return ik_solutions_filtered

    def _rank_ik_solution(self, ik_solutions: list[JointConfigurationType]) -> list[JointConfigurationType]:
        """Used by plan_to_tcp_pose() to rank IK solutions."""
        if self.rank_goal_configurations_fn is None:
            return ik_solutions

        logger.info("Ranking IK solutions.")
        ik_solutions_ranked = self.rank_goal_configurations_fn(ik_solutions)

        if len(ik_solutions_ranked) != len(ik_solutions):
            logger.warning(
                f"Ranking function changed the number of IK solutions from {len(ik_solutions)} to {len(ik_solutions_ranked)}."
            )

        if len(ik_solutions_ranked) == 0:
            raise RuntimeError("Ranking function removed all IK solutions. This is probably an implementation error.")

        return ik_solutions_ranked

    def plan_to_tcp_pose(
        self, start_configuration: JointConfigurationType, tcp_pose: HomogeneousMatrixType
    ) -> JointPathType:

        ik_solutions = self._calculate_ik_solutions(tcp_pose)
        ik_solutions_within_bounds = self._check_ik_solutions_bounds(ik_solutions)
        ik_solutions_valid = self._check_ik_solutions_validity(ik_solutions_within_bounds)
        ik_solutions_filtered = self._filter_ik_solutions(ik_solutions_valid)
        ik_solutions_ranked = self._rank_ik_solution(ik_solutions_filtered)

        logger.info(f"Running OMPL towards {len(ik_solutions_ranked)} IK solutions.")
        goal_configurations = ik_solutions_ranked
        return_first_success = True if self.rank_goal_configurations_fn is not None else False

        # Try solving to each IK solution in joint space.
        start_time = time.time()
        paths = []
        for i, goal_configuration in enumerate(goal_configurations):
            try:
                path = self.plan_to_joint_configuration(start_configuration, goal_configuration)
            except NoPathFoundError:
                logger.info(f"No path found to goal configuration: {i}.")
                continue

            if return_first_success:
                logger.info(f"Returning first successful path (planning time: {time.time() - start_time:.2f} s).")
                return path
            paths.append(path)

        end_time = time.time()

        if len(paths) == 0:
            raise NoPathFoundError(start_configuration, goal_configurations)

        self._all_paths = paths  # Saved for debugging
        logger.info(f"Found {len(paths)} paths towards IK solutions, (planning time: {end_time - start_time:.2f} s).")

        solution_path = self.choose_path_fn(paths)

        if solution_path is None:
            raise RuntimeError(f"Path choosing function did not return a path out of {len(paths)} options.")

        return solution_path

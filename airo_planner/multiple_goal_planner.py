import abc
import time
from typing import Callable

import numpy as np
from airo_typing import JointConfigurationCheckerType, JointConfigurationType, JointPathType
from loguru import logger

from airo_planner import (
    AllGoalConfigurationsRemovedError,
    JointBoundsType,
    NoInverseKinematicsSolutionsWithinBoundsError,
    NoPathFoundError,
    NoValidInverseKinematicsSolutionsError,
    choose_shortest_path,
)

# TODO move this to airo_typing?
JointConfigurationsModifierType = Callable[[list[JointConfigurationType]], list[JointConfigurationType]]
JointPathChooserType = Callable[[list[JointPathType]], JointPathType]


class MultipleGoalPlanner(abc.ABC):
    """Base class for planning to a finite set of candidate goal configurations."""

    def __init__(
        self,
        is_state_valid_fn: JointConfigurationCheckerType,
        joint_bounds: JointBoundsType,
        filter_goal_configurations_fn: JointConfigurationsModifierType | None = None,
        rank_goal_configurations_fn: JointConfigurationsModifierType | None = None,
        choose_path_fn: JointPathChooserType = choose_shortest_path,
    ):
        self.is_state_valid_fn = is_state_valid_fn
        self.joint_bounds = joint_bounds

        # Functions for planning to multiple goal configurations
        self.filter_goal_configurations_fn = filter_goal_configurations_fn
        self.rank_goal_configurations_fn = rank_goal_configurations_fn
        self.choose_path_fn = choose_path_fn

        # Used for debugging
        self._all_paths: list[JointPathType] | None = None
        self._goal_configurations: list[JointConfigurationType] | None = None

    @abc.abstractmethod
    def _plan_to_joint_configuration_stacked(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> JointPathType:
        raise NotImplementedError

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

    def _rank_ik_solutions(self, ik_solutions: list[JointConfigurationType]) -> list[JointConfigurationType]:
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

    def _plan_to_goal_configurations(
        self,
        start_configuration: JointConfigurationType,
        goal_configurations: list[JointConfigurationType],
        return_first_success: bool = False,
    ) -> JointPathType:
        self._goal_configurations = goal_configurations  # Saved for debugging

        # Try solving to each goal configuration
        start_time = time.time()
        paths = []
        for i, goal_configuration in enumerate(goal_configurations):
            try:
                path = self._plan_to_joint_configuration_stacked(start_configuration, goal_configuration)
            except NoPathFoundError:
                logger.info(f"No path found to goal configuration: {i}.")
                continue

            if return_first_success:
                logger.success(f"Returning first successful path (planning time: {time.time() - start_time:.2f} s).")
                return path
            paths.append(path)

        end_time = time.time()

        if len(paths) == 0:
            raise NoPathFoundError(start_configuration, goal_configurations)

        self._all_paths = paths  # Saved for debugging
        logger.success(
            f"Found {len(paths)} paths towards IK solutions, (planning time: {end_time - start_time:.2f} s)."
        )

        solution_path = self.choose_path_fn(paths)

        if solution_path is None:
            raise RuntimeError(f"Path choosing function did not return a path out of {len(paths)} options.")

        logger.success(f"Chose path with {len(solution_path)} waypoints.")

        return solution_path

    def plan_to_ik_solutions(
        self, start_configuration: JointConfigurationType, ik_solutions: list[JointConfigurationType]
    ) -> JointPathType:
        """Plan to a list of IK solutions."""
        ik_solutions_within_bounds = self._check_ik_solutions_bounds(ik_solutions)
        ik_solutions_valid = self._check_ik_solutions_validity(ik_solutions_within_bounds)
        ik_solutions_filtered = self._filter_ik_solutions(ik_solutions_valid)
        ik_solutions_ranked = self._rank_ik_solutions(ik_solutions_filtered)

        logger.info(f"Running OMPL towards {len(ik_solutions_ranked)} IK solutions.")

        return_first_success = True if self.rank_goal_configurations_fn is not None else False

        solution_path = self._plan_to_goal_configurations(
            start_configuration, ik_solutions_ranked, return_first_success
        )
        return solution_path

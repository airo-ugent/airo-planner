import sys
import time
from typing import Callable, List

import numpy as np
from airo_typing import (
    HomogeneousMatrixType,
    InverseKinematicsFunctionType,
    JointConfigurationCheckerType,
    JointConfigurationType,
    JointPathType,
)
from loguru import logger
from ompl import geometric as og

from airo_planner import JointBoundsType, SingleArmPlanner, path_from_ompl, state_to_ompl
from airo_planner.ompl.utilities import create_simple_setup
from airo_planner.selection.path_selection import choose_shortest_path

# TODO move this to airo_typing
JointConfigurationsModifierType = Callable[[List[JointConfigurationType]], List[JointConfigurationType]]
JointPathChooserType = Callable[[List[JointPathType]], JointPathType]


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
        choose_path_fn: JointPathChooserType | None = choose_shortest_path,
        degrees_of_freedom: int = 6,
    ):
        """Instiatiate a single-arm motion planner that uses OMPL. This creates
        a SimpleSetup object. Note that planning to TCP poses is only possible
        if the inverse kinematics function is provided.

        Note: you are free to change many of these atttributes after creation,
        but you might have to call `_create_simple_setup` again to update the
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
            self.joint_bounds = np.full(degrees_of_freedom, -2 * np.pi), np.full(degrees_of_freedom, 2 * np.pi)
        else:
            self.joint_bounds = joint_bounds

        self._simple_setup = create_simple_setup(self.is_state_valid_fn, self.joint_bounds)

        # Debug attributes
        self._ik_solutions: List[JointConfigurationType] | None = None
        self._all_paths: List[JointPathType] | None = None

    def _set_start_and_goal_configurations(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> None:
        space = self._simple_setup.getStateSpace()
        start_state = state_to_ompl(start_configuration, space)
        goal_state = state_to_ompl(goal_configuration, space)
        self._simple_setup.setStartAndGoalStates(start_state, goal_state)

    def plan_to_joint_configuration(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> JointPathType | None:

        # Set start and goal
        self._simple_setup.clear()  # Needed to support multiple calls with different start/goal configurations
        self._set_start_and_goal_configurations(start_configuration, goal_configuration)
        simple_setup = self._simple_setup

        # Run planning algorithm
        simple_setup.solve()

        if not simple_setup.haveExactSolutionPath():
            return None

        # Simplify and smooth the path, note that this conserves path validity
        simple_setup.simplifySolution()
        path_simplifier = og.PathSimplifier(simple_setup.getSpaceInformation())
        path = simple_setup.getSolutionPath()
        path_simplifier.smoothBSpline(path)

        # Extract path
        path_numpy = path_from_ompl(path, self._degrees_of_freedom)
        return path_numpy

    def plan_to_tcp_pose(  # noqa: C901
        self,
        start_configuration: JointConfigurationType,
        tcp_pose_in_base: HomogeneousMatrixType,
    ) -> JointPathType | None:
        # TODO: add options for specifying a preferred IK solutions, e.g. min distance to a joint configuration
        # desirable_goal_joint_configurations = Optional[List[JointConfigurationType]]
        # Without this we plan to all joint configs and pick the shortest path
        # With it, we try the closest IK solution first and if it fails we try the next closest etc.
        pass

        if self.inverse_kinematics_fn is None:
            logger.warning("Planning to TCP pose attempted but inverse_kinematics_fn was provided, returing None.")
            return None

        ik_solutions = self.inverse_kinematics_fn(tcp_pose_in_base)
        ik_solutions = [solution.squeeze() for solution in ik_solutions]
        self._ik_solutions = ik_solutions  # Save for debugging

        if ik_solutions is None or len(ik_solutions) == 0:
            logger.info("IK returned no solutions, returning None.")
            return None
        else:
            logger.info(f"IK returned {len(ik_solutions)} solutions.")

        ik_solutions_within_bounds = []
        for ik_solution in ik_solutions:
            if np.all(ik_solution >= self.joint_bounds[0]) and np.all(ik_solution <= self.joint_bounds[1]):
                ik_solutions_within_bounds.append(ik_solution)

        if len(ik_solutions_within_bounds) == 0:
            logger.info("No IK solutions are within the joint bounds, returning None.")
            return None
        else:
            logger.info(f"Found {len(ik_solutions_within_bounds)}/{len(ik_solutions)} solutions within joint bounds.")

        ik_solutions_valid = [s for s in ik_solutions_within_bounds if self.is_state_valid_fn(s)]
        if len(ik_solutions_valid) == 0:
            logger.info("All IK solutions within bounds are invalid, returning None.")
            return None
        else:
            logger.info(f"Found {len(ik_solutions_valid)}/{len(ik_solutions_within_bounds)} valid solutions.")

        # TODO filter the IK solutions -> warn when nothing is left

        # TODO rank the IK solutions
        terminate_on_first_success = self.rank_goal_configurations_fn is not None

        logger.info(f"Running OMPL towards {len(ik_solutions_valid)} IK solutions.")
        start_time = time.time()
        # Try solving to each IK solution in joint space.
        paths = []
        for i, ik_solution in enumerate(ik_solutions_valid):
            path = self.plan_to_joint_configuration(start_configuration, ik_solution)

            # My attempt to get the OMPL Info: messages to show up in the correct order between the loguru logs
            # This however does not seems to work, so I don't know where those prints are buffered
            sys.stdout.flush()

            if path is not None:
                logger.info(f"Successfully found path to the valid IK solution with index {i}.")
                if terminate_on_first_success is True:
                    end_time = time.time()
                    logger.info(f"Terminating on first success (planning time: {end_time - start_time:.2f} s).")
                    return path
                paths.append(path)

        end_time = time.time()
        logger.info(f"Found {len(paths)} paths towards IK solutions, (planning time: {end_time - start_time:.2f} s).")

        self._all_paths = paths  # Save for debugging

        if len(paths) == 0:
            logger.info("No paths founds towards any IK solutions, returning None.")
            return None

        solution_path = self.choose_path_fn(paths)
        return solution_path

        # TODO: exhaustive mode vs iterative mode (~= greedy mode)
        # depending on the helper functions provide to the planner, it will operate in different modes
        # by default, it operates in an exhaustive mode, meaning that it will treat all IK solutions the same, and plan a path to them all
        # then it will return the shortest path

        # path_distances = [np.linalg.norm(path[-1] - start_configuration) for path in paths]

        # path_desirablities = None
        # if desirable_goal_configurations is not None:
        #     path_desirablities = []
        #     for path in paths:
        #         min_distance = np.inf
        #         for desirable_goal in desirable_goal_configurations:
        #             distance = np.linalg.norm(path[-1] - desirable_goal)
        #             min_distance = min(min_distance, distance)  # type: ignore # I don't understand the mypy error here
        #         path_desirablities.append(min_distance)

        # path_lengths = [] # TODO calculate

        # lengths_str = f"{[np.round(l, 3) for l in path_lengths]}"
        # distances_str = f"{[np.round(d, 3) for d in path_distances]}"
        # logger.info(f"Found {len(paths)} paths towards IK solutions:")
        # logger.info(f"Path lengths: {lengths_str}")
        # logger.info(f"Path distances: {distances_str}")

        # if path_desirablities is not None:
        #     desirabilities_str = f"{[np.round(d, 3) for d in path_desirablities]}"
        #     logger.info(f"Path desirabilities: {desirabilities_str}")

        # use_desirability = path_desirablities is not None

        # if use_desirability:
        #     idx = np.argmin(path_desirablities)
        #     logger.info(
        #         f"Length of chosen solution (= most desirable end): {path_lengths[idx]:.3f}, desirability: {path_desirablities[idx]:.3f}"
        #     )
        # else:
        #     idx = np.argmin(path_lengths)
        #     logger.info(f"Length of chosen solution (= shortest path): {path_lengths[idx]:.3f}")
        #
        # solution_path = paths[idx]
        # return solution_path

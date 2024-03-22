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

from airo_planner import (
    JointBoundsType,
    SingleArmPlanner,
    choose_shortest_path,
    create_simple_setup,
    is_within_bounds,
    solve_and_smooth_path,
    state_to_ompl,
    uniform_symmetric_joint_bounds,
)

# TODO move this to airo_typing?
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
        choose_path_fn: JointPathChooserType = choose_shortest_path,
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
            self.joint_bounds = uniform_symmetric_joint_bounds(self._degrees_of_freedom)
        else:
            self.joint_bounds = joint_bounds

        self._simple_setup = create_simple_setup(self.is_state_valid_fn, self.joint_bounds)

        # Debug attributes
        self._ik_solutions: List[JointConfigurationType] | None = None
        self._all_paths: List[JointPathType] | None = None

    def _set_start_and_goal_configurations(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> None:
        if not self.is_state_valid_fn(start_configuration):
            raise ValueError("Planner was given an invalid start configuration.")

        if not self.is_state_valid_fn(goal_configuration):
            raise ValueError("Planner was given an invalid goal configuration.")

        # TODO check joint bounds as well
        if not is_within_bounds(start_configuration, self.joint_bounds):
            raise ValueError("Start configuration is not within the joint bounds.")

        if not is_within_bounds(goal_configuration, self.joint_bounds):
            raise ValueError("Goal configuration is not within the joint bounds.")

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

        path = solve_and_smooth_path(simple_setup)
        return path

    def plan_to_tcp_pose(  # noqa: C901
        self,
        start_configuration: JointConfigurationType,
        tcp_pose: HomogeneousMatrixType,
    ) -> JointPathType | None:

        if self.inverse_kinematics_fn is None:
            raise AttributeError("Inverse kinematics function is required for planning to TCP poses.")

        ik_solutions = self.inverse_kinematics_fn(tcp_pose)
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

        if self.filter_goal_configurations_fn is not None:
            ik_solutions_filtered = self.filter_goal_configurations_fn(ik_solutions_valid)
            if len(ik_solutions_filtered) == 0:
                logger.info("All IK solutions were filtered out, returning None.")
                return None
            else:
                logger.info(
                    f"Filtered IK solutions to {len(ik_solutions_filtered)}/{len(ik_solutions_valid)} solutions."
                )
        else:
            ik_solutions_filtered = ik_solutions_valid

        if self.rank_goal_configurations_fn is not None:
            logger.info("Ranking IK solutions.")
            terminate_on_first_success = True
            ik_solutions_filtered = self.rank_goal_configurations_fn(ik_solutions_filtered)
        else:
            logger.info("No ranking function provided, will try planning to all valid IK solutions.")
            terminate_on_first_success = False

        logger.info(f"Running OMPL towards {len(ik_solutions_filtered)} IK solutions.")
        start_time = time.time()
        # Try solving to each IK solution in joint space.
        paths = []
        for i, ik_solution in enumerate(ik_solutions_filtered):
            path = self.plan_to_joint_configuration(start_configuration, ik_solution)

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

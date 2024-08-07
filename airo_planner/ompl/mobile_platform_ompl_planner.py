import numpy as np
from airo_typing import Vector3DType, JointPathType, JointConfigurationCheckerType
from loguru import logger

from airo_planner import create_simple_setup, state_to_ompl, NoPathFoundError, solve_and_smooth_path
from airo_planner.interfaces import MobilePlatformPlanner


class MobilePlatformOmplPlanner(MobilePlatformPlanner):
    """Utility class for mobile platform motion planning using OMPL.

    The purpose of this class is to make working with OMPL easier. It basically
    just handles the creation of OMPL objects and the conversion between numpy
    arrays and OMPL states and paths. After creating an instance of this class,
    you can also extract the SimpleSetup object and use it directly if you want.
    This can be useful for benchmarking with the OMPL benchmarking tools.
    """

    def __init__(self, is_state_valid_fn: JointConfigurationCheckerType):
        # TODO: JointConfigurationCheckerType is not ideal here, should be a new type.
        self.is_state_valid_fn = is_state_valid_fn

        joint_bounds = (
            np.full(3, -np.inf),
            np.full(3, np.inf),
        )
        self._simple_setup = create_simple_setup(self.is_state_valid_fn, joint_bounds)

    def plan_to_pose(self, start_pose: Vector3DType, goal_pose: Vector3DType) -> JointPathType:
        self._simple_setup.clear()

        space = self._simple_setup.getStateSpace()
        start_state = state_to_ompl(start_pose, space)
        goal_state = state_to_ompl(goal_pose, space)
        self._simple_setup.setStartAndGoalStates(start_state, goal_state)

        path = solve_and_smooth_path(self._simple_setup)

        if path is None:
            raise NoPathFoundError(start_pose, goal_pose)

        logger.success(f"Successfully found path (with {len(path)} waypoints).")

        return path

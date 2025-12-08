from os import PathLike

import numpy as np
import torch
from airo_typing import HomogeneousMatrixType, JointConfigurationType, JointPathContainer, SingleArmTrajectory
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from loguru import logger

from airo_planner import NoPathFoundError, SingleArmPlanner


class SingleArmCuroboPlanner(SingleArmPlanner):
    """Utility class for single-arm motion planning using Curobo.

    This class only plans in joint space.

    The purpose of this class is to make working with Curobo easier.
    """

    def __init__(
        self,
        curobo_robot_file: PathLike,
        curobo_world_file: PathLike,
        allowed_planning_time: float = 1.0,
    ):
        """Instiatiate a single-arm motion planner that uses Curobo.

        Args:
            curobo_robot_file: Path to the robot YAML file.
            curobo_world_file: Path to the environment (world) YAML file.
            allowed_planning_time: Maximum planning time in seconds.
        """
        tensor_args = TensorDeviceType()

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            curobo_robot_file,
            curobo_world_file,
            tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=False,
        )

        self.motion_gen_plan_config = MotionGenPlanConfig(
            timeout=allowed_planning_time,
            enable_graph=False,
            enable_opt=True,
            max_attempts=1,
            num_trajopt_seeds=10,
            num_graph_seeds=10,
        )

        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.optimize_dt = False
        self.motion_gen.warmup()

    def plan_to_joint_configuration(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> SingleArmTrajectory:
        start_state = JointState.from_position(
            torch.tensor(start_configuration, dtype=torch.float32)[None, :].cuda().contiguous()
        )
        goal_state = JointState.from_position(
            torch.tensor(goal_configuration, dtype=torch.float32)[None, :].cuda().contiguous()
        )

        result = self.motion_gen.plan_single_js(start_state, goal_state, self.motion_gen_plan_config)

        if not result.success:
            logger.warning(f"Failed to plan with status: {result.status}")
            raise NoPathFoundError(start_configuration, goal_configuration)

        path = result.interpolated_plan.position.cpu().numpy()

        logger.success(f"Successfully found path (with {len(path)} waypoints).")

        dt = result.interpolation_dt
        times = np.arange(len(path)) * dt

        return SingleArmTrajectory(times, JointPathContainer(positions=path))

    def plan_to_tcp_pose(
        self, start_configuration: JointConfigurationType, tcp_pose: HomogeneousMatrixType
    ) -> SingleArmTrajectory:
        start_state = JointState.from_position(
            torch.tensor(start_configuration, dtype=torch.float32)[None, :].cuda().contiguous()
        )
        goal_pose = Pose.from_matrix(tcp_pose)

        result = self.motion_gen.plan_single(start_state, goal_pose, self.motion_gen_plan_config)

        if not result.success:
            logger.warning(f"Failed to plan with status: {result.status}")
            raise NoPathFoundError(start_configuration, tcp_pose)

        path = result.interpolated_plan.position.cpu().numpy()

        logger.success(f"Successfully found path (with {len(path)} waypoints).")

        dt = result.interpolation_dt
        times = np.arange(len(path)) * dt

        return SingleArmTrajectory(times, JointPathContainer(positions=path))

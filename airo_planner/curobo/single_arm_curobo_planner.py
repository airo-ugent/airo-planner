from os import PathLike
from typing import List

import numpy as np
import torch
from airo_typing import HomogeneousMatrixType, JointConfigurationType, JointPathContainer, SingleArmTrajectory
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from loguru import logger
from scipy.spatial.transform import Rotation as R

from airo_planner import NoPathFoundError, SingleArmPlanner


class SingleArmCuroboPlanner(SingleArmPlanner):
    """Utility class for single-arm motion planning using cuRobo.
    The purpose of this class is to make working with cuRobo easier.
    """

    def __init__(
        self,
        curobo_robot_file: PathLike,
        curobo_world_file: PathLike,
        allowed_planning_time: float = 1.0,
    ):
        """Instantiate a single-arm motion planner using cuRobo.

        Args:
            curobo_robot_file: Path to the robot YAML file defining kinematics and limits.
            curobo_world_file: Path to the environment/world YAML file containing obstacles.
            allowed_planning_time: Maximum allowed planning duration in seconds.
        """
        self.tensor_args = TensorDeviceType()

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            curobo_robot_file,
            curobo_world_file,
            self.tensor_args,
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

        logger.debug("Creating MotionGen instance...")
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.optimize_dt = False
        logger.debug("Warming up MotionGen instance...")
        self.motion_gen.warmup()

    def plan_to_joint_configuration(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> SingleArmTrajectory:
        """Plan a collision-free trajectory between two joint configurations.

        Args:
            start_configuration: Starting robot joint vector.
            goal_configuration: Desired target joint vector.

        Returns:
            A SingleArmTrajectory containing timestamps and joint positions.

        Raises:
            NoPathFoundError: If planning is unsuccessful.
        """
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

        logger.debug(f"Planning took {result.total_time} seconds.")

        path = result.optimized_plan.position.cpu().numpy()

        logger.success(f"Successfully found path (with {len(path)} waypoints).")

        times = np.arange(len(path)) * result.optimized_dt.item()

        return SingleArmTrajectory(times, JointPathContainer(positions=path))

    def plan_to_joint_configurations_batched(
        self, start_configurations: List[JointConfigurationType], goal_configurations: List[JointConfigurationType]
    ) -> List[SingleArmTrajectory]:
        """Batch-plan from joint-space starts to joint-space goals. If any failures to plan occur, this function returns an empty list.

        Args:
            start_configurations: List of start joint vectors.
            goal_configurations: List of goal joint vectors.

        Returns:
            A list of SingleArmTrajectory objects. Empty if planning fails.
        """
        goal_tcp_poses = [self.forward_kinematics(q) for q in goal_configurations]
        return self.plan_to_tcp_poses_batched(start_configurations, goal_tcp_poses)

    def plan_to_tcp_pose(
        self, start_configuration: JointConfigurationType, tcp_pose: HomogeneousMatrixType
    ) -> SingleArmTrajectory:
        """Plan a motion from a joint configuration to a Cartesian TCP pose.

        Args:
            start_configuration: Initial joint configuration.
            tcp_pose: Desired end-effector homogeneous transform (4x4).

        Returns:
            A computed joint trajectory reaching the target TCP pose.

        Raises:
            NoPathFoundError: If planning fails.
        """
        start_state = JointState.from_position(
            torch.tensor(start_configuration, dtype=torch.float32)[None, :].cuda().contiguous()
        )
        goal_pose = Pose.from_matrix(tcp_pose)

        result = self.motion_gen.plan_single(start_state, goal_pose, self.motion_gen_plan_config)

        if not result.success:
            logger.warning(f"Failed to plan with status: {result.status}")
            raise NoPathFoundError(start_configuration, tcp_pose)

        logger.debug(f"Planning took {result.total_time} seconds.")

        path = result.optimized_plan.position.cpu().numpy()

        logger.success(f"Successfully found path (with {len(path)} waypoints).")

        times = np.arange(len(path)) * result.optimized_dt.item()

        return SingleArmTrajectory(times, JointPathContainer(positions=path))

    def plan_to_tcp_poses_batched(
        self, start_configurations: List[JointConfigurationType], tcp_poses: List[HomogeneousMatrixType]
    ) -> List[SingleArmTrajectory]:
        """Batch-plan from joint-space starts to Cartesian-space targets. If any failures to plan occur, this function returns an empty list.

        Args:
            start_configurations: List of starting joint configurations.
            tcp_poses: List of 4x4 end-effector target poses.

        Returns:
            List of SingleArmTrajectory objects for each successful plan.
        """
        start_states = JointState.from_position(
            torch.tensor(np.stack(start_configurations), dtype=torch.float32).cuda()
        )
        goal_states = Pose.from_batch_list(
            [Pose.from_matrix(tcp_pose).to_list() for tcp_pose in tcp_poses], self.tensor_args
        )

        result = self.motion_gen.plan_batch(start_states, goal_states, self.motion_gen_plan_config)

        trajectories = []
        if torch.all(result.success):
            logger.success("Successfully planned to all goal states.")

            for result_index in range(len(result.success)):
                path = result.optimized_plan.position[result_index].cpu().numpy()
                times = np.arange(len(path)) * result.optimized_dt[result_index].item()
                trajectories.append(SingleArmTrajectory(times, JointPathContainer(positions=path)))
        else:
            logger.warning(f"Failed to plan with status: {result.status}.")
        return trajectories

    def forward_kinematics(self, q: JointConfigurationType) -> HomogeneousMatrixType:
        """Compute forward kinematics for a joint configuration.

        Args:
            q: Joint vector.

        Returns:
            A 4x4 homogeneous transform representing the TCP pose.
        """
        crms = self.motion_gen.kinematics.get_state(torch.tensor(q, dtype=torch.float32).cuda())
        tcp_pose = np.eye(4)
        tcp_pose[:3, 3] = crms.ee_position.cpu().numpy()
        tcp_pose[:3, :3] = R.from_quat(crms.ee_quaternion.cpu().numpy(), scalar_first=True).as_matrix()
        return tcp_pose

    def inverse_kinematics(self, X_B_Tcp: HomogeneousMatrixType) -> JointConfigurationType:
        """Compute inverse kinematics for a desired TCP pose.

        Args:
            X_B_Tcp: Target TCP pose as a 4x4 transform.

        Returns:
            A joint configuration that reaches the target pose.
        """
        goal = Pose.from_matrix(X_B_Tcp)
        result = self.motion_gen.ik_solver.solve_single(goal)
        q_solution = result.solution[result.success]
        return q_solution

    def add_collider_cuboid(self, cuboid: Cuboid, update_world: bool = True) -> None:
        """Add a cuboid obstacle to the world model.

        Args:
            cuboid: The Cuboid object to add.
            update_world: If True, updates cuRobo's internal world model.
        """
        self.motion_gen.world_model.cuboid.append(cuboid)
        if update_world:
            self.update_world()

    def update_collider_cuboid(self, name: str, new_cuboid: Cuboid, update_world: bool = True) -> None:
        """Replace an existing cuboid collider with a new one.

        Args:
            name: Name of the cuboid to update.
            new_cuboid: Replacement cuboid.
            update_world: Whether to refresh cuRobo's world model.
        """
        for i, cuboid in enumerate(self.motion_gen.world_model.cuboid):
            if name == cuboid.name:
                self.motion_gen.world_model.cuboid[i] = new_cuboid
                break
        if update_world:
            self.update_world()

    def get_collider_cuboids(self) -> List[Cuboid]:
        """Return all cuboid obstacles currently in the world model.

        Returns:
            List of Cuboid objects.
        """
        return self.motion_gen.world_model.cuboid

    def set_collider_cuboids(self, cuboids: List[Cuboid], update_world: bool = True) -> None:
        """Replace all cuboid obstacles in the world model.

        Args:
            cuboids: New list of cuboid obstacles.
            update_world: Whether to refresh the cuRobo world model.
        """
        self.motion_gen.world_model.cuboid = []
        for cuboid in cuboids:
            self.motion_gen.world_model.add_obstacle(cuboid)
        if update_world:
            self.update_world()

    def update_world(self) -> None:
        """Synchronize cuRobo's world model to apply any obstacle changes."""
        self.motion_gen.update_world(self.motion_gen.world_model)

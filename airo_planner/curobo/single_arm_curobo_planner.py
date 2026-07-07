from os import PathLike
from typing import List, Optional

import numpy as np
import torch
from airo_typing import HomogeneousMatrixType, JointConfigurationType, JointPathContainer, SingleArmTrajectory
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, JointState, Pose
from loguru import logger

from airo_planner import NoPathFoundError, SingleArmPlanner


class SingleArmCuroboPlanner(SingleArmPlanner):
    """Utility class for single-arm motion planning using cuRobo (v0.8.0+, a.k.a. "cuRobo v2").
    The purpose of this class is to make working with cuRobo easier.

    Note: cuRobo v2 is a near-total rewrite of cuRobo v1 with an incompatible API and
    incompatible robot/world YAML config files (cuRobo v1 configs will not work here).
    """

    def __init__(
        self,
        curobo_robot_file: PathLike,
        curobo_world_file: PathLike,
        max_attempts: int = 5,
        max_collider_cuboids: int = 32,
        use_cuda_graph: bool = False,
    ):
        """Instantiate a single-arm motion planner using cuRobo.

        Args:
            curobo_robot_file: Path to the robot YAML file defining kinematics and limits.
            curobo_world_file: Path to the environment/world YAML file containing obstacles.
            max_attempts: Maximum number of IK/trajectory-optimization retries per planning call.
            max_collider_cuboids: Size of cuRobo's cuboid collision cache. cuRobo pre-allocates
                this at construction time, so it must cover both the cuboids already present in
                `curobo_world_file` and any you plan to add later via `add_collider_cuboid`.
            use_cuda_graph: Enable cuRobo's CUDA-graph capture for ~5x faster steady-state planning.
                A captured graph is frozen to the first solver pipeline it sees, so an instance with
                this enabled must be used for a *single* query type only (e.g. only
                `plan_to_joint_configuration`). Mixing `plan_cspace` and `plan_pose` (or IK/FK) on the
                same instance requires this to be False (the default).
        """
        self.curobo_robot_file = curobo_robot_file
        self.curobo_world_file = curobo_world_file
        self.max_attempts = max_attempts
        self.max_collider_cuboids = max_collider_cuboids
        self.use_cuda_graph = use_cuda_graph

        self.motion_planner_config = MotionPlannerCfg.create(
            robot=str(curobo_robot_file),
            scene_model=str(curobo_world_file),
            use_cuda_graph=use_cuda_graph,
            collision_cache={"cuboid": max_collider_cuboids},
        )

        logger.debug("Creating MotionPlanner instance...")
        self.motion_planner = MotionPlanner(self.motion_planner_config)
        logger.debug("Warming up MotionPlanner instance...")
        self.motion_planner.warmup()

        self._batch_planner: Optional[BatchMotionPlanner] = None
        self._batch_planner_size: Optional[int] = None

    @property
    def tool_frame(self) -> str:
        """Name of the (first) tool/end-effector frame used for TCP planning and IK/FK."""
        return self.motion_planner.tool_frames[0]

    @property
    def _scene(self) -> Scene:
        return self.motion_planner_config.scene_collision_cfg.scene_model

    def _get_batch_planner(self, batch_size: int) -> BatchMotionPlanner:
        """Get (or lazily create) a BatchMotionPlanner sized for `batch_size`.

        cuRobo v2 fixes `max_batch_size` at BatchMotionPlanner construction time, unlike
        the single-problem MotionPlanner, so a new instance is built whenever the
        requested batch size changes.
        """
        if self._batch_planner is None or self._batch_planner_size != batch_size:
            batch_config = MotionPlannerCfg.create(
                robot=str(self.curobo_robot_file),
                scene_model=str(self.curobo_world_file),
                use_cuda_graph=self.use_cuda_graph,
                max_batch_size=batch_size,
                collision_cache={"cuboid": self.max_collider_cuboids},
            )
            logger.debug(f"Creating BatchMotionPlanner instance for batch size {batch_size}...")
            self._batch_planner = BatchMotionPlanner(batch_config)
            self._batch_planner.warmup()
            self._batch_planner_size = batch_size
        return self._batch_planner

    def _to_joint_state(self, configuration: JointConfigurationType, joint_names: List[str]) -> JointState:
        return JointState.from_position(
            torch.tensor(configuration, dtype=torch.float32)[None, :].cuda().contiguous(),
            joint_names=joint_names,
        )

    def _to_goal_pose(self, tcp_pose: HomogeneousMatrixType) -> GoalToolPose:
        pose = Pose.from_matrix(tcp_pose)
        return GoalToolPose.from_poses({self.tool_frame: pose}, num_goalset=1)

    def plan_to_joint_configuration(  # type: ignore[override]
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
        joint_names = self.motion_planner.joint_names
        start_state = self._to_joint_state(start_configuration, joint_names)
        goal_state = self._to_joint_state(goal_configuration, joint_names)

        result = self.motion_planner.plan_cspace(goal_state, start_state, max_attempts=self.max_attempts)

        if result is None or not result.success.any():
            logger.warning("Failed to plan to joint configuration.")
            raise NoPathFoundError(start_configuration, goal_configuration)

        logger.debug(f"Planning took {result.total_time} seconds.")

        interpolated = result.get_interpolated_plan()
        path = interpolated.position.cpu().numpy().reshape(-1, interpolated.position.shape[-1])
        dt = float(interpolated.dt.reshape(-1)[0])
        times = np.arange(len(path)) * dt

        logger.success(f"Successfully found path (with {len(path)} waypoints).")

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
        batch_size = len(start_configurations)
        batch_planner = self._get_batch_planner(batch_size)
        joint_names = batch_planner.joint_names

        start_states = JointState.from_position(
            torch.tensor(np.stack(start_configurations), dtype=torch.float32).cuda(), joint_names=joint_names
        )
        goal_states = JointState.from_position(
            torch.tensor(np.stack(goal_configurations), dtype=torch.float32).cuda(), joint_names=joint_names
        )

        result = batch_planner.plan_cspace(goal_states, start_states, max_attempts=self.max_attempts)

        trajectories: List[SingleArmTrajectory] = []
        if result is not None and torch.all(result.success.any(dim=-1)):
            logger.success("Successfully planned to all goal states.")
            trajectories = self._trajectories_from_batch_result(result, batch_size)
        else:
            logger.warning("Failed to plan to at least one of the goal states.")
        return trajectories

    def plan_to_tcp_pose(  # type: ignore[override]
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
        start_state = self._to_joint_state(start_configuration, self.motion_planner.joint_names)
        goal_pose = self._to_goal_pose(tcp_pose)

        result = self.motion_planner.plan_pose(goal_pose, start_state, max_attempts=self.max_attempts)

        if result is None or not result.success.any():
            logger.warning("Failed to plan to TCP pose.")
            raise NoPathFoundError(start_configuration, tcp_pose)

        logger.debug(f"Planning took {result.total_time} seconds.")

        interpolated = result.get_interpolated_plan()
        path = interpolated.position.cpu().numpy().reshape(-1, interpolated.position.shape[-1])
        dt = float(interpolated.dt.reshape(-1)[0])
        times = np.arange(len(path)) * dt

        logger.success(f"Successfully found path (with {len(path)} waypoints).")

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
        batch_size = len(start_configurations)
        batch_planner = self._get_batch_planner(batch_size)
        joint_names = batch_planner.joint_names

        start_states = JointState.from_position(
            torch.tensor(np.stack(start_configurations), dtype=torch.float32).cuda(), joint_names=joint_names
        )
        goal_pose = Pose.from_matrix(np.stack(tcp_poses))
        goal_poses = GoalToolPose.from_poses({self.tool_frame: goal_pose}, num_goalset=1)

        result = batch_planner.plan_pose(goal_poses, start_states, max_attempts=self.max_attempts)

        trajectories: List[SingleArmTrajectory] = []
        if result is not None and torch.all(result.success.any(dim=-1)):
            logger.success("Successfully planned to all goal states.")
            trajectories = self._trajectories_from_batch_result(result, batch_size)
        else:
            logger.warning("Failed to plan to at least one of the goal states.")
        return trajectories

    def _trajectories_from_batch_result(self, result, batch_size: int) -> List[SingleArmTrajectory]:  # type: ignore[no-untyped-def]
        """Extract one SingleArmTrajectory per batch item from a batched TrajOptSolverResult.

        Unlike the single-problem result, batched results don't support `get_interpolated_plan()`
        (it raises for batch_size > 1), so trajectories are trimmed manually using
        `interpolated_last_tstep`.
        """
        positions_all = result.interpolated_trajectory.position.cpu().numpy()
        dt_all = result.interpolated_trajectory.dt.cpu().numpy().reshape(batch_size, -1)
        last_tsteps = result.interpolated_last_tstep.cpu().numpy().reshape(batch_size, -1)

        trajectories = []
        for i in range(batch_size):
            last_tstep = int(last_tsteps[i, 0])
            path = positions_all[i].reshape(-1, positions_all.shape[-1])[:last_tstep]
            dt = float(dt_all[i, 0])
            times = np.arange(len(path)) * dt
            trajectories.append(SingleArmTrajectory(times, JointPathContainer(positions=path)))
        return trajectories

    def forward_kinematics(self, q: JointConfigurationType) -> HomogeneousMatrixType:
        """Compute forward kinematics for a joint configuration.

        Args:
            q: Joint vector.

        Returns:
            A 4x4 homogeneous transform representing the TCP pose.
        """
        joint_state = self._to_joint_state(q, self.motion_planner.joint_names)
        kinematics_state = self.motion_planner.compute_kinematics(joint_state)
        tool_pose = kinematics_state.tool_poses.get_link_pose(self.tool_frame)
        return tool_pose.get_numpy_matrix()[0]

    def inverse_kinematics(self, X_B_Tcp: HomogeneousMatrixType) -> JointConfigurationType:
        """Compute inverse kinematics for a desired TCP pose.

        Args:
            X_B_Tcp: Target TCP pose as a 4x4 transform.

        Returns:
            The joint configurations that reach the target pose (empty if none found).
        """
        goal_pose = self._to_goal_pose(X_B_Tcp)
        result = self.motion_planner.ik_solver.solve_pose(goal_pose)
        return result.solution[result.success].cpu().numpy()

    def _remove_collider_cuboid(self, name: str) -> None:
        """Remove a cuboid by name from the scene's cuboid list.

        `Scene.remove_obstacle` only drops obstacles from the internal `.objects` list, not
        from the type-specific `.cuboid` list that `update_world` actually reads, which leads
        to spurious "already exists" errors on the next `update_world()`. Filtering `.cuboid`
        (and `.objects`, for consistency) directly works around this.
        """
        self._scene.cuboid = [c for c in (self._scene.cuboid or []) if c.name != name]
        self._scene.objects = [o for o in self._scene.objects if o.name != name]

    def add_collider_cuboid(self, cuboid: Cuboid, update_world: bool = True) -> None:
        """Add a cuboid obstacle to the world model.

        Args:
            cuboid: The Cuboid object to add.
            update_world: If True, updates cuRobo's internal world model.
        """
        self._scene.add_obstacle(cuboid)
        if update_world:
            self.update_world()

    def update_collider_cuboid(self, name: str, new_cuboid: Cuboid, update_world: bool = True) -> None:
        """Replace an existing cuboid collider with a new one.

        Args:
            name: Name of the cuboid to update.
            new_cuboid: Replacement cuboid.
            update_world: Whether to refresh cuRobo's world model.
        """
        self._remove_collider_cuboid(name)
        self._scene.add_obstacle(new_cuboid)
        if update_world:
            self.update_world()

    def get_collider_cuboids(self) -> List[Cuboid]:
        """Return all cuboid obstacles currently in the world model.

        Returns:
            List of Cuboid objects.
        """
        return list(self._scene.cuboid or [])

    def set_collider_cuboids(self, cuboids: List[Cuboid], update_world: bool = True) -> None:
        """Replace all cuboid obstacles in the world model.

        Args:
            cuboids: New list of cuboid obstacles.
            update_world: Whether to refresh the cuRobo world model.
        """
        for existing_cuboid in self.get_collider_cuboids():
            self._remove_collider_cuboid(existing_cuboid.name)
        for cuboid in cuboids:
            self._scene.add_obstacle(cuboid)
        if update_world:
            self.update_world()

    def update_world(self) -> None:
        """Synchronize cuRobo's world model to apply any obstacle changes.

        cuRobo's live collision cache is additive (`update_world` fails if an obstacle name
        is already loaded), so the cache is cleared first and fully repopulated from the
        current `Scene` every time.
        """
        self.motion_planner.clear_scene_cache()
        self.motion_planner.update_world(self._scene)
        if self._batch_planner is not None:
            self._batch_planner.clear_scene_cache()
            self._batch_planner.update_world(self._scene)

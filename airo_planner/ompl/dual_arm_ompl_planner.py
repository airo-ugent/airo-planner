import numpy as np
from airo_typing import (
    HomogeneousMatrixType,
    InverseKinematicsFunctionType,
    JointConfigurationCheckerType,
    JointConfigurationType,
    JointPathType,
)

from airo_planner import (
    DualArmPlanner,
    JointBoundsType,
    SingleArmOmplPlanner,
    concatenate_joint_bounds,
    create_simple_setup,
    solve_and_smooth_path,
    stack_joints,
    state_to_ompl,
    uniform_symmetric_joint_bounds,
)


class DualArmOmplPlanner(DualArmPlanner):
    def __init__(
        self,
        is_state_valid_fn: JointConfigurationCheckerType,
        joint_bounds_left: JointBoundsType | None = None,
        joint_bounds_right: JointBoundsType | None = None,
        inverse_kinematics_left_fn: InverseKinematicsFunctionType | None = None,
        inverse_kinematics_right_fn: InverseKinematicsFunctionType | None = None,
        degrees_of_freedom_left: int = 6,
        degrees_of_freedom_right: int = 6,
    ):
        self.is_state_valid_fn = is_state_valid_fn
        self.inverse_kinematics_left_fn = inverse_kinematics_left_fn
        self.inverse_kinematics_right_fn = inverse_kinematics_right_fn

        # Combine the joint bounds for the left and right arm for simplicity
        self.degrees_of_freedom: int = degrees_of_freedom_left + degrees_of_freedom_right

        self._initialize_joint_bounds(
            joint_bounds_left, joint_bounds_right, degrees_of_freedom_left, degrees_of_freedom_right
        )

        # The OMPL SimpleSetup for dual arm plannig is exactly the same as for single arm planning
        self._simple_setup = create_simple_setup(self.is_state_valid_fn, self._joint_bounds)

        # We use SingleArmOmplPlanner to handle planning for a single arm requests
        # Note that we (re)create these when start and goal config are set, because
        # their is_state_valid_fn needs to be updated.
        self._single_arm_planner_left: SingleArmOmplPlanner | None = None
        self._single_arm_planner_right: SingleArmOmplPlanner | None = None

    def _initialize_joint_bounds(
        self,
        joint_bounds_left: JointBoundsType | None,
        joint_bounds_right: JointBoundsType | None,
        degrees_of_freedom_left: int,
        degrees_of_freedom_right: int,
    ) -> None:
        if joint_bounds_left is None:
            joint_bounds_left_ = uniform_symmetric_joint_bounds(degrees_of_freedom_left)
        else:
            joint_bounds_left_ = joint_bounds_left

        if joint_bounds_right is None:
            joint_bounds_right_ = uniform_symmetric_joint_bounds(degrees_of_freedom_right)
        else:
            joint_bounds_right_ = joint_bounds_right
        joint_bounds = concatenate_joint_bounds([joint_bounds_left_, joint_bounds_right_])

        self._joint_bounds = joint_bounds
        self._joint_bounds_left = joint_bounds_left_
        self._joint_bounds_right = joint_bounds_right_

    def _set_start_and_goal_configurations(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType,
        goal_configuration_right: JointConfigurationType,
    ) -> None:
        # Set the starts and goals for dual arm planning
        space = self._simple_setup.getStateSpace()
        start_configuration = np.concatenate([start_configuration_left, start_configuration_right])
        goal_configuration = np.concatenate([goal_configuration_left, goal_configuration_right])
        start_state = state_to_ompl(start_configuration, space)
        goal_state = state_to_ompl(goal_configuration, space)
        self._simple_setup.setStartAndGoalStates(start_state, goal_state)

        # Replace single arm planners for the left and right arm
        def is_left_state_valid_fn(left_state: JointConfigurationType) -> bool:
            return self.is_state_valid_fn(np.concatenate((left_state, start_configuration_right)))

        def is_right_state_valid_fn(right_state: JointConfigurationType) -> bool:
            return self.is_state_valid_fn(np.concatenate((start_configuration_left, right_state)))

        self._single_arm_planner_left = SingleArmOmplPlanner(
            is_left_state_valid_fn,
            self._joint_bounds_left,
            self.inverse_kinematics_left_fn,
        )

        self._single_arm_planner_right = SingleArmOmplPlanner(
            is_right_state_valid_fn,
            self._joint_bounds_right,
            self.inverse_kinematics_right_fn,
        )

        self._single_arm_planner_left._set_start_and_goal_configurations(
            start_configuration_left, goal_configuration_left
        )

        self._single_arm_planner_right._set_start_and_goal_configurations(
            start_configuration_right, goal_configuration_right
        )

    def _plan_to_joint_configuration_dual_arm(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType,
        goal_configuration_right: JointConfigurationType,
    ) -> JointPathType | None:

        # Set start and goal
        simple_setup = self._simple_setup
        simple_setup.clear()
        self._set_start_and_goal_configurations(
            start_configuration_left, start_configuration_right, goal_configuration_left, goal_configuration_right
        )

        # Run planning algorithm
        path = solve_and_smooth_path(simple_setup)
        return path

    def _plan_to_joint_configuration_left_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType,
    ) -> JointPathType | None:
        # Set right goal to right start configuration
        self._set_start_and_goal_configurations(
            start_configuration_left, start_configuration_right, goal_configuration_left, start_configuration_right
        )

        assert self._single_arm_planner_left is not None  # Mypy needs this to be explicit

        left_path = self._single_arm_planner_left.plan_to_joint_configuration(
            start_configuration_left, goal_configuration_left
        )

        if left_path is None:
            return None

        path = stack_joints(left_path, start_configuration_right)
        return path

    def _plan_to_joint_configuration_right_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_right: JointConfigurationType,
    ) -> JointPathType | None:
        # Set left goal to left start configuration
        self._set_start_and_goal_configurations(
            start_configuration_left, start_configuration_right, start_configuration_left, goal_configuration_right
        )

        assert self._single_arm_planner_right is not None  # Mypy needs this to be explicit

        right_path = self._single_arm_planner_right.plan_to_joint_configuration(
            start_configuration_right, goal_configuration_right
        )

        if right_path is None:
            return None

        path = stack_joints(start_configuration_left, right_path)
        return path

    def plan_to_joint_configuration(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType | None,
        goal_configuration_right: JointConfigurationType | None,
    ) -> JointPathType | None:
        if goal_configuration_left is None and goal_configuration_right is None:
            raise ValueError("A goal configurations must be specified for at least one of the arms.")

        if goal_configuration_right is None:
            assert goal_configuration_left is not None  # Mypy needs this to be explicit
            path = self._plan_to_joint_configuration_left_arm_only(
                start_configuration_left, start_configuration_right, goal_configuration_left
            )
            return path

        if goal_configuration_left is None:
            assert goal_configuration_right is not None  # Mypy needs this to be explicit
            path = self._plan_to_joint_configuration_right_arm_only(
                start_configuration_left, start_configuration_right, goal_configuration_right
            )
            return path

        # Do 12 DoF dual arm planning
        path = self._plan_to_joint_configuration_dual_arm(
            start_configuration_left, start_configuration_right, goal_configuration_left, goal_configuration_right
        )
        return path

    def plan_to_tcp_pose(
        self,
        start_configuration_left: HomogeneousMatrixType,
        start_configuration_right: HomogeneousMatrixType,
        tcp_pose_left: HomogeneousMatrixType | None,
        tcp_pose_right: HomogeneousMatrixType | None,
    ) -> JointPathType | None:
        if tcp_pose_left is None and tcp_pose_right is None:
            raise ValueError("A goal TCP pose must be specified for at least one of the arms.")
        return None

    # def _plan_to_tcp_pose_left_arm_only(
    #     self,
    #     start_configuration_left: JointConfigurationType,
    #     start_configuration_right: JointConfigurationType,
    #     tcp_pose_left_in_base: HomogeneousMatrixType,
    #     desirable_goal_configurations_left: List[JointConfigurationType] | None = None,
    # ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
    #     # Set right goal to right start configuration
    #     self._set_start_and_goal_configurations(
    #         start_configuration_left, start_configuration_right, start_configuration_left, start_configuration_right
    #     )

    #     left_path = self._single_arm_planner_left.plan_to_tcp_pose(
    #         start_configuration_left, tcp_pose_left_in_base, desirable_goal_configurations_left
    #     )

    #     if left_path is None:
    #         return None

    #     path = [(left_state, start_configuration_right) for left_state in left_path]
    #     return path

    # def _plan_to_tcp_pose_right_arm_only(
    #     self,
    #     start_configuration_left: JointConfigurationType,
    #     start_configuration_right: JointConfigurationType,
    #     tcp_pose_right_in_base: HomogeneousMatrixType,
    #     desirable_goal_configurations_right: List[JointConfigurationType] | None = None,
    # ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
    #     # Set left goal to left start configuration
    #     self._set_start_and_goal_configurations(
    #         start_configuration_left, start_configuration_right, start_configuration_left, start_configuration_right
    #     )

    #     right_path = self._single_arm_planner_right.plan_to_tcp_pose(
    #         start_configuration_right, tcp_pose_right_in_base, desirable_goal_configurations_right
    #     )

    #     if right_path is None:
    #         return None

    #     path = [(start_configuration_left, right_state) for right_state in right_path]
    #     return path

    # def _plan_to_tcp_pose_dual_arm(  # noqa: C901
    #     self,
    #     start_configuration_left: JointConfigurationType,
    #     start_configuration_right: JointConfigurationType,
    #     tcp_pose_left_in_base: HomogeneousMatrixType,
    #     tcp_pose_right_in_base: HomogeneousMatrixType,
    # ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
    #     if self.inverse_kinematics_left_fn is None or self.inverse_kinematics_right_fn is None:
    #         logger.info(
    #             "Planning to left and right TCP poses attempted but inverse_kinematics_fn was not provided for both arms, returing None."
    #         )

    #     # MyPy needs this to be explicit
    #     assert self.inverse_kinematics_left_fn is not None
    #     assert self.inverse_kinematics_right_fn is not None

    #     # 1. do IK for both arms
    #     ik_solutions_left = self.inverse_kinematics_left_fn(tcp_pose_left_in_base)
    #     ik_solutions_right = self.inverse_kinematics_right_fn(tcp_pose_right_in_base)

    #     if ik_solutions_left is None or len(ik_solutions_left) == 0:
    #         logger.info("IK for left arm returned no solutions, returning None.")
    #         return None
    #     else:
    #         logger.info(f"Found {len(ik_solutions_left)} IK solutions for left arm.")

    #     if ik_solutions_right is None or len(ik_solutions_right) == 0:
    #         logger.info("IK for right arm returned no solutions, returning None.")
    #         return None
    #     else:
    #         logger.info(f"Found {len(ik_solutions_right)} IK solutions for right arm.")

    #     # 2. filter out IK solutions that are outside the joint bounds
    #     ik_solutions_in_bounds_left = []
    #     for ik_solution in ik_solutions_left:
    #         if np.all(ik_solution >= self._joint_bounds_left[0]) and np.all(ik_solution <= self._joint_bounds_left[1]):
    #             ik_solutions_in_bounds_left.append(ik_solution)

    #     ik_solutions_in_bounds_right = []
    #     for ik_solution in ik_solutions_right:
    #         if np.all(ik_solution >= self._joint_bounds_right[0]) and np.all(ik_solution <= self._joint_bounds_right[1]):
    #             ik_solutions_in_bounds_right.append(ik_solution)

    #     if len(ik_solutions_in_bounds_left) == 0:
    #         logger.info("No IK solutions for left arm are within the joint bounds, returning None.")
    #         return None
    #     else:
    #         logger.info(
    #             f"Found {len(ik_solutions_in_bounds_left)}/{len(ik_solutions_left)} IK solutions within the joint bounds for left arm."
    #         )

    #     if len(ik_solutions_in_bounds_right) == 0:
    #         logger.info("No IK solutions for right arm are within the joint bounds, returning None.")
    #         return None
    #     else:
    #         logger.info(
    #             f"Found {len(ik_solutions_in_bounds_right)}/{len(ik_solutions_right)} IK solutions within the joint bounds for right arm."
    #         )

    #     # 2. create all goal pairs
    #     goal_configurations = []
    #     for ik_solution_left in ik_solutions_in_bounds_left:
    #         for ik_solution_right in ik_solutions_in_bounds_right:
    #             goal_configurations.append(np.concatenate((ik_solution_left, ik_solution_right)))

    #     n_goal_configurations = len(goal_configurations)

    #     # 3. filter out invalid goal pairs
    #     goal_configurations_valid = [s for s in goal_configurations if self.is_state_valid_fn(s)]
    #     n_valid_goal_configurations = len(goal_configurations_valid)

    #     if n_valid_goal_configurations == 0:
    #         logger.info(f"All {n_goal_configurations} goal pairs are invalid, returning None.")
    #         return None
    #     else:
    #         logger.info(f"Found {n_valid_goal_configurations}/{n_goal_configurations} valid goal pairs.")

    #     # 4. for each pair, plan to the goal pair
    #     paths = []
    #     path_lengths: list[float] = []
    #     for goal_configuration in goal_configurations_valid:
    #         path = self.plan_to_joint_configuration(
    #             start_configuration_left, start_configuration_right, goal_configuration[:6], goal_configuration[6:]
    #         )
    #         if path is not None:
    #             paths.append(path)
    #             if self._path_length_dual is None:
    #                 raise ValueError("For devs: path length should not be None at this point")
    #             path_lengths.append(self._path_length_dual)

    #     if len(paths) == 0:
    #         logger.info("No paths founds towards any goal pairs, returning None.")
    #         return None

    #     logger.info(f"Found {len(paths)} paths towards goal pairs.")

    #     # 5. return the shortest path among all the planned paths
    #     shortest_path_idx = np.argmin(path_lengths)
    #     shortest_path = paths[shortest_path_idx]
    #     return shortest_path

    # def plan_to_tcp_pose(
    #     self,
    #     start_configuration_left: JointConfigurationType,
    #     start_configuration_right: JointConfigurationType,
    #     tcp_pose_left_in_base: HomogeneousMatrixType | None,
    #     tcp_pose_right_in_base: HomogeneousMatrixType | None,
    #     desirable_goal_configurations_left: List[JointConfigurationType] | None = None,
    #     desirable_goal_configurations_right: List[JointConfigurationType] | None = None,
    # ) -> List[Tuple[JointConfigurationType, JointConfigurationType]] | None:
    #     if tcp_pose_left_in_base is None and tcp_pose_right_in_base is None:
    #         raise ValueError("A goal TCP pose must be specified for at least one of the arms.")

    #     if tcp_pose_right_in_base is None:
    #         assert tcp_pose_left_in_base is not None  # Mypy needs this to be explicit
    #         path = self._plan_to_tcp_pose_left_arm_only(
    #             start_configuration_left,
    #             start_configuration_right,
    #             tcp_pose_left_in_base,
    #             desirable_goal_configurations_left,
    #         )
    #         return path

    #     if tcp_pose_left_in_base is None:
    #         assert tcp_pose_right_in_base is not None  # Mypy needs this to be explicit
    #         path = self._plan_to_tcp_pose_right_arm_only(
    #             start_configuration_left,
    #             start_configuration_right,
    #             tcp_pose_right_in_base,
    #             desirable_goal_configurations_right,
    #         )
    #         return path

    #     # TODO use desirable_goal_configurations for dual arm planning
    #     if desirable_goal_configurations_left is not None or desirable_goal_configurations_right is not None:
    #         logger.warning(
    #             "Desirable goal configurations are not implemented yet for dual arm planning, ignoring them."
    #         )

    #     path = self._plan_to_tcp_pose_dual_arm(
    #         start_configuration_left, start_configuration_right, tcp_pose_left_in_base, tcp_pose_right_in_base
    #     )

    #     return path

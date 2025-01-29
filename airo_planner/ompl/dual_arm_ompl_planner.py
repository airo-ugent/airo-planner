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
    DualArmPlanner,
    JointBoundsType,
    JointConfigurationsModifierType,
    JointPathChooserType,
    MultipleGoalPlanner,
    NoInverseKinematicsSolutionsError,
    NoPathFoundError,
    SingleArmOmplPlanner,
    choose_shortest_path,
    concatenate_joint_bounds,
    convert_dual_arm_joints_modifier_to_single_arm,
    convert_dual_arm_path_chooser_to_single_arm,
    create_simple_setup,
    solve_and_smooth_path,
    stack_joints,
    state_to_ompl,
    uniform_symmetric_joint_bounds,
)


class DualArmOmplPlanner(DualArmPlanner, MultipleGoalPlanner):
    def __init__(
        self,
        is_state_valid_fn: JointConfigurationCheckerType,
        joint_bounds_left: JointBoundsType | None = None,
        joint_bounds_right: JointBoundsType | None = None,
        inverse_kinematics_left_fn: InverseKinematicsFunctionType | None = None,
        inverse_kinematics_right_fn: InverseKinematicsFunctionType | None = None,
        filter_goal_configurations_fn: JointConfigurationsModifierType | None = None,
        rank_goal_configurations_fn: JointConfigurationsModifierType | None = None,
        choose_path_fn: JointPathChooserType = choose_shortest_path,
        degrees_of_freedom_left: int = 6,
        degrees_of_freedom_right: int = 6,
        allowed_planning_time: float = 1.0,
    ):

        # Planner properties
        self.is_state_valid_fn = is_state_valid_fn
        self.inverse_kinematics_left_fn = inverse_kinematics_left_fn
        self.inverse_kinematics_right_fn = inverse_kinematics_right_fn
        self.allowed_planning_time = allowed_planning_time

        self.degrees_of_freedom: int = degrees_of_freedom_left + degrees_of_freedom_right
        self.degrees_of_freedom_left: int = degrees_of_freedom_left
        self.degrees_of_freedom_right: int = degrees_of_freedom_right

        # Combine the joint bounds for the left and right arm for simplicity
        self._initialize_joint_bounds(
            joint_bounds_left, joint_bounds_right, degrees_of_freedom_left, degrees_of_freedom_right
        )

        # Initialize MultipleGoalPlanner
        super().__init__(
            is_state_valid_fn,
            self.joint_bounds,
            filter_goal_configurations_fn,
            rank_goal_configurations_fn,
            choose_path_fn,
        )

        # The OMPL SimpleSetup for dual arm plannig is exactly the same as for single arm planning
        self._simple_setup = create_simple_setup(self.is_state_valid_fn, self.joint_bounds)

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
        joint_bounds = concatenate_joint_bounds(joint_bounds_left_, joint_bounds_right_)

        self.joint_bounds = joint_bounds
        self.joint_bounds_left = joint_bounds_left_
        self.joint_bounds_right = joint_bounds_right_

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

    def _create_single_arm_planners(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
    ) -> None:
        """Create single arm planners for the left and right arm.

        The implementation of this function is quite tricky, because we have to convert all the functions that work
        on dual arm joint configurations, to work on single arm joint configurations. As we do this, we freeze
        the configuration of one of the arms to its start configuration.

        TODO: consider creating a "sinlge_arm_ompl_planner_from_dual_arm_planner()" helper function or similar.
        """

        # Replace single arm planners for the left and right arm
        # We basically have to freeze the start configuration of one of the arms in all the helper functions
        def is_left_state_valid_fn(left_state: JointConfigurationType) -> bool:
            return self.is_state_valid_fn(stack_joints(left_state, start_configuration_right))

        def is_right_state_valid_fn(right_state: JointConfigurationType) -> bool:
            return self.is_state_valid_fn(stack_joints(start_configuration_left, right_state))

        filter_goal_configurations_fn_left = None
        filter_goal_configurations_fn_right = None
        rank_goal_configurations_fn_left = None
        rank_goal_configurations_fn_right = None

        if self.filter_goal_configurations_fn is not None:
            filter_goal_configurations_fn_left = convert_dual_arm_joints_modifier_to_single_arm(
                self.filter_goal_configurations_fn,
                start_configuration_right,
                self.degrees_of_freedom_left,
                to_left=True,
            )
            filter_goal_configurations_fn_right = convert_dual_arm_joints_modifier_to_single_arm(
                self.filter_goal_configurations_fn,
                start_configuration_left,
                self.degrees_of_freedom_left,
                to_left=False,
            )

        if self.rank_goal_configurations_fn is not None:
            rank_goal_configurations_fn_left = convert_dual_arm_joints_modifier_to_single_arm(
                self.rank_goal_configurations_fn,
                start_configuration_right,
                self.degrees_of_freedom_left,
                to_left=True,
            )
            rank_goal_configurations_fn_right = convert_dual_arm_joints_modifier_to_single_arm(
                self.rank_goal_configurations_fn,
                start_configuration_left,
                self.degrees_of_freedom_left,
                to_left=False,
            )

        if self.choose_path_fn is not None:
            choose_path_fn_left = convert_dual_arm_path_chooser_to_single_arm(
                self.choose_path_fn,
                start_configuration_right,
                self.degrees_of_freedom_left,
                to_left=True,
            )
            choose_path_fn_right = convert_dual_arm_path_chooser_to_single_arm(
                self.choose_path_fn,
                start_configuration_left,
                self.degrees_of_freedom_left,
                to_left=False,
            )

        self._single_arm_planner_left = SingleArmOmplPlanner(
            is_left_state_valid_fn,
            self.joint_bounds_left,
            self.inverse_kinematics_left_fn,
            filter_goal_configurations_fn_left,
            rank_goal_configurations_fn_left,
            choose_path_fn_left,
            self.degrees_of_freedom_left,
        )

        self._single_arm_planner_right = SingleArmOmplPlanner(
            is_right_state_valid_fn,
            self.joint_bounds_right,
            self.inverse_kinematics_right_fn,
            filter_goal_configurations_fn_right,
            rank_goal_configurations_fn_right,
            choose_path_fn_right,
            self.degrees_of_freedom_right,
        )

    def _plan_to_joint_configuration_dual_arm(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType,
        goal_configuration_right: JointConfigurationType,
    ) -> JointPathType:

        # Set start and goal
        simple_setup = self._simple_setup
        simple_setup.clear()
        self._set_start_and_goal_configurations(
            start_configuration_left, start_configuration_right, goal_configuration_left, goal_configuration_right
        )

        # Run planning algorithm
        path = solve_and_smooth_path(simple_setup, self.allowed_planning_time)

        if path is None:
            start_configuration = stack_joints(start_configuration_left, start_configuration_right)
            goal_configuration = stack_joints(goal_configuration_left, goal_configuration_right)
            raise NoPathFoundError(start_configuration, goal_configuration)

        return path

    def _plan_to_joint_configuration_left_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType,
    ) -> JointPathType:
        self._create_single_arm_planners(
            start_configuration_left,
            start_configuration_right,
        )
        assert self._single_arm_planner_left is not None  # Mypy needs this to be explicit

        left_path = self._single_arm_planner_left.plan_to_joint_configuration(
            start_configuration_left, goal_configuration_left
        )

        if left_path is None:
            raise NoPathFoundError(start_configuration_left, goal_configuration_left)

        path = stack_joints(left_path, start_configuration_right)
        return path

    def _plan_to_joint_configuration_right_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_right: JointConfigurationType,
    ) -> JointPathType:
        self._create_single_arm_planners(
            start_configuration_left,
            start_configuration_right,
        )
        assert self._single_arm_planner_right is not None  # Mypy needs this to be explicit

        right_path = self._single_arm_planner_right.plan_to_joint_configuration(
            start_configuration_right, goal_configuration_right
        )

        if right_path is None:
            NoPathFoundError(start_configuration_right, goal_configuration_right)

        path = stack_joints(start_configuration_left, right_path)
        return path

    def plan_to_joint_configuration(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType | None,
        goal_configuration_right: JointConfigurationType | None,
    ) -> JointPathType:
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

    def _plan_to_joint_configuration_stacked(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> JointPathType:
        """This function is used by the MultipleGoalPlanner class."""
        start_configuration_left = start_configuration[: self.degrees_of_freedom_left]
        start_configuration_right = start_configuration[self.degrees_of_freedom_left :]
        goal_configuration_left = goal_configuration[: self.degrees_of_freedom_left]
        goal_configuration_right = goal_configuration[self.degrees_of_freedom_left :]
        return self._plan_to_joint_configuration_dual_arm(
            start_configuration_left, start_configuration_right, goal_configuration_left, goal_configuration_right
        )

    def _plan_to_tcp_pose_left_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        tcp_pose_left: HomogeneousMatrixType,
    ) -> JointPathType:
        self._create_single_arm_planners(
            start_configuration_left,
            start_configuration_right,
        )

        assert self._single_arm_planner_left is not None  # Mypy needs this to be explicit

        left_path = self._single_arm_planner_left.plan_to_tcp_pose(start_configuration_left, tcp_pose_left)
        path = stack_joints(left_path, start_configuration_right)
        return path

    def _plan_to_tcp_pose_right_arm_only(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        tcp_pose_right: HomogeneousMatrixType,
    ) -> JointPathType:
        self._create_single_arm_planners(
            start_configuration_left,
            start_configuration_right,
        )

        assert self._single_arm_planner_right is not None

        right_path = self._single_arm_planner_right.plan_to_tcp_pose(start_configuration_right, tcp_pose_right)
        path = stack_joints(start_configuration_left, right_path)
        return path

    def _calculate_ik_solutions(
        self, tcp_pose_left: HomogeneousMatrixType, tcp_pose_right: HomogeneousMatrixType
    ) -> list[JointConfigurationType]:
        """Used by plan_to_tcp_pose() to calculate IK solutions when planning to two TCP poses.

        TODO: This function is very similar to the one in SingleArmPlanner. Consider sharing some logic.
        """

        if self.inverse_kinematics_left_fn is None:
            raise AttributeError(
                "No inverse kinematics function for left arm was provided (required for planning to TCP poses)."
            )

        if self.inverse_kinematics_right_fn is None:
            raise AttributeError(
                "No inverse kinematics function for right arm was provided (required for planning to TCP poses)."
            )

        ik_solutions_left = self.inverse_kinematics_left_fn(tcp_pose_left)
        ik_solutions_right = self.inverse_kinematics_right_fn(tcp_pose_right)

        ik_solutions_left = [solution.squeeze() for solution in ik_solutions_left]
        ik_solutions_right = [solution.squeeze() for solution in ik_solutions_right]

        if ik_solutions_left is None or len(ik_solutions_left) == 0:
            raise NoInverseKinematicsSolutionsError(tcp_pose_left)

        if ik_solutions_right is None or len(ik_solutions_right) == 0:
            raise NoInverseKinematicsSolutionsError(tcp_pose_right)

        ik_solution_pairs_stacked = []
        for ik_solution_left in ik_solutions_left:
            for ik_solution_right in ik_solutions_right:
                ik_solution_pairs_stacked.append(stack_joints(ik_solution_left, ik_solution_right))

        logger.info(
            f"IK returned {len(ik_solution_pairs_stacked)} pairs of IK solutions ({len(ik_solutions_left)} x {len(ik_solutions_right)})."
        )
        return ik_solution_pairs_stacked

    def plan_to_tcp_pose(
        self,
        start_configuration_left: HomogeneousMatrixType,
        start_configuration_right: HomogeneousMatrixType,
        tcp_pose_left: HomogeneousMatrixType | None,
        tcp_pose_right: HomogeneousMatrixType | None,
    ) -> JointPathType:
        if tcp_pose_left is None and tcp_pose_right is None:
            raise ValueError("A goal TCP pose must be specified for at least one of the arms.")

        # 0. TODO if one of the two if None, use the single arm planners
        if tcp_pose_right is None:
            assert tcp_pose_left is not None  # Mypy needs this to be explicit
            return self._plan_to_tcp_pose_left_arm_only(
                start_configuration_left,
                start_configuration_right,
                tcp_pose_left,
            )

        if tcp_pose_left is None:
            assert tcp_pose_right is not None
            return self._plan_to_tcp_pose_right_arm_only(
                start_configuration_left,
                start_configuration_right,
                tcp_pose_right,
            )

        # 1.1. stack start_configuration
        start_configuration = stack_joints(start_configuration_left, start_configuration_right)

        # 1.2. stack goal_configurations (combine ik_solutions_left and ik_solutions_right)
        goal_configurations = self._calculate_ik_solutions(tcp_pose_left, tcp_pose_right)

        # all steps below are in multiple_goal_planner.py
        # 2. Remove IK solutions that are outside the joint bounds
        # 3. Create all goal pairs (use stack_joints() to concatenate left and right arm configurations)
        # from this point on, all function from the single arm case can be reused
        # 4. Remove invalid goal pairs
        # 5. Apply the user's filter on goal pairs (stacked)
        # 6. Rank the goal pairs with the user's ranking function
        # 7. Plan with early stop or exhaustively
        solution_path = self.plan_to_ik_solutions(start_configuration, goal_configurations)
        return solution_path

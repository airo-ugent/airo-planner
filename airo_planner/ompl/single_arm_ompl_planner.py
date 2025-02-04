from airo_typing import (
    HomogeneousMatrixType,
    InverseKinematicsFunctionType,
    JointConfigurationCheckerType,
    JointConfigurationType,
    JointPathType,
)
from loguru import logger

from airo_planner import (
    GoalConfigurationOutOfBoundsError,
    InvalidGoalConfigurationError,
    InvalidStartConfigurationError,
    JointBoundsType,
    JointConfigurationsModifierType,
    JointPathChooserType,
    MultipleGoalPlanner,
    NoInverseKinematicsSolutionsError,
    NoPathFoundError,
    SingleArmPlanner,
    StartConfigurationOutOfBoundsError,
    choose_shortest_path,
    create_simple_setup,
    is_within_bounds,
    solve_and_smooth_path,
    state_to_ompl,
    uniform_symmetric_joint_bounds,
)


class SingleArmOmplPlanner(SingleArmPlanner, MultipleGoalPlanner):
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
        allowed_planning_time: float = 1.0,
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

        # Planning parameters
        self.allowed_planning_time = allowed_planning_time
        self._degrees_of_freedom = degrees_of_freedom
        if joint_bounds is None:
            self.joint_bounds = uniform_symmetric_joint_bounds(self._degrees_of_freedom)
        else:
            self.joint_bounds = joint_bounds

        # Initialize MultipleGoalPlanner
        super().__init__(
            is_state_valid_fn,
            self.joint_bounds,
            filter_goal_configurations_fn,
            rank_goal_configurations_fn,
            choose_path_fn,
        )

        # TODO consider creating a JointSpaceOmplPlanner class that can be used as base for both single and dual arm planning
        # The SingleArmOmplPlanner then becomes even simpler, it bascially only handles IK and planning to TCP poses.
        self._simple_setup = create_simple_setup(self.is_state_valid_fn, self.joint_bounds)

        # Debug attributes
        self._ik_solutions: list[JointConfigurationType] | None = None

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

        path = solve_and_smooth_path(simple_setup, self.allowed_planning_time)

        if path is None:
            raise NoPathFoundError(start_configuration, goal_configuration)

        logger.success(f"Successfully found path (with {len(path)} waypoints).")

        return path

    def _plan_to_joint_configuration_stacked(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> JointPathType:
        """This function's only purpose is to called used by the MultipleGoalPlanner class.

        MultipleGoalPlanner can't call plan_to_joint_configuration for both signal arm planning and dual arm planning
        because the parameters are different. This function is a workaround for that.
        """
        return self.plan_to_joint_configuration(start_configuration, goal_configuration)

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

    def plan_to_tcp_pose(
        self, start_configuration: JointConfigurationType, tcp_pose: HomogeneousMatrixType
    ) -> JointPathType:
        ik_solutions = self._calculate_ik_solutions(tcp_pose)
        solution_path = self.plan_to_ik_solutions(start_configuration, ik_solutions)
        return solution_path

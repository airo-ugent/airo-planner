from airo_typing import HomogeneousMatrixType, JointConfigurationType

from airo_planner.joint_bounds import JointBoundsType


class PlannerError(RuntimeError):
    """Base class for all things that can go wrong during planning.
    Mostly intended for problems that can happen, but are not per se
    the user's fault, e.g. no valid IK solutions, no valid paths etc.

    Note that not all planner problems need to be a subclass of this
    error, e.g. you can still use ValueError, AttributeError etc.
    """


class NoPathFoundError(PlannerError):
    """Raised when no path is found between configurations that are valid and
    within bounds. This means either the underlying planning algorithm ran out
    of time, is incomplete, or there is an obstacle in configuration space that
    prevents connecting start and goal.

    Note that this exception can also be raised with multiple goals to signal
    that no path was found to any of the goals.
    """

    def __init__(
        self,
        start_configuration: JointConfigurationType,
        goal_configurations: JointConfigurationType | list[JointConfigurationType] | None = None,
        message: str | None = None,
    ):
        if message is None:
            message_lines = [
                "No path found between the start and goal configuration:",
                f"Start configuration: {start_configuration}",
            ]
            if isinstance(goal_configurations, list):
                message_lines.append("Goal configurations:")
                for goal_configuration in goal_configurations:
                    message_lines.append(f"  {goal_configuration}")
            else:
                message_lines.append(f"Goal configuration: {goal_configurations}")

            message = "\n".join(message_lines)
        super().__init__(message)


class InvalidConfigurationError(PlannerError, ValueError):
    """Base class for errors related to invalid configurations."""

    def __init__(self, joint_configuration: JointConfigurationType, message: str | None = None):
        if message is None:
            message = f"The provided configuration is invalid: {joint_configuration}"
        super().__init__(message)
        self.joint_configuration = joint_configuration


class InvalidStartConfigurationError(InvalidConfigurationError):
    """Raised when the start configuration is invalid."""


class InvalidGoalConfigurationError(InvalidConfigurationError):
    """Raised when the goal configuration is invalid."""


class JointOutOfBoundsError(PlannerError, ValueError):
    """Raised when a joint configuration exceeds specified bounds."""

    def __init__(
        self, joint_configuration: JointConfigurationType, joint_bounds: JointBoundsType, message: str | None = None
    ):
        if message is None:
            violations = self._find_violations(joint_configuration, joint_bounds)
            message = self._generate_detailed_message(violations)

        super().__init__(message)
        self.joint_configuration = joint_configuration
        self.joint_bounds = joint_bounds

    def _find_violations(
        self, joint_configuration: JointConfigurationType, joint_bounds: JointBoundsType
    ) -> list[tuple[int, float, float, float]]:
        violations = []
        joints_lower, joints_upper = joint_bounds
        for i, (value, lower, upper) in enumerate(zip(joint_configuration, joints_lower, joints_upper)):
            if not (lower <= value <= upper):
                violations.append((i, value, lower, upper))
        return violations

    def _generate_detailed_message(self, violations: list[tuple[int, float, float, float]]) -> str:
        message_lines = ["Joint values exceed bounds:"]
        for index, value, lower, upper in violations:
            message_lines.append(f"  Joint {index}: Value {value} is outside the range [{lower}, {upper}]")
        return "\n".join(message_lines)


class StartConfigurationOutOfBoundsError(JointOutOfBoundsError):
    """Raised when the start configuration is out of bounds."""


class GoalConfigurationOutOfBoundsError(JointOutOfBoundsError):
    """Raised when the goal configuration is out of bounds."""


class NoInverseKinematicsSolutionsError(PlannerError):
    """Raised when no IK solutions are found, can happen when the pose is
    unreachable or the inverse kinematics function is incomplete."""

    def __init__(self, tcp_pose: HomogeneousMatrixType, message: str | None = None):
        if message is None:
            message = f"No IK solutions found for the TCP pose:\n{tcp_pose}"
        super().__init__(message)
        self.tcp_pose = tcp_pose


class NoInverseKinematicsSolutionsWithinBoundsError(PlannerError):
    """Raised when all IK solutions are out of bounds."""

    def __init__(
        self, ik_solutions: list[JointConfigurationType], joint_bounds: JointBoundsType, message: str | None = None
    ):
        if message is None:
            message_lines = [
                "All inverse kinematics solutions are out of bounds:",
                f"Lower bounds: {joint_bounds[0]}",
                f"Upper bounds: {joint_bounds[1]}",
            ]
            message_lines.append("Solutions:")
            for solution in ik_solutions:
                message_lines.append(f"  {solution}")
            message = "\n".join(message_lines)
        super().__init__(message)
        self.ik_solutions = ik_solutions


class NoValidInverseKinematicsSolutionsError(PlannerError):
    """Raised when no valid IK solutions are found."""

    def __init__(self, ik_solutions: list[JointConfigurationType], message: str | None = None):
        if message is None:
            # message = f"All inverse kinematics solutions are invalid:\n{ik_solutions}"
            message_lines = ["All inverse kinematics solutions are invalid:"]
            for solution in ik_solutions:
                message_lines.append(f"  {solution}")
            message_lines.append("This can happen when your collision checking is not configured correctly.")
            message = "\n".join(message_lines)
        super().__init__(message)
        self.ik_solutions = ik_solutions


class AllGoalConfigurationsRemovedError(PlannerError):
    """Raised when all IK solutions are removed by the goal configuration filter."""

    def __init__(self, goal_configurations: list[JointConfigurationType], message: str | None = None):
        if message is None:
            message_lines = ["Valid goal configurations were found, but all were removed by the filter:"]
            for goal_configuration in goal_configurations:
                message_lines.append(f"  {goal_configuration}")
            message = "\n".join(message_lines)
        super().__init__(message)

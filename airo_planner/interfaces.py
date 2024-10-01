import abc

from airo_typing import HomogeneousMatrixType, JointConfigurationType, JointPathType, Vector3DType


# TODO: plan based on wheel configs as well or instead?
class MobilePlatformPlanner(abc.ABC):
    """Base class that defines an interface for mobile platform motion planners,
    where the platform is represented as an unbounded planar joint.

    The idea is that the custom settings for each motion planner are provided
    through the constructor. After creation all motion planners can then be
    used in the same way, e.g. for benchmarking.
    """

    @abc.abstractmethod
    # TODO: Attitude2DType?
    def plan_to_pose(self, start_pose: Vector3DType, goal_pose: Vector3DType) -> JointPathType:
        """Plan a path from a start pose to a goal pose.

        Args:
            start_pose: The (x, y, theta) start pose.
            goal_pose: The (x, y, theta) goal pose.

        Returns:
            A discretized path from the start pose to the goal pose.

        Raises:
            NoPathFoundError: If no path could be found between the start and
            goal configuration."""


class SingleArmPlanner(abc.ABC):
    """Base class that defines an interface for single-arm motion planners.

    The idea is that the custom settings for each motion planner are provided
    through the constructor. After creation all motion planners can then be
    used in the same way, e.g. for benchmarking.
    """

    def plan_to_joint_configuration(
        self, start_configuration: JointConfigurationType, goal_configuration: JointConfigurationType
    ) -> JointPathType:
        """Plan a path from a start configuration to a goal configuration.

        Note that this path is not guarenteed to be dense, i.e. the distance
        between consecutive configurations in the path may be large. For this
        reason, you might need to post-process these path before executing it.

        Args:
            start_configuration: The start joint configuration.
            goal_configuration: The goal joint configuration.

        Returns:
            A discretized path from the start configuration to the goal
            configuration.

        Raises:
            NoPathFoundError: If no path could be found between the start and
            goal configuration.
        """
        raise NotImplementedError("This planner has not implemented planning to joint configurations (yet).")

    def plan_to_tcp_pose(
        self, start_configuration: JointConfigurationType, tcp_pose: HomogeneousMatrixType
    ) -> JointPathType:
        raise NotImplementedError("This planner has not implemented planning to TCP poses (yet).")


class DualArmPlanner(abc.ABC):
    """Base class that defines an interface for dual-arm motion planners.

    This class follows the pattern we use in DualArmPositionManipulator, where
    we allow the user to use None to signal when one of the arms should not be
    used for a particular motion.
    """

    def plan_to_joint_configuration(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: JointConfigurationType | None,
        goal_configuration_right: JointConfigurationType | None,
    ) -> JointPathType:
        """Plan a path from a start configurations to a goal configurations.

        The start cofinguration of the left and right arm must always be given.
        The goal configuration of at most one of the arms can be None, which
        signals that that arm should remain stationary. If the goal
        configuration is the same as the start configuration, then the planner
        is allowed to more that arm out of the way and move it back. e.g. if
        that makes avoiding collisions easier.

        Args:
            start_configuration_left: The start configuration of the left arm.
            start_configuration_right: The start configuration of the right arm.
            goal_configuration_left: The goal configuration of the left arm.
            goal_configuration_right: The goal configuration of the right arm.

        Returns:
            A discretized path from the start configuration to the goal
            configuration. If the goal_configuration of an arm is None, then
            the start_configuration will simply be repeated in the path for that
            arm. If no solution could be found, then None is returned.

        Raises:
            NoPathFoundError: If no path could be found between the start and
            goal configurations.
        """
        raise NotImplementedError("This planner has not implemented planning to joint configurations (yet).")

    def plan_to_tcp_pose(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        tcp_pose_left: HomogeneousMatrixType | None,
        tcp_pose_right: HomogeneousMatrixType | None,
    ) -> JointPathType:
        raise NotImplementedError("This planner has not implemented planning to TCP poses (yet).")

import abc
from typing import List, Tuple, Union

from airo_typing import HomogeneousMatrixType, JointConfigurationType, JointPathType


class SingleArmPlanner(abc.ABC):
    """Base class that defines an interface for single-arm motion planners.

    The idea is that the custom settings for each motion planner are provided
    through the constructor. After creation the motion planner can then be

      and from then on all motion planners can be used
    in the same way, e.g. for benchmarking.
    """

    @abc.abstractmethod
    def plan_to_joint_configuration(
        self,
        start_configuration: JointConfigurationType,
        goal_configuration: JointConfigurationType,
    ) -> JointPathType | None:
        """Plan a path from a start configuration to a goal configuration.

        Note that this path is not guarenteed to be dense, i.e. the distance
        between consecutive configurations in the path may be large. For this
        reason, you might need to post-process these path before executing it.

        Args:
            start_configuration: The start joint configuration.
            goal_configuration: The goal joint configuration.

        Returns:
            A discretized path from the start configuration to the goal
            configuration. If no solution could be found, then None is returned.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def plan_to_tcp_pose(
        self,
        start_configuration: JointConfigurationType,
        tcp_pose_in_base: HomogeneousMatrixType,
    ) -> JointPathType | None:
        """TODO"""
        raise NotImplementedError


class DualArmPlanner(abc.ABC):
    """Base class that defines an interface for dual-arm motion planners.

    This class follows the pattern we use in DualArmPositionManipulator, where
    we allow the user to use None to signal when one of the arms should not be
    used for a particular motion.
    """

    @abc.abstractmethod
    def plan_to_joint_configuration(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        goal_configuration_left: Union[JointConfigurationType, None],
        goal_configuration_right: Union[JointConfigurationType, None],
    ) -> Union[List[Tuple[JointConfigurationType, JointConfigurationType]], None]:
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
        """
        raise NotImplementedError

    @abc.abstractmethod
    def plan_to_tcp_pose(
        self,
        start_configuration_left: JointConfigurationType,
        start_configuration_right: JointConfigurationType,
        tcp_pose_left_in_base: Union[HomogeneousMatrixType, None],
        tcp_pose_right_in_base: Union[HomogeneousMatrixType, None],
    ) -> Union[List[Tuple[JointConfigurationType, JointConfigurationType]], None]:
        """TODO"""
        raise NotImplementedError

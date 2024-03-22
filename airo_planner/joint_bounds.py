import numpy as np
from airo_typing import JointConfigurationType

# TODO consider moving this to airo_typing
# note: can be used both for single arm and dual arm joint bounds
JointBoundsType = tuple[JointConfigurationType, JointConfigurationType]


def is_within_bounds(joint_configuration: JointConfigurationType, joint_bounds: JointBoundsType) -> bool:
    """Checks if a joint configuration is within the given bounds.

    Args:
        joint_configuration: The joint configuration to check.
        joint_bounds: A tuple containing the lower and upper bounds for each joint.

    Returns:
        True if the joint configuration is within the bounds, False otherwise.
    """
    lower_bounds, upper_bounds = joint_bounds
    is_within = np.all(lower_bounds <= joint_configuration) and np.all(joint_configuration <= upper_bounds)
    return bool(is_within)


def uniform_symmetric_joint_bounds(degrees_of_freedom: int, value: float = 2 * np.pi) -> JointBoundsType:
    """Creates joint bounds that are the same for each joint and where the min is the negative of the max.

    Args:
        degrees_of_freedom: The number of joints for which to create bounds.
        value: The maximum positive (and negative) value for each joint bound.

    Returns:
        A tuple containing:
            * Lower bounds for each joint.
            * Upper bounds for each joint.
    """

    joint_bounds = (
        np.full(degrees_of_freedom, -value),
        np.full(degrees_of_freedom, value),
    )
    return joint_bounds


def concatenate_joint_bounds(joint_bounds: list[JointBoundsType]) -> JointBoundsType:
    """Concatenates a list of joint bounds into a single combined bounds tuple.

    Args:
        joint_bounds: A list of bounds for a set of joints.

    Returns:
        A tuple containing:
            * Concatenated lower bounds.
            * Concatenated upper bounds.
    """
    lower_bounds = np.concatenate([bounds[0] for bounds in joint_bounds])
    upper_bounds = np.concatenate([bounds[1] for bounds in joint_bounds])
    concatenated_joints_bounds = (lower_bounds, upper_bounds)
    return concatenated_joints_bounds

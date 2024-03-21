import numpy as np
from airo_typing import JointConfigurationType, JointPathType


def stack_joints(
    joints_a: JointConfigurationType | JointPathType, joints_b: JointConfigurationType | JointPathType
) -> JointConfigurationType | JointPathType:
    """Stacks joint configurations or joint paths column-wise.

    Valid input shapes:
    - (d,), (1, d) (n, d)

    Possible output shapes:
    - (2 * d,), (1, 2 * d), (n, 2 * d)

    Args:
        joints_a: Joint configurations or a joint path.
        joints_b: Joint configurations or a joint path.

    Returns:
        The stacked configurations.
    """

    joints_a = np.asarray(joints_a)  # Ensure inputs are NumPy arrays
    joints_b = np.asarray(joints_b)

    if joints_a.ndim == 1 and joints_b.ndim == 1:
        return np.concatenate([joints_a, joints_b])

    if len(joints_a.shape) == 1:
        joints_a = joints_a.reshape(1, -1)

    if len(joints_b.shape) == 1:
        joints_b = joints_b.reshape(1, -1)

    if joints_a.shape == joints_b.shape:
        return np.column_stack([joints_a, joints_b])

    if joints_a.shape[0] == 1:
        joints_a = np.repeat(joints_a, joints_b.shape[0], axis=0)

    if joints_b.shape[0] == 1:
        joints_b = np.repeat(joints_b, joints_a.shape[0], axis=0)

    return np.column_stack([joints_a, joints_b])

import numpy as np
from airo_typing import JointConfigurationType


def rank_by_distance_to_desirable_configurations(
    configurations: list[JointConfigurationType], desirable_configurations: list[JointConfigurationType]
) -> list[JointConfigurationType]:
    """Ranks joint configurations based on their distance to a set of desirable configurations.

    Args:
        configurations: The list of joint configurations to be ranked.
        desirable_configurations: A list of desirable joint configurations.

    Returns:
        A list of joint configurations, sorted in ascending order of their
        distance to the closest desirable configuration.
    """

    distances = []
    for config in configurations:
        distances_to_desirable = [
            np.linalg.norm(config - desirable_config) for desirable_config in desirable_configurations
        ]
        min_distance = np.min(distances_to_desirable)
        distances.append(min_distance)

    ranked_configurations = [x for _, x in sorted(zip(distances, configurations))]
    return ranked_configurations


def filter_with_distance_to_configurations(
    joint_configurations: list[JointConfigurationType],
    joint_configurations_close: list[JointConfigurationType],
    distance_threshold: float = 0.01,
) -> list[JointConfigurationType]:
    """Filters a list of joint configurations, keeping only those within a specified distance of a reference set.

    Args:
        joint_configurations: The list of joint configurations to be filtered.
        joint_configurations_close: A list of reference joint configurations.
        distance_threshold: The maximum allowable distance between a configuration
            and a reference configuration for it to be kept.

    Returns:
        A list of filtered joint configurations that are within the distance threshold
        of at least one configuration in the reference set.
    """
    joint_configurations_to_keep = []

    for joint_configuration in joint_configurations:
        for joints_close in joint_configurations_close:
            if np.linalg.norm(joint_configuration - joints_close) < distance_threshold:
                joint_configurations_to_keep.append(joint_configuration)
                break

    return joint_configurations_to_keep

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
        min_distance = min(distances_to_desirable)
        distances.append(min_distance)

    ranked_configurations = [x for _, x in sorted(zip(distances, configurations))]
    return ranked_configurations

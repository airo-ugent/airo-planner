import numpy as np
from airo_typing import JointPathType


def choose_shortest_path(paths: list[JointPathType]) -> JointPathType:
    """Selects the shortest path from a list of possible paths.

    Args:
        paths: A list of potential joint paths.

    Returns:
        The path from the input list that has the shortest length.
    """
    path_lengths = [np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) for p in paths]
    shortest_idx = np.argmin(path_lengths)
    return paths[shortest_idx]

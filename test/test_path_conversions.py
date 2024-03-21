import numpy as np

from airo_planner import stack_joints


def test_stack_joints():
    config_a = np.zeros(6)
    config_b = np.ones(6)

    config_a_1x6 = config_a.reshape(1, -1)
    config_b_1x6 = config_b.reshape(1, -1)

    path_a = np.repeat(config_a_1x6, 10, axis=0)
    path_b = np.repeat(config_b_1x6, path_a.shape[0], axis=0)

    stacked_aa = stack_joints(path_a, path_b)
    assert stacked_aa.shape == (10, 12)

    stacked_ab = stack_joints(path_a, config_b)
    assert stacked_ab.shape == (10, 12)

    stacked_ab = stack_joints(config_a, path_b)
    assert stacked_ab.shape == (10, 12)

    stacked_ab = stack_joints(config_a, config_b)
    assert stacked_ab.shape == (12,)

    stacked_ab = stack_joints(config_a_1x6, config_b_1x6)
    assert stacked_ab.shape == (1, 12)

    stacked_ab = stack_joints(config_a_1x6, path_b)
    assert stacked_ab.shape == (10, 12)

    stacked_ab = stack_joints(path_a, config_b_1x6)
    assert stacked_ab.shape == (10, 12)

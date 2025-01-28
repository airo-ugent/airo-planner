from typing import Any, Callable

import numpy as np
from airo_typing import JointConfigurationCheckerType, JointPathType
from ompl import base as ob
from ompl import geometric as og

from airo_planner import JointBoundsType


def state_from_ompl(state: ob.State, n: int) -> np.ndarray:
    return np.array([state[i] for i in range(n)])


def state_to_ompl(state_numpy: np.ndarray, space: ob.StateSpace) -> ob.State:
    state = ob.State(space)
    for i in range(len(state_numpy)):
        state()[i] = state_numpy[i]
    return state


def path_from_ompl(path: og.PathGeometric, n: int) -> JointPathType:
    return np.array([state_from_ompl(path.getState(i), n) for i in range(path.getStateCount())])


def function_to_ompl(function: Callable[[np.ndarray], Any], n: int) -> Callable[[ob.State], Any]:
    """Take a function that has a single numpy array as input,
    and wrap it so that it takes an OMPL state as input instead."""

    def wrapper(ompl_state: ob.State) -> Any:
        numpy_state = state_from_ompl(ompl_state, n)
        return function(numpy_state)

    return wrapper


def bounds_to_ompl(joint_bounds: JointBoundsType) -> ob.RealVectorBounds:
    degrees_of_freedom = len(joint_bounds[0])
    bounds = ob.RealVectorBounds(degrees_of_freedom)
    joint_bounds_lower = joint_bounds[0]
    joint_bounds_upper = joint_bounds[1]
    for i in range(degrees_of_freedom):
        bounds.setLow(i, joint_bounds_lower[i])
        bounds.setHigh(i, joint_bounds_upper[i])
    return bounds


def create_simple_setup(
    is_state_valid_fn: JointConfigurationCheckerType, joint_bounds: JointBoundsType
) -> og.SimpleSetup:
    """Create a simple setup for a robot with given joints bounds and a state validity checker.
    Can be used for single arm and dual arm robots.

    Args:
        is_state_valid_fn: A function that checks if a joint configuration is valid.
        joint_bounds: The joint bounds.

    Returns:
        A OMPL simple setup object.

    """
    degrees_of_freedom = len(joint_bounds[0])
    space = ob.RealVectorStateSpace(degrees_of_freedom)
    space.setBounds(bounds_to_ompl(joint_bounds))

    is_state_valid_ompl = function_to_ompl(is_state_valid_fn, degrees_of_freedom)

    # Configure the SimpleSetup object
    simple_setup = og.SimpleSetup(space)
    simple_setup.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid_ompl))

    # Most planners don't need this, but we set it just in case you want to use an optimizing planner.
    simple_setup.setOptimizationObjective(ob.PathLengthOptimizationObjective(simple_setup.getSpaceInformation()))

    # TODO: We should investigate tje effect of this setting further
    step = float(np.deg2rad(5))
    resolution = step / space.getMaximumExtent()
    simple_setup.getSpaceInformation().setStateValidityCheckingResolution(resolution)

    # Set default planner, note that you may change this
    planner = og.RRTConnect(simple_setup.getSpaceInformation())
    simple_setup.setPlanner(planner)

    return simple_setup


def solve_and_smooth_path(simple_setup: og.SimpleSetup, allowed_planning_time: float = 1.0) -> JointPathType | None:
    # Should be called after start and goal have been set

    # Run planning algorithm
    simple_setup.solve(allowed_planning_time)

    if not simple_setup.haveExactSolutionPath():
        return None

    # Simplify and smooth the path, note that this conserves path validity
    simple_setup.simplifySolution()
    path_simplifier = og.PathSimplifier(simple_setup.getSpaceInformation())
    path = simple_setup.getSolutionPath()
    path_simplifier.smoothBSpline(path)

    # Extract path
    degrees_of_freedom = simple_setup.getStateSpace().getDimension()
    path_numpy = path_from_ompl(path, degrees_of_freedom)

    return path_numpy

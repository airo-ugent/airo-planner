# init files are still required in modern python!
# https://peps.python.org/pep-0420/ introduced implicit namespace packages
# but for building and many toolings, you still need to have __init__ files (at least in the root of the package).
# e.g. if you remove this init file and try to build with pip install .
# you won't be able to import the dummy module.

# We disable isort for the entire __init__.py file, because the order of imports is important here.
# If a function depends on another function, the dependent function should be imported after the function it depends on.
# isort: skip_file
from typing import Any

from airo_planner.path_conversions import stack_joints
from airo_planner.exceptions import (
    AllGoalConfigurationsRemovedError,
    GoalConfigurationOutOfBoundsError,
    InvalidConfigurationError,
    InvalidGoalConfigurationError,
    InvalidStartConfigurationError,
    JointOutOfBoundsError,
    NoInverseKinematicsSolutionsError,
    NoInverseKinematicsSolutionsWithinBoundsError,
    NoPathFoundError,
    NoValidInverseKinematicsSolutionsError,
    PlannerError,
    StartConfigurationOutOfBoundsError,
)
from airo_planner.joint_bounds import (
    JointBoundsType,
    concatenate_joint_bounds,
    is_within_bounds,
    uniform_symmetric_joint_bounds,
)
from airo_planner.selection.goal_selection import (
    filter_with_distance_to_configurations,
    rank_by_distance_to_desirable_configurations,
)
from airo_planner.selection.path_selection import choose_shortest_path

from airo_planner.multiple_goal_planner import (
    MultipleGoalPlanner,
    JointConfigurationsModifierType,
    JointPathChooserType,
)

from airo_planner.function_conversions import (
    convert_dual_arm_joints_modifier_to_single_arm,
    convert_dual_arm_path_chooser_to_single_arm,
)

from airo_planner.interfaces import DualArmPlanner, SingleArmPlanner, MobilePlatformPlanner

# OMPL is an optional dependency (pip install "airo-planner[ompl]").
# The OMPL names below are imported lazily (PEP 562), so that `import airo_planner` also works without OMPL,
# e.g. when you only use cuRobo.
_OMPL_NAMES = {
    "function_to_ompl": "airo_planner.ompl.utilities",
    "state_to_ompl": "airo_planner.ompl.utilities",
    "path_from_ompl": "airo_planner.ompl.utilities",
    "state_from_ompl": "airo_planner.ompl.utilities",
    "bounds_to_ompl": "airo_planner.ompl.utilities",
    "create_simple_setup": "airo_planner.ompl.utilities",
    "solve_and_smooth_path": "airo_planner.ompl.utilities",
    "SingleArmOmplPlanner": "airo_planner.ompl.single_arm_ompl_planner",
    "DualArmOmplPlanner": "airo_planner.ompl.dual_arm_ompl_planner",
    "MobilePlatformOmplPlanner": "airo_planner.ompl.mobile_platform_ompl_planner",
}


def __getattr__(name: str) -> Any:
    if name in _OMPL_NAMES:
        import importlib

        try:
            module = importlib.import_module(_OMPL_NAMES[name])
        except ImportError as e:
            raise ImportError(
                f"'{name}' requires OMPL, which is not installed. Install it with: pip install 'airo-planner[ompl]'"
            ) from e
        return getattr(module, name)
    raise AttributeError(f"module 'airo_planner' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_OMPL_NAMES))


__all__ = [
    "SingleArmPlanner",
    "DualArmPlanner",
    "MobilePlatformPlanner",
    "MultipleGoalPlanner",
    "JointConfigurationsModifierType",
    "JointPathChooserType",
    "SingleArmOmplPlanner",
    "DualArmOmplPlanner",
    "MobilePlatformOmplPlanner",
    "JointBoundsType",
    "is_within_bounds",
    "uniform_symmetric_joint_bounds",
    "concatenate_joint_bounds",
    "state_from_ompl",
    "state_to_ompl",
    "path_from_ompl",
    "function_to_ompl",
    "bounds_to_ompl",
    "create_simple_setup",
    "solve_and_smooth_path",
    "JointBoundsType",
    "choose_shortest_path",
    "rank_by_distance_to_desirable_configurations",
    "filter_with_distance_to_configurations",
    "stack_joints",
    "convert_dual_arm_joints_modifier_to_single_arm",
    "convert_dual_arm_path_chooser_to_single_arm",
    "PlannerError",
    "NoPathFoundError",
    "InvalidConfigurationError",
    "InvalidStartConfigurationError",
    "InvalidGoalConfigurationError",
    "JointOutOfBoundsError",
    "StartConfigurationOutOfBoundsError",
    "GoalConfigurationOutOfBoundsError",
    "NoInverseKinematicsSolutionsError",
    "NoInverseKinematicsSolutionsWithinBoundsError",
    "NoValidInverseKinematicsSolutionsError",
    "AllGoalConfigurationsRemovedError",
]

# init files are still required in modern python!
# https://peps.python.org/pep-0420/ introduced implicit namespace packages
# but for building and many toolings, you still need to have __init__ files (at least in the root of the package).
# e.g. if you remove this init file and try to build with pip install .
# you won't be able to import the dummy module.

# We disable isort for the entire __init__.py file, because the order of imports is important here.
# If a function depends on another function, the dependent function should be imported after the function it depends on.
# isort: skip_file
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

from airo_planner.ompl.utilities import (
    function_to_ompl,
    state_to_ompl,
    path_from_ompl,
    state_from_ompl,
    bounds_to_ompl,
    create_simple_setup,
    solve_and_smooth_path,
)

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

from airo_planner.ompl.single_arm_ompl_planner import SingleArmOmplPlanner
from airo_planner.ompl.dual_arm_ompl_planner import DualArmOmplPlanner
from airo_planner.ompl.mobile_platform_ompl_planner import MobilePlatformOmplPlanner


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

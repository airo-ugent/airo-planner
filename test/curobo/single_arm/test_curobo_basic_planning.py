import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("cuRobo tests require a CUDA-enabled GPU.", allow_module_level=True)

pytest.importorskip("curobo")

from curobo.scene import Cuboid  # noqa: E402

from airo_planner import NoPathFoundError  # noqa: E402
from airo_planner.curobo.single_arm_curobo_planner import SingleArmCuroboPlanner  # noqa: E402

ROBOT_FILE = "ur10e.yml"
WORLD_FILE = "collision_test.yml"


def test_obstacle_free():
    planner = SingleArmCuroboPlanner(ROBOT_FILE, WORLD_FILE)

    start_configuration = planner.motion_planner.default_joint_state.position.cpu().numpy()
    goal_configuration = start_configuration.copy()
    goal_configuration[0] += 0.5
    goal_configuration[1] += 0.3

    trajectory = planner.plan_to_joint_configuration(start_configuration, goal_configuration)

    assert len(trajectory.path.positions) > 0


def test_blocked_by_obstacle():
    planner = SingleArmCuroboPlanner(ROBOT_FILE, WORLD_FILE, max_attempts=2)

    start_configuration = planner.motion_planner.default_joint_state.position.cpu().numpy()
    goal_configuration = start_configuration.copy()
    goal_configuration[0] += 0.5
    goal_configuration[1] += 0.3
    goal_tcp_pose = planner.forward_kinematics(goal_configuration)

    # Enclose the goal TCP position in a large obstacle, so it cannot be reached collision-free.
    obstacles = planner.get_collider_cuboids()
    obstacles.append(
        Cuboid(
            name="blocker",
            dims=[1.0, 1.0, 1.0],
            pose=[goal_tcp_pose[0, 3], goal_tcp_pose[1, 3], goal_tcp_pose[2, 3], 1, 0, 0, 0],
        )
    )
    planner.set_collider_cuboids(obstacles)

    with pytest.raises(NoPathFoundError):
        planner.plan_to_joint_configuration(start_configuration, goal_configuration)

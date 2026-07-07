"""Benchmark single-arm joint-to-joint motion planning: OMPL+Drake vs cuRobo (UR5e).

This script compares two ``SingleArmPlanner`` implementations on the *same* UR5e robot
and the *same* set of collision-free start/goal joint-configuration pairs:

* **OMPL+Drake** -- :class:`airo_planner.SingleArmOmplPlanner` using a Drake
  ``SceneGraphCollisionChecker`` as the state-validity checker (CPU, sampling-based). This is run
  with two collision models: ``primitive`` (the default ``ur5e.urdf`` sphere/cylinder geometry)
  and ``convex`` (``ur5e_convex_collisions.urdf``, convex .obj meshes on the arm links) -- reported
  as planners ``ompl_drake_primitive`` and ``ompl_drake_convex``.
* **OMPL+Drake+TOPPRA** -- same as above but with an additional TOPPRA time-parameterization step
  applied to the geometric path, reported as ``ompl_drake_primitive_toppra`` and
  ``ompl_drake_convex_toppra``.  The recorded time is the sum of OMPL planning time and TOPPRA
  overhead, giving a fairer comparison with cuRobo which already includes time parameterization.
* **cuRobo** -- :class:`airo_planner.curobo.single_arm_curobo_planner.SingleArmCuroboPlanner`
  (GPU, optimization-based).

Both planners are built from the *same* airo-models UR5e URDF so the kinematics and
collision geometry are identical. The cuRobo robot config (collision spheres +
self-collision matrix) is generated at runtime with cuRobo's ``RobotBuilder`` (fit to the
collision meshes, base link clipped at z=0), mirroring ``notebooks/07_curobo_custom_robot.ipynb``.

Two scenarios are benchmarked:

* ``free``     -- no environment obstacles (self-collision + joint limits only).
* ``obstacle`` -- a floor plus a box obstacle, added identically to both worlds.

Because Drake welds UR arms with a 180 deg base rotation (``X_URBASE_ROSBASE``) while cuRobo
uses the URDF base frame directly, obstacle poses are defined once in the *robot base frame*
and transformed into each planner's world frame so the environments match exactly.

For each (scenario, planner, problem) the script records success, planning wall-time, joint-space
path length and waypoint count, prints a summary table and writes a CSV.

Fairness caveats: the two planners use *different* collision models (Drake checks the URDF's
conservative primitive geometry on the CPU; cuRobo checks fitted spheres on the GPU), so a
configuration that is valid for one may be marginally invalid for the other -- cuRobo may report
"Start or End state in collision" on a few Drake-sampled problems. Timings also compare a GPU
optimization-based planner against a CPU sampling-based one, so absolute times are indicative only.
Because this benchmark only performs joint-to-joint (cspace) planning, cuRobo is run with CUDA-graph
capture enabled for representative steady-state timings; a graph-captured instance is locked to this
single query type and must not be reused for TCP-pose/IK planning.

Run with (cuRobo needs a CUDA GPU; only GPU 0 is torch-compatible on this machine)::

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_ompl_vs_curobo.py
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_ompl_vs_curobo.py --num-problems 50 --planning-time 5.0

Requires cuRobo, torch and airo-drake to be installed.
"""

import argparse
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np
import yaml

# ----------------------------------------------------------------------------------------------
# Shared robot / scenario definition
# ----------------------------------------------------------------------------------------------

ROBOT_NAME = "ur5e"
TOOL_FRAME = "tool0"
DOF = 6

# The two airo-models collision models compared for OMPL+Drake:
#   * "primitive" -- the default ur5e.urdf, whose <collision> geometry is hand-authored spheres
#                    and cylinders (conservative, cheap).
#   * "convex"    -- ur5e_convex_collisions.urdf, whose arm links use convex .obj collision meshes
#                    (tighter, closer to cuRobo's mesh-fitted spheres).
DRAKE_COLLISION_MODELS = {
    "primitive": "ur5e",
    "convex": "ur5e_convex_collisions",
}

# UR joint limits are +/- 2*pi for every joint; used for both planners so the search space matches.
JOINT_BOUNDS = (np.full(DOF, -2.0 * np.pi), np.full(DOF, 2.0 * np.pi))


@dataclass
class BoxObstacle:
    """An axis-aligned box obstacle, defined in the *robot base frame*."""

    name: str
    dims: Tuple[float, float, float]  # x, y, z full extents
    position: Tuple[float, float, float]  # center, in the robot base frame


@dataclass
class Scenario:
    name: str
    obstacles: List[BoxObstacle] = field(default_factory=list)


def build_scenarios() -> List[Scenario]:
    """The benchmark scenarios (obstacles are expressed in the robot base frame)."""
    # A thin "table" whose top surface is at z=0 (matches airo-drake add_floor with thickness 0.2),
    # plus a box sitting in front of the robot that the arm must plan around.
    table = BoxObstacle(name="floor", dims=(2.0, 2.0, 0.2), position=(0.0, 0.0, -0.1))
    box = BoxObstacle(name="box", dims=(0.2, 0.2, 0.6), position=(0.45, 0.0, 0.3))
    return [
        Scenario(name="free", obstacles=[]),
        Scenario(name="obstacle", obstacles=[table, box]),
    ]


# ----------------------------------------------------------------------------------------------
# Base-frame -> world-frame transforms (Drake welds UR arms with a 180 deg z rotation)
# ----------------------------------------------------------------------------------------------


def curobo_cuboid_pose(obstacle: BoxObstacle) -> List[float]:
    """cuRobo uses the URDF base frame directly, so the base-frame pose is used as-is.

    Returns a cuRobo pose ``[x, y, z, qw, qx, qy, qz]`` (identity orientation).
    """
    x, y, z = obstacle.position
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


def drake_obstacle_transform(obstacle: BoxObstacle):  # type: ignore[no-untyped-def]
    """Transform a base-frame box into Drake's world frame via X_URBASE_ROSBASE (180 deg about z)."""
    from airo_drake.building.manipulator import X_URBASE_ROSBASE

    p = np.array(obstacle.position, dtype=float)
    return X_URBASE_ROSBASE.multiply(p)


# ----------------------------------------------------------------------------------------------
# cuRobo setup
# ----------------------------------------------------------------------------------------------


def _first_collision_link_name(urdf_path: str) -> str:
    """Return the name of the first link in the URDF that has ``<collision>`` geometry.

    The airo-models UR URDFs start with an empty ``base_link`` dummy; the base collision geometry
    (and thus the base spheres to clip at z=0) actually lives on the next link, ``base_link_inertia``.
    Selecting the first link with collision geometry makes the z=0 base clip target the right link
    for any robot, matching the ``base_link_inertia`` clip used in notebooks/07_curobo_custom_robot.ipynb.
    """
    from airo_models import urdf as urdf_utils

    links = urdf_utils.read_urdf(urdf_path)["robot"]["link"]
    if isinstance(links, dict):
        links = [links]
    for link in links:
        if "collision" in link:
            return link["@name"]
    raise ValueError(f"No link with collision geometry found in {urdf_path}")


def build_curobo_robot_config(output_path: str) -> str:
    """Generate a cuRobo v2 UR5e robot .yml from the airo-models URDF using RobotBuilder.

    Spheres are fit to the *collision* meshes and the base link is clipped at z=0 so its spheres
    don't dip below the mounting plane. The self-collision ignore matrix is computed as well.
    """
    import airo_models
    from curobo.robot_builder import RobotBuilder

    urdf_path = airo_models.get_urdf_path(ROBOT_NAME)
    asset_path = os.path.dirname(urdf_path)
    base_collision_link = _first_collision_link_name(urdf_path)

    builder = RobotBuilder(str(urdf_path), str(asset_path), tool_frames=[TOOL_FRAME])
    builder.fit_collision_spheres(
        sphere_density=3.0,
        use_collision_mesh=True,
        clip_links={base_collision_link: ("z", 0.0)},
    )
    builder.compute_collision_matrix(num_samples=1000)

    config = builder.build()
    builder.save(config, output_path)

    # Work around the load_collision_spheres/num_envs fields that break re-loading (see notebook 07).
    with open(output_path) as f:
        data = yaml.safe_load(f)
    data["kinematics"].pop("load_collision_spheres", None)
    data["kinematics"].pop("num_envs", None)
    with open(output_path, "w") as f:
        yaml.safe_dump(data, f)

    return output_path


def write_curobo_world_file(scenario: Scenario, output_path: str) -> str:
    """Write a cuRobo scene .yml containing the scenario's obstacles (in the base frame)."""
    cuboids = {obs.name: {"dims": list(obs.dims), "pose": curobo_cuboid_pose(obs)} for obs in scenario.obstacles}
    # cuRobo requires at least an (empty) cuboid mapping to construct a Scene.
    with open(output_path, "w") as f:
        yaml.safe_dump({"cuboid": cuboids}, f)
    return output_path


def make_curobo_planner(
    robot_file: str, scenario: Scenario, world_dir: str, max_attempts: int
):  # type: ignore[no-untyped-def]
    from airo_planner.curobo.single_arm_curobo_planner import SingleArmCuroboPlanner

    world_file = write_curobo_world_file(scenario, os.path.join(world_dir, f"world_{scenario.name}.yml"))
    # This benchmark only calls plan_to_joint_configuration (cspace), so CUDA graphs can be enabled
    # for ~5x faster steady-state planning. The instance must not be used for TCP-pose/IK planning.
    return SingleArmCuroboPlanner(robot_file, world_file, max_attempts=max_attempts, use_cuda_graph=True)


# ----------------------------------------------------------------------------------------------
# OMPL + Drake setup
# ----------------------------------------------------------------------------------------------


def _self_collision_pairs(robot_diagram, arm_index, num_samples: int, seed: int):  # type: ignore[no-untyped-def]
    """Return {(bodyA_index, bodyB_index): collision_fraction} for arm self-collision pairs.

    Samples the neutral config plus `num_samples` random configs and, using the SceneGraph
    penetration query, records how often each pair of arm bodies is in collision.
    """
    from collections import Counter

    plant = robot_diagram.plant()
    scene_graph = robot_diagram.scene_graph()
    diagram_context = robot_diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    scene_graph_context = scene_graph.GetMyContextFromRoot(diagram_context)

    rng = np.random.default_rng(seed)
    low = np.full(DOF, -np.pi) * 0.9
    counts: "Counter[tuple[int, int]]" = Counter()
    configs = [np.zeros(DOF)] + [rng.uniform(low, -low) for _ in range(num_samples)]
    for q in configs:
        plant.SetPositions(plant_context, arm_index, q)
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
        inspector = query_object.inspector()
        pairs_this_config = set()
        for penetration in query_object.ComputePointPairPenetration():
            body_a = plant.GetBodyFromFrameId(inspector.GetFrameId(penetration.id_A))
            body_b = plant.GetBodyFromFrameId(inspector.GetFrameId(penetration.id_B))
            if body_a.model_instance() == arm_index and body_b.model_instance() == arm_index:
                pairs_this_config.add(tuple(sorted((int(body_a.index()), int(body_b.index())))))
        for pair in pairs_this_config:
            counts[pair] += 1
    n = len(configs)
    return {pair: count / n for pair, count in counts.items()}


def filter_persistent_self_collisions(
    collision_checker, robot_diagram, arm_index, threshold: float = 0.99, num_samples: int = 400, seed: int = 0
) -> "list[tuple[str, str, float]]":  # type: ignore[no-untyped-def]
    """Filter self-collision pairs that collide in >= `threshold` fraction of sampled configs.

    The airo-models UR5e uses a *conservative primitive* collision model in which the base
    collision sphere always overlaps the upper-arm cylinder, producing a self-collision at every
    configuration. Filtering such always-colliding pairs is the Drake analog of cuRobo's
    self-collision ignore matrix (``RobotBuilder.compute_collision_matrix``), and is required for
    the OMPL planner to find any valid state. Genuine, configuration-dependent self-collisions
    (which occur in only a few percent of samples) are left intact.
    """
    from pydrake.multibody.tree import BodyIndex

    plant = robot_diagram.plant()
    fractions = _self_collision_pairs(robot_diagram, arm_index, num_samples, seed)
    filtered = []
    for (a, b), fraction in fractions.items():
        if fraction >= threshold:
            collision_checker.SetCollisionFilteredBetween(BodyIndex(a), BodyIndex(b), True)
            filtered.append((plant.get_body(BodyIndex(a)).name(), plant.get_body(BodyIndex(b)).name(), fraction))
    return filtered


def make_drake_collision_checker(scenario: Scenario, urdf_name: str = ROBOT_NAME):  # type: ignore[no-untyped-def]
    """Build a Drake SceneGraphCollisionChecker for the UR5e plus the scenario's obstacles.

    `urdf_name` selects which airo-models collision model to load (see DRAKE_COLLISION_MODELS):
    the primitive ``ur5e`` URDF or the convex-mesh ``ur5e_convex_collisions`` URDF. The arm is
    added *without* a gripper, so the Drake model matches the cuRobo robot config (built from the
    bare UR5e URDF). Like ``airo_drake.add_manipulator``, the UR arm base is welded to the world
    with a 180 deg rotation (``X_URBASE_ROSBASE``); obstacles are placed in that same world frame
    via :func:`drake_obstacle_transform`. Always-colliding self-collision pairs (a
    conservative-collision-model artifact) are filtered, mirroring cuRobo's ignore matrix.

    Returns:
        A ``(collision_checker, robot_diagram)`` tuple.  The ``robot_diagram`` is exposed so that
        callers can retrieve its ``plant`` for TOPPRA time parameterization.
    """
    import airo_models
    from airo_drake import finish_build
    from airo_drake.building.manipulator import X_URBASE_ROSBASE
    from pydrake.math import RigidTransform
    from pydrake.planning import RobotDiagramBuilder, SceneGraphCollisionChecker

    robot_diagram_builder = RobotDiagramBuilder()
    plant = robot_diagram_builder.plant()
    parser = robot_diagram_builder.parser()
    parser.SetAutoRenaming(True)
    world_frame = plant.world_frame()

    # Add the bare UR5e arm and weld its base to the world with the UR base rotation.
    arm_urdf_path = airo_models.get_urdf_path(urdf_name)
    arm_index = parser.AddModels(arm_urdf_path)[0]
    arm_frame = plant.GetFrameByName("base_link", arm_index)
    plant.WeldFrames(world_frame, arm_frame, X_URBASE_ROSBASE)

    for obstacle in scenario.obstacles:
        box_urdf_path = airo_models.box_urdf_path(obstacle.dims, obstacle.name)
        index = parser.AddModels(box_urdf_path)[0]
        frame = plant.GetFrameByName("base_link", index)
        transform = RigidTransform(p=drake_obstacle_transform(obstacle))
        plant.WeldFrames(world_frame, frame, transform)

    robot_diagram, _ = finish_build(robot_diagram_builder)

    collision_checker = SceneGraphCollisionChecker(
        model=robot_diagram,
        robot_model_instances=[arm_index],
        edge_step_size=0.125,
        env_collision_padding=0.005,
        self_collision_padding=0.005,
    )

    filtered = filter_persistent_self_collisions(collision_checker, robot_diagram, arm_index)
    for name_a, name_b, fraction in filtered:
        print(f"    filtered persistent self-collision pair: {name_a} <-> {name_b} ({fraction:.0%} of samples)")

    return collision_checker, robot_diagram


def make_ompl_planner(collision_checker, planning_time: float):  # type: ignore[no-untyped-def]
    from airo_planner import SingleArmOmplPlanner

    return SingleArmOmplPlanner(
        is_state_valid_fn=collision_checker.CheckConfigCollisionFree,
        joint_bounds=JOINT_BOUNDS,
        degrees_of_freedom=DOF,
        allowed_planning_time=planning_time,
    )


def make_toppra_plan_fn(ompl_planner, robot_diagram):  # type: ignore[no-untyped-def]
    """Return a planning function that applies TOPPRA time parameterization after OMPL planning.

    The total elapsed time measured by :func:`run_single` will include both the OMPL planning time
    and the TOPPRA overhead, making the comparison with cuRobo (which includes its own time
    parameterization) more apples-to-apples.  On TOPPRA failure the error is re-raised as
    :class:`~airo_planner.NoPathFoundError` so that :func:`run_single` records it as a failure.

    The geometric path is unchanged by TOPPRA; only the timing differs.  The returned positions
    array is therefore identical to the OMPL output.
    """
    plant = robot_diagram.plant()

    def plan_fn(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        from airo_drake import time_parametrize_toppra
        from airo_drake.exceptions import TimeParameterizationError

        from airo_planner import NoPathFoundError

        path = ompl_planner.plan_to_joint_configuration(start, goal)
        try:
            # The trajectory result is intentionally discarded: we only need to measure the
            # overhead of TOPPRA so that run_single captures planning + parameterization time.
            # The geometric path is unchanged by TOPPRA, so returning `path` is correct.
            time_parametrize_toppra(plant, path)
        except TimeParameterizationError as e:
            raise NoPathFoundError(str(e)) from e
        return path

    return plan_fn


# ----------------------------------------------------------------------------------------------
# Problem suite
# ----------------------------------------------------------------------------------------------


def sample_valid_configuration(is_valid: Callable[[np.ndarray], bool], rng: np.random.Generator) -> np.ndarray:
    """Rejection-sample a collision-free joint configuration within limited bounds.

    We sample from a limited range (not the full +/-2pi) so that start/goal poses are "reasonable"
    manipulation configurations rather than fully wrapped-around joints.
    """
    low = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]) * 0.9
    high = -low
    for _ in range(1000):
        q = rng.uniform(low, high)
        if is_valid(q):
            return q
    raise RuntimeError("Could not sample a valid configuration within 1000 attempts.")


def build_problem_suite(
    is_valid: Callable[[np.ndarray], bool], num_problems: int, seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate `num_problems` (start, goal) pairs that are both collision-free for the given scenario."""
    rng = np.random.default_rng(seed)
    problems = []
    for _ in range(num_problems):
        start = sample_valid_configuration(is_valid, rng)
        goal = sample_valid_configuration(is_valid, rng)
        problems.append((start, goal))
    return problems


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------


@dataclass
class Result:
    scenario: str
    planner: str
    problem_index: int
    success: bool
    planning_time_s: float
    path_length: float
    num_waypoints: int


def _positions_from_result(path_or_trajectory) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Extract an (N, DOF) joint array from either an OMPL JointPathType or a SingleArmTrajectory."""
    positions = getattr(getattr(path_or_trajectory, "path", None), "positions", None)
    if positions is not None:  # SingleArmTrajectory (cuRobo)
        return np.asarray(positions)
    return np.asarray(path_or_trajectory)  # OMPL returns the path array directly


def run_single(
    planner_name: str,
    plan_fn: Callable[[np.ndarray, np.ndarray], object],
    start: np.ndarray,
    goal: np.ndarray,
    scenario_name: str,
    problem_index: int,
) -> Result:
    from airo_drake import calculate_joint_path_length

    from airo_planner import NoPathFoundError

    t0 = time.perf_counter()
    try:
        raw = plan_fn(start, goal)
        elapsed = time.perf_counter() - t0
        positions = _positions_from_result(raw)
        success = positions is not None and len(positions) > 0
        path_length = float(calculate_joint_path_length(positions)) if success else float("nan")
        num_waypoints = int(len(positions)) if success else 0
    except NoPathFoundError:
        elapsed = time.perf_counter() - t0
        success, path_length, num_waypoints = False, float("nan"), 0

    return Result(
        scenario=scenario_name,
        planner=planner_name,
        problem_index=problem_index,
        success=success,
        planning_time_s=elapsed,
        path_length=path_length,
        num_waypoints=num_waypoints,
    )


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------


def summarize(results: List[Result]) -> "list[dict]":
    """Aggregate per (scenario, planner): success rate, mean time (successful), mean path length."""
    keys = sorted({(r.scenario, r.planner) for r in results})
    rows = []
    for scenario, planner in keys:
        subset = [r for r in results if r.scenario == scenario and r.planner == planner]
        successful = [r for r in subset if r.success]
        n = len(subset)
        rows.append(
            {
                "scenario": scenario,
                "planner": planner,
                "n": n,
                "success_rate": len(successful) / n if n else float("nan"),
                "mean_time_s": float(np.mean([r.planning_time_s for r in successful])) if successful else float("nan"),
                "median_time_s": (
                    float(np.median([r.planning_time_s for r in successful])) if successful else float("nan")
                ),
                "mean_path_length": (
                    float(np.mean([r.path_length for r in successful])) if successful else float("nan")
                ),
                "mean_waypoints": (
                    float(np.mean([r.num_waypoints for r in successful])) if successful else float("nan")
                ),
            }
        )
    return rows


def print_summary(rows: "list[dict]") -> None:
    header = [
        "scenario",
        "planner",
        "n",
        "success_rate",
        "mean_time_s",
        "median_time_s",
        "mean_path_length",
        "mean_waypoints",
    ]
    widths = {h: max(len(h), 14) for h in header}
    print("\n" + "=" * 118)
    print("BENCHMARK SUMMARY (OMPL+Drake vs cuRobo, UR5e)")
    print("=" * 118)
    print("  ".join(h.ljust(widths[h]) for h in header))
    print("-" * 118)
    for row in rows:
        cells = []
        for h in header:
            value = row[h]
            if isinstance(value, float):
                cells.append(f"{value:.4f}".ljust(widths[h]))
            else:
                cells.append(str(value).ljust(widths[h]))
        print("  ".join(cells))
    print("=" * 118)


def write_csv(results: List[Result], path: str) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["scenario", "planner", "problem_index", "success", "planning_time_s", "path_length", "num_waypoints"]
        )
        for r in results:
            writer.writerow(
                [r.scenario, r.planner, r.problem_index, r.success, r.planning_time_s, r.path_length, r.num_waypoints]
            )


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-problems", type=int, default=20, help="Number of start/goal pairs per scenario.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the problem suite.")
    parser.add_argument("--planning-time", type=float, default=5.0, help="OMPL allowed planning time (s).")
    parser.add_argument("--curobo-max-attempts", type=int, default=5, help="cuRobo max IK/trajopt attempts.")
    parser.add_argument("--csv", default="benchmark_results.csv", help="Output CSV path.")
    parser.add_argument(
        "--robot-config",
        default=None,
        help="Reuse an existing cuRobo UR5e .yml instead of rebuilding it (speeds up repeated runs).",
    )
    args = parser.parse_args()

    work_dir = tempfile.mkdtemp(prefix="curobo_benchmark_")

    print(f"Building / loading cuRobo UR5e config (work dir: {work_dir}) ...")
    robot_file = args.robot_config or build_curobo_robot_config(os.path.join(work_dir, "ur5e_curobo.yml"))
    print(f"cuRobo robot config: {robot_file}")

    all_results: List[Result] = []

    for scenario in build_scenarios():
        print(f"\n### Scenario: {scenario.name} ({len(scenario.obstacles)} obstacles) ###")

        # Build a Drake collision checker per OMPL collision model (primitive + convex meshes).
        drake_checkers = {}
        drake_diagrams = {}
        for model_label, urdf_name in DRAKE_COLLISION_MODELS.items():
            print(f"  Building Drake collision checker ({model_label}: {urdf_name}) ...")
            checker, diagram = make_drake_collision_checker(scenario, urdf_name)
            drake_checkers[model_label] = checker
            drake_diagrams[model_label] = diagram

        # Sample a single shared problem suite that is valid under BOTH Drake models, so the
        # primitive and convex OMPL runs (and, as far as possible, cuRobo) solve identical problems.
        def is_valid_under_all(q: np.ndarray) -> bool:
            return all(checker.CheckConfigCollisionFree(q) for checker in drake_checkers.values())

        print(f"  Sampling {args.num_problems} valid (start, goal) pairs (valid under all models) ...")
        problems = build_problem_suite(is_valid_under_all, args.num_problems, args.seed)

        # --- OMPL + Drake, once per collision model ---
        for model_label, checker in drake_checkers.items():
            planner_name = f"ompl_drake_{model_label}"
            print(f"  Planning with {planner_name} ...")
            ompl_planner = make_ompl_planner(checker, args.planning_time)
            for i, (start, goal) in enumerate(problems):
                all_results.append(
                    run_single(planner_name, ompl_planner.plan_to_joint_configuration, start, goal, scenario.name, i)
                )

            # --- OMPL + Drake + TOPPRA (same geometric path, with time parameterization overhead) ---
            toppra_planner_name = f"ompl_drake_{model_label}_toppra"
            print(f"  Planning with {toppra_planner_name} ...")
            toppra_plan_fn = make_toppra_plan_fn(ompl_planner, drake_diagrams[model_label])
            for i, (start, goal) in enumerate(problems):
                all_results.append(
                    run_single(toppra_planner_name, toppra_plan_fn, start, goal, scenario.name, i)
                )

        # --- cuRobo ---
        print("  Planning with cuRobo ...")
        curobo_planner = make_curobo_planner(robot_file, scenario, work_dir, args.curobo_max_attempts)
        # Warm-up plan (excluded from results) so JIT/graph compilation doesn't skew the first timing.
        try:
            curobo_planner.plan_to_joint_configuration(problems[0][0], problems[0][1])
        except Exception:  # noqa: BLE001 - warmup failures are non-fatal
            pass
        for i, (start, goal) in enumerate(problems):
            all_results.append(
                run_single("curobo", curobo_planner.plan_to_joint_configuration, start, goal, scenario.name, i)
            )

    rows = summarize(all_results)
    print_summary(rows)
    write_csv(all_results, args.csv)
    print(f"\nWrote per-problem results to {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()

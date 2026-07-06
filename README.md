# airo-planner
Python package for single and dual robot arm motion planning.

**Key motivation:**
  - 🔗 **Provide unified interfaces** for different motion planners and collision checkers, such as [**OMPL**](https://ompl.kavrakilab.org/)'s powerful (but robot-agnostic) sampling-based planners and [**Drake**](https://drake.mit.edu/)'s collision checking for robots.
  - 🦾 **Standardize** and add other features tailored to robotic arm motion planning such as joint limits and planning to TCP poses.

## Overview 🧾

**Features:** this packages provides two main things:
  - 🤝 **Interfaces:** specify interfaces for robot arm motion planning
    - `SingleArmPlanner`
    - `DualArmPlanner`
  - 🔌 **Implementations:** reliable and well-tested implementations of these interfaces.
    - OMPL for single and dual arm planning to joint configurations or TCP poses
    - cuRobo for single arm planning to joint configurations or TCP poses

**Design goals:**
  - ⚓ **Robustness and stability:** provide an *off-the-shelf* motion planner that supports research by reliably covering most (not *all*) use cases at our labs, prioritizing dependability over niche, cutting-edge features.
  - 🧩 **Modularity and flexibility** in the core components:
    - 🧭 Motion planning algorithms
    - 💥 Collision checker
    - 🔙 Inverse kinematics
  - 🐛 **Debuggability and transparency**: many things can go wrong in motion planning, so we log generously and store debugging information (IK solutions, alternative paths) to troubleshoot issues.

  - 🧪 **Enable experimentation:** Facilitate the benchmarking and exploration of experimental planning algorithms.

**Planned features:**
  - Drake optimization-based planning


## Getting started 🚀
See the getting started [notebooks](notebooks), where we set up:
* 🎲 [OMPL](https://ompl.kavrakilab.org/) for sampling-based motion planning
* 🐉 [Drake](https://drake.mit.edu/) for collision checking
* 🧮 [ur-analytic-ik](https://github.com/Victorlouisdg/ur-analytic-ik) for inverse kinematics of a UR5e
* 🟢 [cuRobo](https://github.com/NVlabs/curobo) for GPU-accelerated motion planning

### Which planner should I use?
If you have mostly static scenes, use OMPL. It’s well tested, fast, and runs on your CPU. If you have dynamic scenes that change often and have access to a CUDA-supporting GPU, use cuRobo.

## Installation 🔧

### Installing `airo-planner`
`airo-planner` is available on PyPI and installable with pip:
```
pip install airo-planner
```

### Dependencies
If you want to use cuRobo with `airo-planner`, you first need to install it yourself; `airo-planner` does not depend on it directly since cuRobo isn't published on PyPI. Note that you will need a CUDA-enabled GPU (newer than Turing, driver ≥580.65.06).

```
git clone https://github.com/NVlabs/curobo.git
cd curobo && git checkout v0.8.0
uv venv --python 3.11 && uv pip install ".[cu12-torch]"  # or .[cu13-torch], matching your CUDA driver
```

See the [official install instructions](https://nvlabs.github.io/curobo/latest/) for details.

### Custom robots with cuRobo
If your robot isn't one of cuRobo's bundled configs (`franka.yml`, `ur10e.yml`, ...), cuRobo ships a first-party `RobotBuilder` (`curobo.robot_builder`) to build a robot config from a URDF, including automatic collision-sphere fitting — no Isaac Sim required:
```python
from curobo.robot_builder import RobotBuilder

builder = RobotBuilder("robot.urdf", "assets/")
builder.fit_collision_spheres()
builder.compute_collision_matrix()
config = builder.build()
builder.save(config, "my_robot.yml")
```

See [`notebooks/07_curobo_custom_robot.ipynb`](notebooks/07_curobo_custom_robot.ipynb) for the full walkthrough — it covers two real gotchas the snippet above hides (a `RobotBuilder.save()` bug that breaks reloading the saved file, and why self-collision passing doesn't mean your fitted spheres are collision-free against your actual scene), verified end-to-end on real hardware.

## Developer guide 🛠️
See the [`airo-mono`](https://github.com/airo-ugent/airo-mono) developer guide.
A very similar process and tools are used for this package.

### Releasing 🏷️
See [`airo-models`](https://github.com/airo-ugent/airo-models/tree/main), releasing `airo-planner` works the same way.

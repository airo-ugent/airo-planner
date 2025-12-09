# airo-planner
Python package for single and dual robot arm motion planning.

**Key motivation:**
  - 🔗 **Provide unified interfaces** for different motion planners and collision checkers, such as [**OMPL**](https://ompl.kavrakilab.org/)'s powerful (but robot-agnostic) sampling-based planners and [**Drake**](https://drake.mit.edu/)'s collision checking for robots.
  - 🦾 **Standardize** and add other features taylored to robotic arm motion planning such as joint limits and planning to TCP poses.

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
* 🟢 [cuRobo](https://curobo.org/) for GPU-accelerated motion planning


### Which planner should I use?
If you have mostly static scenes, use OMPL. It’s well tested, fast, and runs on your CPU. If you have dynamic scenes that change often and have access to a CUDA-supporting GPU, use cuRobo.


## Installation 🔧
### Dependencies
If you want to use cuRobo with `airo-planner`, you first need to install it according to [these instructions](https://curobo.org/get_started/1_install_instructions.html). Note that you will need a CUDA-enabled GPU.

### Installing `airo-planner`
`airo-planner` is available on PyPI and installable with pip:
```
pip install airo-planner
```

For visualization of cuRobo worlds (see `notebooks/06_curobo.ipynb`), you can use the optional `rerun` dependency.
```
pip install airo-planner[rerun]
```

### Custom robots with cuRobo
You can use the [official cuRobo instructions](https://curobo.org/tutorials/1_robot_configuration.html#tutorial-with-a-ur5e-robot) to configure a new robot, but this requires Isaac Sim. An easier method is to fill in a YAML file with the output of `bubblify`.

```
pip install bubblify
bubblify --urdf_path /path/to/urdf --show_collision
```

Copy the `collision_spheres` section to the YAML file of your robot, and you are ready to use your own robot!

## Developer guide 🛠️
See the [`airo-mono`](https://github.com/airo-ugent/airo-mono) developer guide.
A very similar process and tools are used for this package.

### Releasing 🏷️
See [`airo-models`](https://github.com/airo-ugent/airo-models/tree/main), releasing `airo-planner` works the same way.

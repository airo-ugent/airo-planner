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

## Installation 🔧

### Installing `airo-planner`
`airo-planner` is available on PyPI and installable with pip:
```
pip install airo-planner
```

## Developer guide 🛠️
See the [`airo-mono`](https://github.com/airo-ugent/airo-mono) developer guide.
A very similar process and tools are used for this package.

### Releasing 🏷️
See [`airo-models`](https://github.com/airo-ugent/airo-models/tree/main), releasing `airo-planner` works the same way.

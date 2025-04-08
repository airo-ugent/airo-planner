# airo-planner
Python package for single and dual robot arm motion planning.

**Key motivation:**
  - 🔗 **Bridge** the gap between [**OMPL**](https://ompl.kavrakilab.org/)'s powerful (but robot-agnostic) sampling-based planners and [**Drake**](https://drake.mit.edu/)'s collision checking for robots.
  - 🦾 **Standardize** and add other features taylored to robotic arm motion planning such as joint limits and planning to TCP poses.

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


🗓️ **Planned features:**
  - 🎯 Drake optimization-based planning


## Getting started 🚀
Complete the [Installation 🔧](#installation-🔧) and then see the getting started [notebooks](notebooks), where we set up:
* 🎲 [OMPL](https://ompl.kavrakilab.org/) for sampling-based motion planning
* 🐉 [Drake](https://drake.mit.edu/) for collision checking
* 🧮 [ur-analytic-ik](https://github.com/Victorlouisdg/ur-analytic-ik) for inverse kinematics of a UR5e


## Installation 🔧
`airo-planner` is available on PyPI and installable with pip:
```
pip install airo-planner
```

**🚧 Important post-installation step**

We depend on `ompl` with its Python bindings, which are not available on PyPI yet. The easiest way to install this for now is to use a pre-release wheel fom their [Github](https://github.com/ompl/ompl/releases):
```
wget https://github.com/ompl/ompl/releases/download/1.7.0/wheels-ubuntu-latest-x86_64.zip
unzip wheels-ubuntu-latest-x86_64.zip && rm wheels-ubuntu-latest-x86_64.zip
pip install https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.7.0-cp310-cp310-manylinux_2_28_x86_64.whl
```

(Replace `cp310` with your Python version, e.g. `cp311` for Python 3.11)

## Developer guide 🛠️
See the [`airo-mono`](https://github.com/airo-ugent/airo-mono) developer guide.
A very similar process and tools are used for this package.

### Releasing 🏷️
See [`airo-models`](https://github.com/airo-ugent/airo-models/tree/main), releasing `airo-planner` works the same way.

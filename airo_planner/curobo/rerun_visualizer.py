"""This file allows you to visualize cuRobo's state with rerun. It requires that rerun is installed
and running."""

import re
from typing import List

import airo_models
import numpy as np
import torch
from airo_typing import JointConfigurationType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import Cuboid, Sphere, WorldConfig
from loguru import logger
from yourdfpy import URDF

try:
    import rerun as rr
    from rerun import RotationAxisAngle
except ImportError:
    logger.error("Please install rerun-sdk.")


class CuRoboRerunVisualizer:
    def __init__(self, rr_application_name: str | None, robot_name: str):
        rr.init(rr_application_name if rr_application_name is not None else "curobo", spawn=False)
        rr.connect_grpc()

        self.robot_name = robot_name

        # Log the geometry of the robot.
        self._robot_urdf = URDF.load(airo_models.get_urdf_path(robot_name))
        pattern = re.compile(rb"<collision\b.*?</collision>", re.DOTALL)
        xml_bytes = pattern.sub(b"", self._robot_urdf.write_xml_string())
        rr.log_file_from_contents(
            f"{self._robot_urdf.robot.name}.urdf",
            xml_bytes,
            entity_path_prefix=self._robot_urdf.robot.name,
            static=True,
        )

        # Obtain joint names and paths from URDF.
        rev = {n: j for n, j in self._robot_urdf.joint_map.items() if j.type == "revolute"}
        cache = {}
        child_to_joint = {j.child: j for j in rev.values()}

        def build(child: str) -> str:
            if child in cache:
                return cache[child]
            j = child_to_joint.get(child)
            path = child if j is None else f"{build(j.parent)}/{j.name}/{child}"
            cache[child] = path
            return path

        self._robot_joint_paths = {name: build(j.child) for name, j in rev.items()}

    def log_curobo_state(self, world_model: WorldConfig, clear: bool = True):
        self._log_curobo_spheres(world_model.sphere, clear)
        self._log_curobo_cuboids(world_model.cuboid, clear)

    def _log_curobo_spheres(self, spheres: List[Sphere], clear: bool):
        if clear:
            rr.log("/world/spheres", rr.Clear(recursive=True))

        centers = [sphere.pose[:3] for sphere in spheres]
        half_sizes = [[sphere.radius, sphere.radius, sphere.radius] for sphere in spheres]

        rr.log(
            "/world/spheres",
            rr.Ellipsoids3D(centers=centers, half_sizes=half_sizes, fill_mode=rr.components.FillMode.Solid),
        )

    def _log_curobo_cuboids(self, cuboids: List[Cuboid], clear: bool):
        if clear:
            rr.log("/world/cuboids", rr.Clear(recursive=True))

        centers = [cuboid.pose[:3] for cuboid in cuboids]
        half_sizes = [0.5 * np.asarray(cuboid.dims) for cuboid in cuboids]
        quaternions = [
            np.asarray(cuboid.pose)[[4, 5, 6, 3]] for cuboid in cuboids
        ]  # rerun expects qx qy qz qw, curobo has qw qx qy qz

        rr.log(
            "/world/cuboids",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                quaternions=quaternions,
                fill_mode=rr.components.FillMode.Solid,
                colors=[(100, 100, 100)] * len(centers),
            ),
        )

    def log_robot_configuration(self, q: JointConfigurationType):
        for j, joint_name in enumerate(self._robot_urdf.actuated_joint_names):
            joint_angle = q[j]
            joint_axis = self._robot_urdf.joint_map[joint_name].axis
            rr.log(
                f"/{self.robot_name}/{self.robot_name}/base_link/base_link-base_link_inertia/"
                + self._robot_joint_paths[joint_name],
                rr.Transform3D(rotation=RotationAxisAngle(axis=joint_axis, angle=joint_angle)),
            )

    def log_robot_collision_spheres(self, kinematics: CudaRobotModel, q: JointConfigurationType):
        spheres = kinematics.get_robot_as_spheres(torch.tensor(q, dtype=torch.float32).cuda())[0]

        centers = [sphere.pose[:3] for sphere in spheres]
        half_sizes = [[sphere.radius, sphere.radius, sphere.radius] for sphere in spheres]

        rr.log(
            f"/{self.robot_name}/collision_spheres",
            rr.Ellipsoids3D(
                centers=centers,
                half_sizes=half_sizes,
                fill_mode=rr.components.FillMode.MajorWireframe,
                colors=[(255, 0, 0)] * len(spheres),
            ),
        )

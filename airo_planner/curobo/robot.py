from os import PathLike

import numpy as np
import torch
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from scipy.spatial.transform import Rotation as R


class RobotRepresentation:
    def __init__(self, robot_config_path: PathLike, urdf_path: PathLike):
        self._tensor_args = TensorDeviceType()

        config = load_yaml(robot_config_path)
        base_link = config["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config["robot_cfg"]["kinematics"]["ee_link"]

        robot_config = RobotConfig.from_basic(urdf_path, base_link, ee_link, self._tensor_args)

        self.kinematic_model = CudaRobotModel(robot_config.kinematics)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_config,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self._tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(ik_config)

    def forward_kinematics(self, q: JointConfigurationType) -> HomogeneousMatrixType:
        crms = self.kinematic_model.get_state(torch.tensor(q, dtype=torch.float32).cuda())
        tcp_pose = np.eye(4)
        tcp_pose[:3, 3] = crms.ee_position.cpu().numpy()
        tcp_pose[:3, :3] = R.from_quat(crms.ee_quaternion.cpu().numpy(), scalar_first=True).as_matrix()
        return tcp_pose

    def inverse_kinematics(self, X_B_Tcp: HomogeneousMatrixType) -> JointConfigurationType:
        goal = Pose.from_matrix(X_B_Tcp)
        result = self.ik_solver.solve_batch(goal)
        print(result.solution)
        print(result.success)
        q_solution = result.solution[result.success]
        return q_solution

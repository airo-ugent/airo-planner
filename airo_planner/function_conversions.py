from airo_typing import JointConfigurationType, JointPathType

from airo_planner import JointConfigurationsModifierType, JointPathChooserType, stack_joints


def convert_dual_arm_joints_modifier_to_single_arm(
    dual_arm_fn: JointConfigurationsModifierType,
    frozen_joint_congiuration: JointConfigurationType,
    degrees_of_freedom_left: int,
    to_left: bool = True,
) -> JointConfigurationsModifierType:
    def single_arm_fn(singe_arm_joint_configurations: list[JointConfigurationType]) -> list[JointConfigurationType]:
        # Create a list of stacked dual arm joint configurations
        dual_arm_joint_configurations = []
        for joint_configuration in singe_arm_joint_configurations:
            if to_left:
                dual_arm_joint_configuration = stack_joints(joint_configuration, frozen_joint_congiuration)
            else:
                dual_arm_joint_configuration = stack_joints(frozen_joint_congiuration, joint_configuration)
            dual_arm_joint_configurations.append(dual_arm_joint_configuration)

        # Apply the dual arm function
        result = dual_arm_fn(dual_arm_joint_configurations)

        # Unstack the joint configurations
        single_arm_joint_configurations = []
        for joint_configuration in result:
            if to_left:
                single_arm_joint_configuration = joint_configuration[:degrees_of_freedom_left]
            else:
                single_arm_joint_configuration = joint_configuration[degrees_of_freedom_left:]
            single_arm_joint_configurations.append(single_arm_joint_configuration)
        return single_arm_joint_configurations

    return single_arm_fn


def convert_dual_arm_path_chooser_to_single_arm(
    dual_arm_fn: JointPathChooserType,
    frozen_joint_congiuration: JointConfigurationType,
    degrees_of_freedom_left: int,
    to_left: bool = True,
) -> JointPathChooserType:
    def single_arm_fn(single_arm_paths: list[JointPathType]) -> JointPathType:
        # Create a list of dual arm joint configurations
        dual_arm_paths = []
        for single_arm_path in single_arm_paths:
            if to_left:
                dual_arm_path = stack_joints(single_arm_path, frozen_joint_congiuration)
            else:
                dual_arm_path = stack_joints(frozen_joint_congiuration, single_arm_path)
            dual_arm_paths.append(dual_arm_path)

        dual_arm_path_chosen = dual_arm_fn(dual_arm_paths)

        if to_left:
            return dual_arm_path_chosen[:, :degrees_of_freedom_left]
        else:
            return dual_arm_path_chosen[:, degrees_of_freedom_left:]

    return single_arm_fn

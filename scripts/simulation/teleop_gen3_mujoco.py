import os

import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    xml_path: str = os.path.join(ASSET_PATH, "arx/Gen/scene_gen3.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "arx/Gen/GEN3-6DOF.urdf"),
    scale_factor: float = 1.5,#不清楚
    visualize_placo: bool = True,
):
    """
    Main function to run the dual UR5e teleoperation in MuJoCo.
    """
    config = {
        "gen3": {
            "link_name": "bracelet_link",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "gen3_target",
        },
    }

    # Create and initialize the teleoperation controller
    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
    )
    #不知道干啥
    # additional constraints hardcoded here for now
    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 1e-4)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)

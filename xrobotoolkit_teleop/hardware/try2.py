from xrobotoolkit_teleop.hardware.gen3_controller import KortexRobotController
robot=KortexRobotController()
while True:
    print(f"robot_joint_positions:{robot.get_joint_positions()}")
    print(f"robot_gripper_position:{robot.get_gripper_position()}")
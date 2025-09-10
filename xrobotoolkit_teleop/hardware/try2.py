from xrobotoolkit_teleop.hardware.gen3_controller import KortexRobotController
robot=KortexRobotController()
robot.home_gripper()
print(f"robot_open_pos:{robot.get_gripper_open_pos()}")
print(f"robot_close_pos:{robot.get_gripper_close_pos()}")
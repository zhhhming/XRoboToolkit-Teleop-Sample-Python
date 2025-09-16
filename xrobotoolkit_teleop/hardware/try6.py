from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController
import time
import numpy as np
robot = KortexRobotController()
while True:
    start_time = time.time()
    speed=np.array([0]*7,dtype=float)
    speed[2]=10

    robot.send_joint_speeds_position_based(speed)
    elapsed_time = time.time() - start_time
    print(f"elapsed_time:{elapsed_time}")
    print(f"velocity:{robot.get_joint_speeds()}")
    sleep_time = (1.0 / 500) - elapsed_time
    if sleep_time > 0:
        time.sleep(sleep_time)
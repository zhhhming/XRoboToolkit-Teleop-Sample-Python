import time
import webbrowser
import placo
import numpy as np
from placo.utils.visualization import(
    frame_viz,
    robot_frame_viz,
    robot_viz,
)
from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController


class JointMatch:
    def __init__(
        self,
        urdf_path,
        ):
        self.controller=KortexRobotController()
        self.placo_robot=placo.RobotWrapper(urdf_path)
        self.placo_viz =robot_viz(self.placo_robot)
        time.sleep(0.5)
        meshcat_url=self.placo_viz.url()
        webbrowser.open(meshcat_url)
        self.joint_order=np.zeros(self.placo_robot.state.q)
        self.placo_viz.display(self.placo_robot.state.q)
        print("placo .........")
        self.controller.home_robot()
        self.controller.home_gripper()
        print(f"home gripper, gripper_open_pos:{self.controller.get_gripper_open_pos},gripper_close_pos:{self.controller.get_gripper_close_pos}")
    def test_joint_order(self, position):
        test_angular_deg=20
        test_angular_rad=np.deg2rad(test_angular_deg)
        qpos=self.placo_robot.state.q
        qpos[position]+=test_angular_rad
        self.placo_robot.state.q=qpos
        self.placo_robot.update_kinematics()
        self.placo_viz.display(self.placo_robot.state.q)
        joint_pos=self.controller.get_joint_positions()
        joint_pos[position]+=test_angular_deg
        self.controller.set_joint_positions(joint_pos)
    def robot2placo(self,placo_pos,robo_pos):
        self.joint_order[placo_pos]=robo_pos
    def get_nq(self):
        return len(self.placo_robot.state.q)
    def try_order(self):
        joint_poss=np.array([0.2]*self.get_nq())
        robo_poss=np.zeros(self.get_nq())
        robo_poss[self.joint_order]=joint_poss
        robo_poss=np.rad2deg(robo_poss)
        self.placo_robot.state.q=joint_poss
        self.placo_robot.update_kinematics()
        self.placo_viz.display(self.placo_robot.state.q)
        self.controller.set_joint_positions(robo_poss)

if __name__ == "__main__":
    urdf_path=""
    match=JointMatch(urdf_path)
    for i in range(match.get_nq()):
        match.test_joint_order(i)
    match.robot2placo(1,1)
    match.try_order()





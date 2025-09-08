from typing import Any, Dict

import mujoco
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
from xrobotoolkit_teleop.utils.mujoco_utils import (
    calc_mujoco_ctrl_from_qpos,
    calc_mujoco_qpos_from_placo_q,
    calc_placo_q_from_mujoco_qpos,
    set_mujoco_joint_pos_by_name,
)


class MujocoTeleopController(BaseTeleopController):
    def __init__(
        self,
        xml_path: str,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        visualize_placo=False,
        scale_factor=1.0,
        dt=0.01,
        mj_qpos_init=None,
    ):
        self.visualize_placo = visualize_placo
        self.xml_path = xml_path
        self.mj_qpos_init = mj_qpos_init

        # To be initialized later
        self.mj_model = None
        self.mj_data = None
        self.target_mocap_idx = {name: -1 for name in manipulator_config.keys()}

        super().__init__(
            robot_urdf_path,
            manipulator_config,
            floating_base,
            R_headset_world,
            scale_factor,
            q_init=None,
            dt=dt,
        )

        if visualize_placo:
            self._init_placo_viz()

    def _robot_setup(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        print("Joint names in the Mujoco model:")
        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  {joint_name}")

        # Configure scene lighting
        self.mj_model.vis.headlight.ambient = [0.4, 0.4, 0.4]
        self.mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.mj_model.vis.headlight.specular = [0.6, 0.6, 0.6]

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.mj_qpos_init is None:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key("home").id)
        else:
            self.mj_data.qpos[:] = self.mj_qpos_init
            self.mj_data.ctrl[:] = calc_mujoco_ctrl_from_qpos(self.mj_model, self.mj_qpos_init)
        #设定机械臂的初始姿态
        mujoco.mj_forward(self.mj_model, self.mj_data)#更新

        # setup mocap target
        for name, config in self.manipulator_config.items():
            if "vis_target" not in config:
                print(f"Warning: 'vis_target' not found in config for {name}. Skipping mocap setup.")
                continue
            vis_target = config["vis_target"]
            mocap_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, vis_target)#由名字找id，每个link都有id
            if mocap_id == -1:
                raise ValueError(f"Mocap body '{vis_target}' not found in the model.")

            if self.mj_model.body_mocapid[mocap_id] == -1:
                raise ValueError(f"Body '{self.vis_target}' is not configured for mocap.")
            else:
                self.target_mocap_idx[name] = self.mj_model.body_mocapid[mocap_id]#返回的是mocap体的索引，mocap非实体，仅用于标明目标位置，应该就类似于一个小的坐标系

            print(f"Mocap ID for '{vis_target}' body: {self.target_mocap_idx[name]}")

    def _send_command(self):#依据placo更新的state q得到仿真机械臂的关节目标位置，再转为执行器的目标
        
        qpos_desired = calc_mujoco_qpos_from_placo_q(
            self.mj_model,
            self.placo_robot,
            self.placo_robot.state.q,
            floating_base=self.floating_base,
        )#从placo获取姿态复制给mujoco，是把placo里同名joint值复制给mujoco同名joint，所以mujoco joint可以多于placo

        for gripper_name, gripper_target in self.gripper_pos_target.items():
            for joint_name, joint_pos in gripper_target.items():
                success = set_mujoco_joint_pos_by_name(
                    self.mj_model,
                    qpos_desired,
                    joint_name,
                    joint_pos,
                )
                if not success:
                    raise ValueError(f"Joint '{gripper_name}' not found in MuJoCo model.")

        self.mj_data.ctrl = calc_mujoco_ctrl_from_qpos(self.mj_model, qpos_desired)#位置伺服的，应该就复制粘贴下就好
        print(f"qpos_desired:{qpos_desired}")
        print(f"real_qpos111:{self.mj_data.qpos}")
        if self.visualize_placo:
            self._update_placo_viz()
#将mujoco机械臂状态复制给placo model
    def _update_robot_state(self):
        mj_qpos = self.mj_data.qpos.copy()
        self.placo_robot.state.q = calc_placo_q_from_mujoco_qpos(
            self.mj_model,
            self.placo_robot,
            mj_qpos,
            floating_base=self.floating_base,
        )
        
        self.placo_robot.update_kinematics()
#更新用于显示目标位置的小坐标系的位置
    def _update_mocap_target(self):
        for name, task in self.effector_task.items():
            T_world_target = task.T_world_frame
            mocap_idx = self.target_mocap_idx.get(name)
            if mocap_idx is not None and mocap_idx != -1:
                self.mj_data.mocap_pos[mocap_idx] = T_world_target[:3, 3]
                self.mj_data.mocap_quat[mocap_idx] = tf.quaternion_from_matrix(T_world_target)
#用于获取仿真机械臂末端执行器位置
    def _get_link_pose(self, ee_name):
        """Get the end effector position and orientation."""
        ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if ee_id == -1:
            raise ValueError(f"End effector body '{ee_name}' not found in the model.")

        ee_xyz = self.mj_data.xpos[ee_id].copy()
        ee_quat = self.mj_data.xquat[ee_id].copy()

        return ee_xyz, ee_quat

    def run(self):
        with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            # Set up viewer camera
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -50
            viewer.cam.distance = 2.0
            viewer.cam.lookat = [0.2, 0, 0]

            while not self._stop_event.is_set():
                try:
                    self._update_robot_state()#调整placo与mujoco模型一致
                    self._update_ik()#获取控制器运动信息，控制placo机械臂移动至对应位置，获取到目标关节信息
                    self._update_gripper_target()#获取夹爪的目标位置
                    self._update_mocap_target()#更新mujoco中用于显示手持控制器的小坐标系的位置，即机械臂目标的位置
                    self._send_command()#位置伺服，依据stateq获得各个执行器的运动指令

                    # Step simulation and update viewer
                    mujoco.mj_step(self.mj_model, self.mj_data)#mujoco按指令运动
                    viewer.sync()
                except KeyboardInterrupt:
                    print("\nTeleoperation stopped.")
                    self._stop_event.set()

import threading
import time
from abc import ABC, abstractmethod
from typing import Dict

import cv2
import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController
from xrobotoolkit_teleop.hardware.interface.base_camera import BaseCameraInterface
from xrobotoolkit_teleop.utils.geometry import apply_delta_pose
from xrobotoolkit_teleop.utils.parallel_gripper_utils import calc_parallel_gripper_position


class HardwareTeleopController(BaseTeleopController, ABC):
    """
    An abstract base class for hardware teleoperation controllers that consolidates
    common logic for threading, logging, and visualization.
    """

    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: dict,
        R_headset_world: np.ndarray,
        floating_base: bool,
        scale_factor: float,
        visualize_placo: bool,
        control_rate_hz: int,
        enable_log_data: bool,
        log_dir: str,
        log_freq: float,
        **kwargs,
    ):
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            floating_base=floating_base,
            R_headset_world=R_headset_world,
            scale_factor=scale_factor,
            q_init=kwargs.get("q_init"),
            dt=1.0 / control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
        )
        self.manipulator_config=manipulator_config
        self._start_time = 0
        self.control_rate_hz = control_rate_hz
        self.log_freq = log_freq
        self.visualize_placo = visualize_placo
        self.robot_controller= None
        if self.visualize_placo:
            self._init_placo_viz()
        self.gripper_config = None
        self._prev_b_button_state = False
        self._is_logging = False
        self.gripper_pos={}
        for name, config in self.manipulator_config.items():
            if "gripper_config" in config:
                self.gripper_config = config["gripper_config"]
                self.gripper_pos[name]=None


    def _robot_setup(self):
        """Initializes hardware-specific interfaces (e.g., CAN, ROS)."""
        self.robot_controller=KortexRobotController()#初始化gen3机械臂
        self.robot_controller.home_robot()#机械臂回零
        self.robot_controller.home_gripper()#夹爪回零


    def _update_robot_state(self):
        """Reads the current robot state from hardware and updates Placo."""
        robo_pos=self.robot_controller.get_joint_positions()
        self.placo_robot.state.q=robo_pos#不知道顺序对不对
        self.placo_robot.update_kinematics()
    
    def _update_ik(self):
        self._update_robot_state()
        self.placo_robot.update_kinematics()
        for src_name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[src_name] = xr_grip_val > 0.9
            
            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated.")
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_link_pose(config["link_name"])#激活机械臂，设置控制机械臂的关节初始参考位置

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)#获取控制器相对运动
                
                if self.effector_control_mode[src_name] == "position":
                    # Position-only control: only apply position delta
                    target_xyz = self.ref_ee_xyz[src_name] + delta_xyz#结合当前机械臂姿态更新目标
                    self.effector_task[src_name].target_world = target_xyz#设定目标
                else:
                    # Full pose control: apply both position and orientation deltas
                    target_xyz, target_quat = apply_delta_pose(
                        self.ref_ee_xyz[src_name],
                        self.ref_ee_quat[src_name],
                        delta_xyz,
                        delta_rot,
                    )
                    target_pose = tf.quaternion_matrix(target_quat)
                    target_pose[:3, 3] = target_xyz
                    self.effector_task[src_name].T_world_frame = target_pose
            else:#没按下trigger就取消追踪了，把控制器的参考位置也删掉，到时重新初始化控制器参考位置。
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_controller_xyz[src_name] = None
        try:
            self.solver.solve(True)#solve后直接更新placo model姿态
        except RuntimeError as e:
            print(f"IK solver failed: {e}")

    def _update_gripper_target(self):
        for gripper_name in self.manipulator_config.keys():
            if "gripper_config" not in self.manipulator_config[gripper_name]:
                continue

            gripper_config = self.manipulator_config[gripper_name]["gripper_config"]
            gripper_type = gripper_config["type"]
            if gripper_type == "parallel":
                trigger_value = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
                    # Calculate the target position based on the trigger value
                gripper_pos = calc_parallel_gripper_position(self.robot_controller.get_gripper_open_pos(), self.robot_controller.get_gripper_close_pos, trigger_value)
                self.gripper_pos[gripper_name] = gripper_pos
            else:
                # TODO: add dexterous hand support
                raise ValueError(f"Unsupported gripper type: {gripper_type}")
            
    def placo_q_to_kortex_deg(self, state_q, reorder=None):
        qj = state_q[7:] if self.floating_base else state_q
        qj_deg = np.rad2deg(qj)
        if reorder is not None:
            qj_deg = qj_deg[reorder]
        return qj_deg
    
    def _send_command(self):
        """Sends motor commands to the hardware."""
        joint_pose_target =self.placo_q_to_kortex_deg(self.placo_robot.state.q)
        self.robot_controller.set_joint_positions(joint_pose_target)
        for name, gripper_target in self.gripper_pos:
            self.robot_controller.set_gripper_position(gripper_target)
        print(f"joint_pose_target:{joint_pose_target}")
        print(f"gripper_target:{gripper_target}")
        if self.visualize_placo:
            self._update_placo_viz

    

    @abstractmethod
    def _get_robot_state_for_logging() -> Dict:
        """Returns a dictionary of robot-specific data for logging."""
        pass

#关机
    def _shutdown_robot(self):
        self.robot_controller.close()

    def _get_link_pose(self, link_name: str):
        """Gets the current world pose for a given link name from Placo."""
        T_world_link = self.placo_robot.get_T_world_frame(link_name)
        pos = T_world_link[:3, 3]
        quat = tf.quaternion_from_matrix(T_world_link)
        return pos, quat

    def _log_data(self):
        """Logs the current state of the robot and camera."""
        if not self.enable_log_data:
            return

        timestamp = time.time() - self._start_time
        data_entry = {"timestamp": timestamp}
        data_entry.update(self._get_robot_state_for_logging())

        if self.enable_camera and self.camera_interface:
            frames = self._get_camera_frame_for_logging()
            if frames:
                data_entry["image"] = frames

        self.data_logger.add_entry(data_entry)

    
    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for running the IK solver."""
        while not stop_event.is_set():
            start_time = time.time()
            self._update_robot_state()#复制粘贴对应机械臂的关节位置给state.q
            self._update_gripper_target()
            self._update_ik()#更新目标并求解，更新placo model位置
            if self.visualize_placo:
                self._update_placo_viz()
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("IK loop has stopped.")

    def _control_thread(self, stop_event: threading.Event):
        """Dedicated thread for sending commands to hardware."""
        while not stop_event.is_set():
            start_time = time.time()
            self._send_command()#发送控制命令给机械臂
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        self._shutdown_robot()
        print("Control loop has stopped.")

    def _data_logging_thread(self, stop_event: threading.Event):
        """Dedicated thread for data logging."""
        while not stop_event.is_set():
            start_time = time.time()
            self._check_logging_button()
            if self._is_logging:
                self._log_data()
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.log_freq) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("Data logging thread has stopped.")
#用于控制是否开始记录数据
    def _check_logging_button(self):
        """Checks for the 'B' button press to toggle data logging."""
        b_button_state = self.xr_client.get_button_state_by_name("B")
        right_axis_click = self.xr_client.get_button_state_by_name("right_axis_click")

        if b_button_state and not self._prev_b_button_state:
            self._is_logging = not self._is_logging
            if self._is_logging:
                print("--- Started data logging ---")
            else:
                print("--- Stopped data logging. Saving data... ---")
                self.data_logger.save()
                self.data_logger.reset()

        if right_axis_click and self._is_logging:
            print("--- Stopped data logging. Discarding data... ---")
            self.data_logger.reset()
            self._is_logging = False

        self._prev_b_button_state = b_button_state

    def _should_keep_running(self) -> bool:
        """Returns True if the main loop should continue running."""
        return not self._stop_event.is_set()

    def run(self):
        """Main entry point that starts all threads."""
        self._robot_setup()

        self._start_time = time.time()
        self._stop_event = threading.Event()
        threads = []

        core_threads = {
            "_ik_thread": self._ik_thread,
            "_control_thread": self._control_thread,
        }
        for name, target in core_threads.items():
            thread = threading.Thread(name=name, target=target, args=(self._stop_event,))
            threads.append(thread)

        if self.enable_log_data:
            log_thread = threading.Thread(
                name="_data_logging_thread",
                target=self._data_logging_thread,
                args=(self._stop_event,),
            )
            threads.append(log_thread)

        for t in threads:
            t.daemon = True
            t.start()

        print("Teleoperation running. Press Ctrl+C to exit.")
        try:
            while self._should_keep_running():
                all_threads_alive = all(t.is_alive() for t in threads)
                if not all_threads_alive:
                    print("A thread has died. Shutting down.")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            print("Shutting down...")
            self._stop_event.set()
            for t in threads:
                t.join(timeout=2.0)
            print("All threads have been shut down.")

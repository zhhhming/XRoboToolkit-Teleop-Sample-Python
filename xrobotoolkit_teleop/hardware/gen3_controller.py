#!/usr/bin/env python3

import threading
import time
import webbrowser
import os
import json
from typing import Dict, Any
from datetime import datetime

import cv2
import meshcat.transformations as tf
import numpy as np
import placo
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)
from google.protobuf import json_format

from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController
from xrobotoolkit_teleop.hardware.ruckigtrajectory   import RuckigTrajectoryPlanner
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import calc_parallel_gripper_position


class DataLogger:
    """Simple data logger for teleoperation sessions"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.data_entries = []
        self.session_start_time = None
        
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def start_session(self):
        """Start a new logging session"""
        self.session_start_time = datetime.now()
        self.data_entries = []
        print(f"Started logging session at {self.session_start_time}")
    
    def add_entry(self, data: Dict):
        """Add a data entry to the current session"""
        if self.session_start_time is not None:
            self.data_entries.append(data)
    
    def save_session(self):
        """Save the current session to a JSON file"""
        if not self.data_entries or self.session_start_time is None:
            print("No data to save")
            return
        
        timestamp_str = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"teleop_session_{timestamp_str}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        session_data = {
            "session_start_time": self.session_start_time.isoformat(),
            "session_duration": len(self.data_entries) / 50.0,  # Assuming 50Hz logging
            "num_entries": len(self.data_entries),
            "data": self.data_entries
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            print(f"Saved {len(self.data_entries)} entries to {filepath}")
        except Exception as e:
            print(f"Error saving session data: {e}")
    
    def reset(self):
        """Reset the current session"""
        self.data_entries = []
        self.session_start_time = None
        print("Reset logging session")


class HardwareTeleopController:
    """
    Hardware teleoperation controller for real robot arms using VR/XR input
    """

    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: dict,
        R_headset_world: np.ndarray,
        scale_factor: float,
        visualize_placo: bool,
        control_rate_hz: int,
        enable_log_data: bool,
        log_dir: str,
        log_freq: float,
        q_init: np.ndarray = None,
        joint_name_to_robot_index: Dict[str,int] = None,  # For handling joint order differences
        **kwargs,
    ):
        # Basic configuration
        self.robot_urdf_path = robot_urdf_path
        self.manipulator_config = manipulator_config
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.q_init = q_init
        self.dt = 1.0 / control_rate_hz
        if joint_name_to_robot_index == None:
            self.joint_name_to_robot_index = {
                "joint_1": 0, 
                "joint_2": 1, 
                "joint_3": 2,
                "joint_4": 3, 
                "joint_5": 4, 
                "joint_6": 5, 
                "joint_7": 6,
            }  # Map from placo joints to robot joints
        else:
            self.joint_name_to_robot_index = joint_name_to_robot_index
        
        # Control parameters
        self.control_rate_hz = control_rate_hz
        self.visualize_placo = visualize_placo
        
        # Logging setup
        self.enable_log_data = enable_log_data
        self.log_freq = log_freq
        if enable_log_data:
            self.data_logger = DataLogger(log_dir=log_dir)
        
        # XR Client
        self.xr_client = XrClient()
        
        # State tracking
        self._start_time = 0
        self._stop_event = threading.Event()
        self._is_logging = False
        self._prev_b_button_state = False
        
        # Robot controller
        self.robot_controller = None
        self.ruckig_planner = None
        
        # Control state variables
        self.ref_ee_xyz = {name: None for name in manipulator_config.keys()}
        self.ref_ee_quat = {name: None for name in manipulator_config.keys()}
        self.ref_controller_xyz = {name: None for name in manipulator_config.keys()}
        self.ref_controller_quat = {name: None for name in manipulator_config.keys()}
        self.active = {name: False for name in manipulator_config.keys()}
        self.gripper_pos = {}
        
        # Placo-related variables
        self.placo_robot = None
        self.solver = None
        self.effector_task = {}
        self.effector_control_mode = {}
        # 线程间共享：最新IK目标（deg）
        self._target_position_lock = threading.Lock()
        self._latest_ik_target = None

        # 机器人状态缓存（deg），由 控制线程刷新，IK线程只读
        self._robot_state_lock = threading.Lock()
        self._robot_pos_deg_cache = np.zeros(7, dtype=float)  # control 写、IK 读
        self._simulation_mode = True  # 新增：simulation模式标志

        # Ruckig 线程频率与计时 deque（ms）
        from collections import deque
        self.waypoint_rate_hz = 100.0      # 你可按需改
        self.control_rate_hz_ll = 300.0   # 低层控制 1 kHz
        self._waypoint_dt = 1.0 / self.waypoint_rate_hz
        self._control_dt_ll = 1.0 / self.control_rate_hz_ll
        self.control_loop_times = deque(maxlen=100)
        self.waypoint_loop_times = deque(maxlen=100)
        self.control_second = 1/self.control_rate_hz_ll
        
        
        # Initialize gripper positions
        for name, config in self.manipulator_config.items():
            if "gripper_config" in config:
                self.gripper_pos[name] = None

        print("Hardware Teleoperation Controller initialized")

    def _robot_setup(self):
        """Initialize the robot hardware interface"""
        print("Setting up robot hardware...")
        
        # Initialize robot controller
        if not self._simulation_mode:
            self.robot_controller = KortexRobotController() 
                # Home the robot
            print("Homing robot...")
            if not self.robot_controller.home_robot():
                raise RuntimeError("Failed to home robot")
        self.ruckig_planner = RuckigTrajectoryPlanner(control_cycle=self.control_second)
        if self._simulation_mode:
            self.ruckig_planner.set_simulation_mode(self._simulation_mode,[290, 15.9, 179, 229, 0, 54, 9])
       
        ok = self._initial_robot_pos_deg_cache()
        print("[INFO] initial pos cache from {}."
            .format("sim/hw" if ok else "zeros (fallback)"))
        
        
        print("Robot setup completed successfully")

    def _placo_setup(self):
        """Set up the placo inverse kinematics solver"""
        print("Setting up Placo IK solver...")
        
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        print("Joint names in the Placo model:")
        for i, joint_name in enumerate(self.placo_robot.model.names):
            print(f"  {i}: {joint_name}")

        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.dt
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Set initial configuration
        if self.q_init is not None:
            self.placo_robot.state.q=self.q_init.copy()
       
        try:
            self._update_robot_state()
        except Exception as e:
            print(f"[WARN] Initial robot->Placo sync failed, keep q_init/default: {e}")


        self.placo_robot.update_kinematics()

        # Set up end effector tasks
        for name, config in self.manipulator_config.items():
            control_mode = config.get("control_mode", "pose")
            self.effector_control_mode[name] = control_mode
            
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            
            if control_mode == "position":
                self.effector_task[name] = self.solver.add_position_task(config["link_name"], ee_xyz)
                print(f"Created position task for {name} -> {config['link_name']}")
            else:
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name] = self.solver.add_frame_task(config["link_name"], ee_target)
                print(f"Created pose task for {name} -> {config['link_name']}")
            
            self.effector_task[name].configure(name, "soft", 1.0)
            
            # Add manipulability task
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure("manipulability", "soft", 1e-2)

        self.placo_robot.update_kinematics()
        print("Placo setup completed")

    def _init_placo_viz(self):
        """Initialize Placo visualization"""
        if not self.visualize_placo:
            return
            
        print("Initializing Placo visualization...")
        self.placo_vis = robot_viz(self.placo_robot)
        webbrowser.open(self.placo_vis.viewer.url())
        self.placo_vis.display(self.placo_robot.state.q)
        
        for name, config in self.manipulator_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            
            if self.effector_control_mode[name] == "position":
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)
            else:
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

    def _update_placo_viz(self):
        """Update Placo visualization"""
        if not self.visualize_placo:
            return
            
        self.placo_vis.display(self.placo_robot.state.q)
        for name, config in self.manipulator_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            
            if self.effector_control_mode[name] == "position":
                target_frame = np.eye(4)
                target_frame[:3, 3] = self.effector_task[name].target_world
                frame_viz(f"vis_target_{name}", target_frame)
            else:
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

    def _idxq_nq(self, joint_name: str):
        """返回该关节在 q 中的起始 idx 和维度 nq"""
        jid = self.placo_robot.model.getJointId(joint_name)
        j = self.placo_robot.model.joints[jid]
        return j.idx_q, j.nq

    def _read_joint_rad(self, joint_name: str) -> float | None:
        """从 state.q 精确读取某个转动关节弧度值"""
        i0, nq = self._idxq_nq(joint_name)
        return float(self.placo_robot.state.q[i0])

    def _write_joint_rad(self, joint_name: str, value_rad: float) -> bool:
        """把标量弧度写回 state.q 的正确槽位；非标量关节直接跳过"""
        i0, nq = self._idxq_nq(joint_name)
        q = self.placo_robot.state.q.copy()
        q[i0] = float(value_rad)
        self.placo_robot.state.q = q
        return True

    def _placo_to_robot_deg_vector(self) -> np.ndarray:
        """
        从 Placo 精确读取 7 个关节转robot 索引放入数组。
        """
        # 计算目标长度
        if not self.joint_name_to_robot_index:
            raise RuntimeError("joint_name_to_robot_index is empty")
        n = max(self.joint_name_to_robot_index.values()) + 1

        robot_deg = np.full(n, np.nan, dtype=float)  # 先用 NaN 填充，便于发现缺失

        for name, idx in self.joint_name_to_robot_index.items():
            v_rad = self._read_joint_rad(name) 
            if v_rad is None:
                print(f"[WARN] skip {name}: not a scalar joint or not found.")
                continue
            robot_deg[idx] = np.degrees(v_rad)

        # 可选：强制要求都读到（7个都不是 NaN），否则抛错/返回
        if np.isnan(robot_deg).any():
            # 这里选择仅打印警告并继续运行；你也可以 raise
            missing = np.where(np.isnan(robot_deg))[0].tolist()
            print(f"[WARN] _placo_to_robot_deg_vector: missing indices {missing}.")
            raise RuntimeError(f"Failed to read joint values for indices: {missing}")
        return robot_deg


    def _robot_deg_to_placo(self, robot_deg: np.ndarray):
        """
        把机器人测得的 7 个关节角写回 Placo 的 state.q（rad）。
        """
        if robot_deg is None or len(robot_deg) < 7:
            raise ValueError("robot_deg must have 7 elements.")
        for name, idx in self.joint_name_to_robot_index.items():
            self._write_joint_rad(name, np.radians(float(robot_deg[idx])))
            print(f"deg to {name}:{np.radians(float(robot_deg[idx]))}")
    def _initial_robot_pos_deg_cache(self, *, max_retries: int = 8) -> bool:
        """
        Prime the joint-position cache before control threads start.
        Ensures _update_robot_state() can read a valid vector from _robot_pos_deg_cache.

        Returns:
            True  -> initialized from sim/hardware data
            False -> fell back to zeros
        """
        dof = int(getattr(self, "dof", 7))
        dt  = float(getattr(self, "_control_dt_ll", 0.004))

        pos = None

        # 1) Simulation snapshot (if any)
        try:
            if getattr(self, "_simulation_mode", False):
                sim_pos = np.asarray(self.ruckig_planner.sim_position, dtype=float).reshape(-1)
                if sim_pos.size >= dof and np.all(np.isfinite(sim_pos[:dof])):
                    pos = sim_pos[:dof].copy()
        except Exception as e:
            print(f"[WARN] initial cache: sim_position read failed: {e}")

        # 2) Hardware read with retries
        if pos is None:
            for k in range(max_retries):
                try:
                    p = self.robot_controller.get_joint_positions()  # should return deg
                    if p is None:
                        raise ValueError("get_joint_positions() returned None")
                    p = np.asarray(p, dtype=float).reshape(-1)
                    if p.size >= dof and np.all(np.isfinite(p[:dof])):
                        pos = p[:dof].copy()
                        break
                    else:
                        print(f"[WARN] initial cache: invalid joint vector (shape={p.shape})")
                except Exception as e:
                    print(f"[WARN] initial cache: read failed (try {k+1}/{max_retries}): {e}")
                time.sleep(dt)

        # 3) Final fallback
        used_fallback = False
        if pos is None:
            pos = np.zeros(dof, dtype=float)
            used_fallback = True
            print("[WARN] initial cache: fallback to zeros.")

        # 4) Optional clamp to joint limits if provided
        if hasattr(self, "joint_position_limits"):
            try:
                lo, hi = self.joint_position_limits  # expect arrays length==dof
                pos = np.clip(pos, np.asarray(lo, float), np.asarray(hi, float))
            except Exception as e:
                print(f"[WARN] initial cache: joint limits clamp skipped: {e}")

        # 5) Publish to cache (thread-safe)
        with self._robot_state_lock:
            self._robot_pos_deg_cache = pos.copy()

        return not used_fallback
    
    def _update_robot_state(self):
        """用缓存的关节角同步到 Placo（IK 线程用）"""
        try:
            with self._robot_state_lock:
                robot_positions = self._robot_pos_deg_cache.copy()  # deg
            if robot_positions.size < 7:
                return
            self._robot_deg_to_placo(robot_positions)
            self.placo_robot.update_kinematics()
            self.placo_vis.display(self.placo_robot.state.q)
        except Exception as e:
            print(f"Error updating robot state from cache: {e}")

    def _process_xr_pose(self, xr_pose, src_name):
        """Process XR controller pose and compute deltas"""
        # Get position and orientation
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = [
            xr_pose[6],  # w
            xr_pose[3],  # x
            xr_pose[4],  # y
            xr_pose[5],  # z
        ]

        # Transform controller position and orientation
        controller_xyz = self.R_headset_world @ controller_xyz

        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )

        # Calculate deltas
        if self.ref_controller_xyz[src_name] is None:
            self.ref_controller_xyz[src_name] = controller_xyz
            self.ref_controller_quat[src_name] = controller_quat
            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            delta_xyz = (controller_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], controller_quat)

        return delta_xyz, delta_rot

    def _get_link_pose(self, link_name: str):
        """Get current world pose for a given link name from Placo"""
        T_world_link = self.placo_robot.get_T_world_frame(link_name)
        pos = T_world_link[:3, 3]
        quat = tf.quaternion_from_matrix(T_world_link)
        return pos, quat
    
    def _set_ik_target_deg(self, q_deg: np.ndarray):
        with self._target_position_lock:
            self._latest_ik_target = q_deg.copy()

    def _update_ik(self):
        """Update inverse kinematics based on XR input"""
        for src_name, config in self.manipulator_config.items():
            # Check if controller is active
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[src_name] = xr_grip_val > 0.9
            
            if self.active[src_name]:
                # Initialize reference poses on activation
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated.")
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_link_pose(config["link_name"])

                # Get XR controller pose and compute deltas
                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)
                
                # Update target based on control mode
                if self.effector_control_mode[src_name] == "position":
                    target_xyz = self.ref_ee_xyz[src_name] + delta_xyz
                    self.effector_task[src_name].target_world = target_xyz
                else:
                    target_xyz, target_quat = apply_delta_pose(
                        self.ref_ee_xyz[src_name],
                        self.ref_ee_quat[src_name],
                        delta_xyz,
                        delta_rot,
                    )
                    target_pose = tf.quaternion_matrix(target_quat)
                    target_pose[:3, 3] = target_xyz
                    self.effector_task[src_name].T_world_frame = target_pose
            else:
                # Reset references when deactivated
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_controller_xyz[src_name] = None

        # Solve IK
        try:
            self.solver.solve(True)            
            self._set_ik_target_deg(self.placo_q_to_robot_deg())
        except RuntimeError as e:
            print(f"IK solver failed: {e}")

    def _update_gripper_target(self):
        """Update gripper target positions based on XR input"""
        for gripper_name in self.manipulator_config.keys():
            if "gripper_config" not in self.manipulator_config[gripper_name]:
                continue

            gripper_config = self.manipulator_config[gripper_name]["gripper_config"]
            gripper_type = gripper_config["type"]
            
            if gripper_type == "parallel":
                trigger_value = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
                
                # Calculate gripper position based on trigger value
                open_pos = 0.01
                close_pos = 0.99
                
                if open_pos is not None and close_pos is not None:
                    gripper_pos = calc_parallel_gripper_position(open_pos, close_pos, trigger_value)
                    self.gripper_pos[gripper_name] = gripper_pos
                else:
                    # Fallback to basic mapping if positions not calibrated
                    self.gripper_pos[gripper_name] = trigger_value
            else:
                raise ValueError(f"Unsupported gripper type: {gripper_type}")

    def placo_q_to_robot_deg(self):
        """Convert Placo joint positions to robot degrees"""     
        return self._placo_to_robot_deg_vector()


    def _check_logging_button(self):
        """Check for B button press to toggle data logging"""
        try:
            b_button_state = self.xr_client.get_button_state_by_name("B")
            
            # Detect button press (rising edge)
            if b_button_state and not self._prev_b_button_state:
                self._is_logging = not self._is_logging
                
                if self._is_logging:
                    print("--- Started data logging ---")
                    self.data_logger.start_session()
                else:
                    print("--- Stopped data logging. Saving data... ---")
                    self.data_logger.save_session()
                    self.data_logger.reset()
            
            # Check for right axis click to discard current session
            right_axis_click = self.xr_client.get_button_state_by_name("right_axis_click")
            if right_axis_click and self._is_logging:
                print("--- Stopped data logging. Discarding data... ---")
                self.data_logger.reset()
                self._is_logging = False
            
            self._prev_b_button_state = b_button_state
            
        except Exception as e:
            print(f"Error checking logging button: {e}")

    def _log_data(self):
        """Log current robot state"""
        if not self.enable_log_data or not self._is_logging:
            return
        
        try:
            timestamp = time.time() - self._start_time
            
            # Get robot state
            robot_joint_positions = self.robot_controller.get_joint_positions()
            robot_tool_pose = self.robot_controller.get_tool_pose()
            gripper_position = self.robot_controller.get_gripper_position()
            
            # Get Placo state
            placo_joint_positions = self.placo_q_to_robot_deg()
            
            # Prepare data entry
            data_entry = {
                "timestamp": timestamp,
                "robot_joint_positions_deg": robot_joint_positions.tolist() if len(robot_joint_positions) > 0 else [],
                "robot_tool_pose": robot_tool_pose.tolist() if len(robot_tool_pose) > 0 else [],
                "gripper_position": gripper_position,
                "placo_joint_positions_deg": placo_joint_positions.tolist(),
                "active_controllers": {name: active for name, active in self.active.items()},
                "gripper_targets": {name: pos for name, pos in self.gripper_pos.items() if pos is not None}
            }
            
            # Add XR controller data if available
            for name, config in self.manipulator_config.items():
                try:
                    xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                    trigger_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
                    data_entry[f"xr_{name}_pose"] = xr_pose
                    data_entry[f"xr_{name}_trigger"] = trigger_val
                except:
                    pass
            
            self.data_logger.add_entry(data_entry)
            
        except Exception as e:
            print(f"Error logging data: {e}")

    def _waypoint_thread(self, stop_event: threading.Event):
        """以 waypoint_rate_hz 从 _latest_ik_target 抽样写入 Ruckig waypoints"""
        print(f"Starting Waypoint thread at {self.waypoint_rate_hz}Hz...")
        last_target = None
        status_print_interval = 2.0
        last_status_time = time.time()

        while not stop_event.is_set():
            loop_start = time.time()
            try:
                with self._target_position_lock:
                    new_target = None if self._latest_ik_target is None else self._latest_ik_target.copy()#不会去清理self._latest_ik_target
                    # 读后不清空，允许丢多次（由去重/阈值控制）
                if new_target is not None:
                    if last_target is None or np.linalg.norm(new_target - last_target) > 0.05:
                        self.ruckig_planner.add_waypoint(new_target)
                        last_target = new_target

                # 周期状态打印（可选）
                now = time.time()
                if now - last_status_time > status_print_interval:
                    status = self.ruckig_planner.get_status()
                    avg_wp = (np.mean(self.waypoint_loop_times) * 1000) if self.waypoint_loop_times else 0.0
                    print(f"[WP] queued={status['num_waypoints']} processed={status['waypoints_processed']} "
                        f"sim={status['simulation_mode']} loop={avg_wp:.1f}ms")
                    last_status_time = now

            except Exception as e:
                print(f"Error in waypoint thread: {e}")

            elapsed = time.time() - loop_start
            self.waypoint_loop_times.append(elapsed)
            sleep_time = self._waypoint_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Waypoint thread stopped")

    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for IK computation"""
        print("Starting IK thread...")
        
        while not stop_event.is_set():
            start_time = time.time()
            
            try:
                self._update_robot_state()
                self._update_gripper_target()
                self._update_ik()
                self._update_robot_state()
                if self.visualize_placo:
                    self._update_placo_viz()
                    
            except Exception as e:
                print(f"Error in IK thread: {e}")
            
            # Maintain loop rate
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("IK thread stopped")
#要先确认有目标，要么就在init的时候把机械臂的坐标赋值到placo去
    def _control_thread(self, stop_event: threading.Event):
        """低层控制 loop：1kHz 读硬件位置→写缓存→Ruckig→发UDP速度"""
        print(f"Starting control thread at {self.control_rate_hz_ll}Hz...")

        status_print_interval = 2.0
        last_status_time = time.time()

        while not stop_event.is_set():
            loop_start = time.time()
            try:
                # 1) 高频直接读硬件位置（deg）
                if self._simulation_mode:
                    # Simulation模式：使用Ruckig的模拟位置
                    current_pos_deg = self.ruckig_planner.sim_position.copy()
                    current_speed = self.ruckig_planner.sim_velocity.copy()
                else:
                    current_pos_deg = self.robot_controller.get_joint_positions()
                    current_speed = self.robot_controller.get_joint_speeds()
                    if len(current_pos_deg) < 7:
                        # 读失败则维持频率
                        time.sleep(self._control_dt_ll)
                        continue
                    current_pos_deg = np.array(current_pos_deg, dtype=float)

                # 2) 把最新硬件位置写入缓存，供 IK 线程使用
                with self._robot_state_lock:
                    self._robot_pos_deg_cache = current_pos_deg.copy()
                # 3) Ruckig 走一步（用硬件读到的 current_pos）
                target_vel_deg_s, target_pos_deg, _ = self.ruckig_planner.compute_trajectory_step(
                    current_pos_deg,
                    current_speed
                )
                v_cmd = np.nan_to_num(target_vel_deg_s, nan=0.0, posinf=0.0, neginf=0.0)
                v_cmd = np.clip(
                    v_cmd,
                    -self.ruckig_planner.max_velocity,
                    self.ruckig_planner.max_velocity
                )
                # 4) 直接**发速度**UDP（优先尝试 set_joint_speeds_udp）
                global_cap = float(np.max(np.abs(self.ruckig_planner.max_velocity)))
                if not self._simulation_mode:
                    ret = self.robot_controller.send_joint_speeds_udp(v_cmd, speed_cap=global_cap)
                    if not ret.get("ok", False):
                        print(f"[WARN] send_joint_speeds_udp failed: {ret.get('err')}")
                        self.robot_controller.send_joint_speeds_udp(np.zeros_like(v_cmd), speed_cap=global_cap)
                

                # 周期状态打印
                now = time.time()
                if now - last_status_time > status_print_interval:
                    status = self.ruckig_planner.get_status()
                    avg_ctrl = (np.mean(self.control_loop_times) * 1000) if self.control_loop_times else 0.0
                    avg_wp = (np.mean(self.waypoint_loop_times) * 1000) if self.waypoint_loop_times else 0.0
                    print(f"[CTRL] vel={status['current_velocity_norm']:.1f}°/s "
                        f"tgtVel={status['filtered_target_velocity_norm']:.1f}°/s "
                        f"steps={status['trajectory_steps']} "
                        f"loop={avg_ctrl:.1f}ms wp_loop={avg_wp:.1f}ms")
                    last_status_time = now

            except Exception as e:
                print(f"Error in control thread: {e}")

            # 定频
            elapsed = time.time() - loop_start
            self.control_loop_times.append(elapsed)
            sleep_time = self._control_dt_ll - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > self._control_dt_ll * 2:
                print(f"Warning: Control loop overrun ({elapsed*1000:.1f}ms)")


        print("Control thread stopped")
        self._shutdown_robot()

    def _data_logging_thread(self, stop_event: threading.Event):
        """Dedicated thread for data logging"""
        print("Starting data logging thread...")
        
        while not stop_event.is_set():
            start_time = time.time()
            
            try:
                self._check_logging_button()
                self._log_data()
            except Exception as e:
                print(f"Error in logging thread: {e}")
            
            # Maintain loop rate
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.log_freq) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("Data logging thread stopped")

    def _shutdown_robot(self):
        """Safely shutdown robot connection"""
        try:
            if self.robot_controller:
                self.robot_controller.close()
                print("Robot controller closed")
        except Exception as e:
            print(f"Error closing robot controller: {e}")

    def run(self):
        """Main entry point - starts all control threads"""
        print("Starting Hardware Teleoperation Controller...")
        
        # Setup robot and placo
        self._robot_setup()
        self._placo_setup()
        
        if self.visualize_placo:
            self._init_placo_viz()
        
        # Initialize timing
        self._start_time = time.time()
        self._stop_event = threading.Event()
        if not self._simulation_mode:
            self.robot_controller.enter_velocity_control()
        # Create and start threads
        threads = []
        
        # 1) IK 线程
        ik_thread = threading.Thread(
            name="IK_Thread", 
            target=self._ik_thread, 
            args=(self._stop_event,)
        )
        threads.append(ik_thread)

        # 2) Waypoint 线程（新增）
        wp_thread = threading.Thread(
            name="Waypoint_Thread",
            target=self._waypoint_thread,
            args=(self._stop_event,)
        )
        threads.append(wp_thread)

        # 3) 控制线程
        ctrl_thread = threading.Thread(
            name="Control_Thread",
            target=self._control_thread,
            args=(self._stop_event,)
        )
        threads.append(ctrl_thread)
        
        
        # Optional logging thread
        if self.enable_log_data:
            logging_thread = threading.Thread(
                name="Logging_Thread",
                target=self._data_logging_thread,
                args=(self._stop_event,)
            )
            threads.append(logging_thread)
        
        # Start all threads
        for thread in threads:
            thread.daemon = True
            thread.start()
        
        print("Teleoperation running. Press Ctrl+C to exit.")
        
        try:
            while not self._stop_event.is_set():
                # Check if all threads are still alive
                all_threads_alive = all(t.is_alive() for t in threads)
                if not all_threads_alive:
                    print("A critical thread has died. Shutting down.")
                    break
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            print("Shutting down...")
            self._stop_event.set()
            
            # Wait for threads to finish
            for thread in threads:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    print(f"Warning: {thread.name} did not shut down gracefully")
            self._shutdown_robot()
            print("All threads shut down.")


# Example usage
if __name__ == "__main__":
    # Example configuration
    manipulator_config = {
        "right_arm": {
            "link_name": "bracelet_link",
            "pose_source": "right_controller",
            "control_trigger": "right_trigger",
            "control_mode": "pose",  
            # "gripper_config": {
            #     "type": "parallel",
            #     "gripper_trigger": "right_grip",
            # }
        }
    }
    
    R_headset_world = R_HEADSET_TO_WORLD
    

    
    try:
        controller = HardwareTeleopController(
            robot_urdf_path="/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-7DOF.urdf",
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            scale_factor=1.0,
            visualize_placo=True,
            control_rate_hz=200,
            enable_log_data=False,
            log_dir="teleop_logs",
            log_freq=50.0,
        )
        
        controller.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

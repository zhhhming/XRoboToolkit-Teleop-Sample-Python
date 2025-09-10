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
        floating_base: bool,
        scale_factor: float,
        visualize_placo: bool,
        control_rate_hz: int,
        enable_log_data: bool,
        log_dir: str,
        log_freq: float,
        q_init: np.ndarray = None,
        joint_reorder_map: np.ndarray = None,  # For handling joint order differences
        **kwargs,
    ):
        # Basic configuration
        self.robot_urdf_path = robot_urdf_path
        self.manipulator_config = manipulator_config
        self.floating_base = floating_base
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.q_init = q_init
        self.dt = 1.0 / control_rate_hz
        self.joint_reorder_map = joint_reorder_map  # Map from placo joints to robot joints
        
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
        
        # Initialize gripper positions
        for name, config in self.manipulator_config.items():
            if "gripper_config" in config:
                self.gripper_pos[name] = None

        print("Hardware Teleoperation Controller initialized")

    def _robot_setup(self):
        """Initialize the robot hardware interface"""
        print("Setting up robot hardware...")
        
        # Initialize robot controller
        self.robot_controller = KortexRobotController()
        
        # Home the robot
        print("Homing robot...")
        if not self.robot_controller.home_robot():
            raise RuntimeError("Failed to home robot")
        
        # Home the gripper
        print("Homing gripper...")
        self.robot_controller.home_gripper()
        
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

        # Set initial configuration
        if self.q_init is not None:
            if self.floating_base:
                self.placo_robot.state.q = self.q_init.copy()
            else:
                self.solver.mask_fbase(True)
                self.placo_robot.state.q[7:] = self.q_init.copy()
        else:
            if not self.floating_base:
                self.solver.mask_fbase(True)
                self.placo_robot.state.q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # Identity quaternion for base
          #应该可能没有前7位吧
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

    def _update_robot_state(self):
        """Read current robot state and update Placo model"""
        try:
            # Get current joint positions from robot
            robot_positions = self.robot_controller.get_joint_positions()
            
            if len(robot_positions) == 0:
                print("Warning: Failed to get robot joint positions")
                return
            
            # Convert degrees to radians
            robot_positions_rad = np.deg2rad(robot_positions)
            
            # Handle joint reordering if needed
            if self.joint_reorder_map is not None:
                # Reorder joints according to mapping
                placo_positions = robot_positions_rad[self.joint_reorder_map]
            else:
                placo_positions = robot_positions_rad
            
            # Update Placo robot state
            if self.floating_base:
                # Keep base pose unchanged, update joint positions
                self.placo_robot.state.q[7:] = placo_positions
            else:
                # For fixed base, update all joint positions
                self.placo_robot.state.q = placo_positions
            
            self.placo_robot.update_kinematics()
            
        except Exception as e:
            print(f"Error updating robot state: {e}")

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
                open_pos = self.robot_controller.get_gripper_open_pos()
                close_pos = self.robot_controller.get_gripper_close_pos()
                
                if open_pos is not None and close_pos is not None:
                    gripper_pos = calc_parallel_gripper_position(open_pos, close_pos, trigger_value)
                    self.gripper_pos[gripper_name] = gripper_pos
                else:
                    # Fallback to basic mapping if positions not calibrated
                    self.gripper_pos[gripper_name] = trigger_value
            else:
                raise ValueError(f"Unsupported gripper type: {gripper_type}")

    def placo_q_to_robot_deg(self, state_q):
        """Convert Placo joint positions to robot degrees"""
        # Extract joint positions (skip base if floating base)
        qj = state_q[7:] if self.floating_base else state_q
        
        # Convert to degrees
        qj_deg = np.rad2deg(qj)
        
        # Handle joint reordering if needed (reverse mapping)
        if self.joint_reorder_map is not None:
            # Create reverse mapping
            robot_positions = np.zeros_like(qj_deg)
            robot_positions[self.joint_reorder_map] = qj_deg
            return robot_positions
        
        return qj_deg

    def _send_command(self):
        """Send computed commands to robot hardware"""
        try:
            # Get target joint positions from Placo
            joint_pose_target = self.placo_q_to_robot_deg(self.placo_robot.state.q)
            
            # Send joint positions to robot
            self.robot_controller.set_joint_positions(joint_pose_target)
            
            # Send gripper commands
            for name, gripper_target in self.gripper_pos.items():
                if gripper_target is not None:
                    self.robot_controller.set_gripper_position(gripper_target)
            
            # Debug output
            if len(self.gripper_pos) > 0:
                gripper_targets = [f"{name}: {pos:.3f}" for name, pos in self.gripper_pos.items() if pos is not None]
                print(f"Joint targets: {joint_pose_target}, Gripper: {gripper_targets}")
            
        except Exception as e:
            print(f"Error sending command: {e}")

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
            placo_joint_positions = self.placo_q_to_robot_deg(self.placo_robot.state.q)
            
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

    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for IK computation"""
        print("Starting IK thread...")
        
        while not stop_event.is_set():
            start_time = time.time()
            
            try:
                self._update_robot_state()
                self._update_gripper_target()
                self._update_ik()
                
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
        """Dedicated thread for sending commands to robot"""
        print("Starting control thread...")
        
        while not stop_event.is_set():
            start_time = time.time()
            
            try:
                self._send_command()
            except Exception as e:
                print(f"Error in control thread: {e}")
            
            # Maintain loop rate
            elapsed_time = time.time() - start_time
            sleep_time = (1.0 / self.control_rate_hz) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
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
        
        # Create and start threads
        threads = []
        
        # Core control threads
        ik_thread = threading.Thread(
            name="IK_Thread", 
            target=self._ik_thread, 
            args=(self._stop_event,)
        )
        control_thread = threading.Thread(
            name="Control_Thread", 
            target=self._control_thread, 
            args=(self._stop_event,)
        )
        
        threads.extend([ik_thread, control_thread])
        
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
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_grip",
            }
        }
    }
    
    # Headset to world transformation (identity for this example)
    R_headset_world = R_HEADSET_TO_WORLD
    
    # Joint reordering map (if needed)
    # Example: if placo joint order [0,1,2,3,4,5,6] maps to robot order [6,5,4,3,2,1,0]
    joint_reorder_map = None  # np.array([6,5,4,3,2,1,0])
    
    try:
        controller = HardwareTeleopController(
            robot_urdf_path="D:\xrobotics\XRoboToolkit-Teleop-Sample-Python\assets\arx\Gen\GEN3-6DOF.urdf",
            manipulator_config=manipulator_config,
            R_headset_world=R_headset_world,
            floating_base=False,
            scale_factor=1.0,
            visualize_placo=True,
            control_rate_hz=100,
            enable_log_data=True,
            log_dir="teleop_logs",
            log_freq=50.0,
            joint_reorder_map=joint_reorder_map,
        )
        
        controller.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

#floating base去掉，send control前，先让placo和机械臂关节位置一致
#!/usr/bin/env python3

import numpy as np
import time
import threading
import sys
import os

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient

from kortex_api.autogen.messages import Session_pb2, Base_pb2, BaseCyclic_pb2, ActuatorConfig_pb2
from kortex_api.Exceptions.KServerException import KServerException

# 连接参数
ROBOT_IP = "192.168.1.10"
ROBOT_TCP_PORT = 10000
ROBOT_UDP_PORT = 10001
USERNAME = "admin"
PASSWORD = "admin"
SESSION_INACTIVITY_TIMEOUT = 60000  # milliseconds
CONNECTION_INACTIVITY_TIMEOUT = 2000  # milliseconds


TIMEOUT_DURATION = 20  # seconds


class KortexRobotController:
    """
    Kortex Robot Arm Controller Class
    Provides high-level control interface for Kinova Kortex robot arms
    """
    
    def __init__(self):
        """Initialize the robot controller with TCP and UDP connections"""
        
        print(f"Initializing Kortex Robot Controller...")
        print(f"Connecting to robot at IP: {ROBOT_IP}")
        
        # Initialize transport layers
        self.tcp_transport = TCPTransport()
        self.udp_transport = UDPTransport()
        
        # Initialize routers
        error_callback = lambda kException: print(f"Router Error: {kException}")
        self.tcp_router = RouterClient(self.tcp_transport, error_callback)
        self.udp_router = RouterClient(self.udp_transport, error_callback)
        
        # Connect transports
        print("Establishing TCP connection...")
        self.tcp_transport.connect(ROBOT_IP, ROBOT_TCP_PORT)#这个tcp udptransport要连不同的port吗
        print("TCP connection established successfully")
        
        print("Establishing UDP connection...")
        self.udp_transport.connect(ROBOT_IP, ROBOT_UDP_PORT)
        print("UDP connection established successfully")
        
        # Create session for TCP connection
        self._create_session()
        
        # Initialize clients
        self.base_client = BaseClient(self.tcp_router)
        self.base_cyclic_client = BaseCyclicClient(self.udp_router)
        self.actuator_config = ActuatorConfigClient(self.tcp_router)
        
        print("Robot clients initialized successfully")
        
        # Get actuator information
        self.actuator_count = self.base_client.GetActuatorCount()
        print(f"Robot has {self.actuator_count.count} actuators")
        
        # Print actuator IDs
        for i in range(self.actuator_count.count):
            print(f"Actuator ID: {i}")

        self.device_config=DeviceConfigClient(self.tcp_router)
        print(self.device_config.GetDeviceType())
        print(self.base_client.GetArmState()) 

        # Get and display current joint positions
        current_positions = self.get_joint_positions()
        print(f"Current joint positions: {current_positions}")
        
        # Set servoing mode to single level
        self._set_single_level_servoing()
        self.gripper_open_pos=0.01
        self.gripper_close_pos=0.99
        self.in_low_level_mode = False
        print("Kortex Robot Controller initialization complete!")
    
    def _create_session(self):
        """Create session for robot communication"""
        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = USERNAME
        session_info.password = PASSWORD
        session_info.session_inactivity_timeout = SESSION_INACTIVITY_TIMEOUT
        session_info.connection_inactivity_timeout = CONNECTION_INACTIVITY_TIMEOUT
        
        print(f"Creating session for tcp_router with username: {USERNAME}")
        self.tcp_session_manager = SessionManager(self.tcp_router)
        self.tcp_session_manager.CreateSession(session_info)
        print("Session for tcp created successfully")

        print(f"Creating session for udp_router with username: {USERNAME}")
        self.udp_session_manager = SessionManager(self.udp_router)
        self.udp_session_manager.CreateSession(session_info)
        print("Session for udp created successfully")
    
    def _set_single_level_servoing(self):
        """Set robot to single level servoing mode"""
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base_client.SetServoingMode(base_servo_mode)
        print("Servoing mode set to SINGLE_LEVEL_SERVOING")

    def _ensure_base_command_ready(self):
        """
        确保在低层(LOW_LEVEL)下，self.base_command已准备好：
        - 有与执行器数量一致的actuators槽位
        - flags=1 (使能)
        - 初始position = 当前反馈位置（防跟随误差）
        """
        if not getattr(self, "in_low_level_mode", False):
            raise RuntimeError("Must be in LOW_LEVEL_SERVOING before preparing base_command.")

        # 若还没有命令或数量不匹配，则重建
        need_rebuild = (
            not hasattr(self, "base_command")
            or len(self.base_command.actuators) != self.actuator_count.count
        )
        if need_rebuild:
            fb = self.base_cyclic_client.RefreshFeedback()
            self.base_feedback = fb

            self.base_command = BaseCyclic_pb2.Command()
            self.base_command.frame_id = 0

            # 逐个执行器建槽位并设置 flags/position
            for i in range(self.actuator_count.count):
                a = self.base_command.actuators.add()
                a.flags = 1
                a.position = fb.actuators[i].position
                a.velocity = 0.0
                a.torque_joint = 0.0
                a.command_id = 0

            # 若有夹爪，也准备一下（可选）
            if fb.HasField('interconnect') and len(fb.interconnect.gripper_feedback.motor) > 0:
                _ = self.base_command.interconnect.gripper_command  # 占位即可

            # 先发一帧建立连续性
            self.base_feedback = self.base_cyclic_client.Refresh(self.base_command)

        if not hasattr(self, "_frame_id"):
            self._frame_id = 0
    def enter_velocity_control(self):
        """
        将 Base 切到 LOW_LEVEL，并把所有执行器切换为 VELOCITY 控制模式。
        """
        # 先确保低层模式
        if not getattr(self, "in_low_level_mode", False):
            self.enter_low_level_mode()

        # 准备命令帧
        self._ensure_base_command_ready()

        # 逐个执行器切模式：VELOCITY
        ctrl = ActuatorConfig_pb2.ControlModeInformation()
        ctrl.control_mode = ActuatorConfig_pb2.ControlMode.Value('VELOCITY')

        # Kortex 执行器 device_id 从 1 开始
        for device_id in range(1, self.actuator_count.count + 1):
            # 带重试更稳妥
            ok = True
            try:
                self.actuator_config.SetControlMode(ctrl, device_id)
            except Exception as e:
                print(f"[WARN] Set VELOCITY failed on actuator {device_id}: {e}")
                ok = False
            if not ok:
                # 可按需再次重试或抛异常
                pass

        self.in_velocity_mode = True
        print("All actuators are now in VELOCITY control mode.")

    def exit_velocity_control(self):
        """
        将所有执行器切回 POSITION 控制模式（常用的高层安全状态）。
        不改变 Base 的 ServoingMode；由调用方决定是否退出低层。
        """
        if not getattr(self, "in_velocity_mode", False):
            return

        ctrl = ActuatorConfig_pb2.ControlModeInformation()
        ctrl.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')

        for device_id in range(1, self.actuator_count.count + 1):
            try:
                self.actuator_config.SetControlMode(ctrl, device_id)
            except Exception as e:
                print(f"[WARN] Set POSITION failed on actuator {device_id}: {e}")

        self.in_velocity_mode = False
        print("All actuators are back to POSITION control mode.")

    def enter_low_level_mode(self):
        """进入低级控制模式"""
        print("Entering low-level servoing mode...")
        
        # 保存当前伺服模式
        self.previous_servoing_mode = self.base_client.GetServoingMode()

        try:
            self.base_client.Stop()
            time.sleep(0.05)
        except Exception:
            pass
        
        # 设置为低级伺服模式
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base_client.SetServoingMode(base_servo_mode)

        # 3) 等待确认（最多 3s）
        ok = False
        for _ in range(300):
            mode = self.base_client.GetServoingMode().servoing_mode
            if mode == Base_pb2.LOW_LEVEL_SERVOING:
                ok = True
                break
            time.sleep(0.01)
        if not ok:
            raise RuntimeError("Failed to enter LOW_LEVEL_SERVOING (timeout)")

        
        # 获取初始反馈
        self.base_feedback = self.base_cyclic_client.RefreshFeedback()
        
        # 初始化命令为当前状态
        self.base_command = BaseCyclic_pb2.Command()
        self.base_command.frame_id = 0
        self.base_command.interconnect.command_id.identifier = 0
        self.base_command.interconnect.gripper_command.command_id.identifier = 0
        
        # 初始化关节命令为当前位置
        for i in range(self.actuator_count.count):
            actuator_command = self.base_command.actuators.add()
            actuator_command.flags = 1
            actuator_command.position = self.base_feedback.actuators[i].position
            actuator_command.velocity = 0.0
            actuator_command.torque_joint = 0.0
            actuator_command.current_motor = 0.0
        
        # 初始化gripper命令为当前状态
        if self.base_feedback.HasField('interconnect')and len(self.base_feedback.interconnect.gripper_feedback.motor) > 0:
            grip_cmd = self.base_command.interconnect.gripper_command
            grip_cmd.Clear()
            m = grip_cmd.motor_cmd.add()
            m.position = self.base_feedback.interconnect.gripper_feedback.motor[0].position
            m.velocity = 0.0
            m.force = 0.0
        
        # 发送第一个命令以建立连续性
        self.base_command.frame_id = 0
        self.base_feedback = self.base_cyclic_client.Refresh(self.base_command)
        self._frame_id = 0

        self.in_low_level_mode = True
        print("Low-level servoing mode activated")

    def exit_low_level_mode(self):
        """退出低级控制模式"""
        print("Exiting low-level servoing mode...")
        
        if hasattr(self, 'previous_servoing_mode'):
            self.base_client.SetServoingMode(self.previous_servoing_mode)
            print("Servoing mode restored")
        else:
            # 恢复到单级伺服模式
            self._set_single_level_servoing()
        
        self.in_low_level_mode = False
        print("Low-level servoing mode deactivated")
    
    def send_joint_speeds_udp(
    self,
    velocities,
    *,
    speed_cap: float | None = None,   # 可选：对输入速度幅值做上限（单位：deg/s）
    ):
        """
        使用 BaseCyclic (UDP) 在 VELOCITY 模式下发送关节速度指令（单位：deg/s）

        velocities: 长度 = 执行器数量 的数组（deg/s）
        """
        # 1) 确保处于低层 & 速度模式
        if not getattr(self, "in_low_level_mode", False):
            print("[INFO] Not in LOW_LEVEL, entering...")
            self.enter_low_level_mode()

        # 若不是 VELOCITY，切过去
        try:
            # 简单检查一个执行器的模式（可选）
            pass
        except:
            pass

        if not getattr(self, "in_velocity_mode", False):
            print("[INFO] Not in VELOCITY mode, switching all actuators to VELOCITY...")
            self.enter_velocity_control()

        # 2) 检查输入长度
        if len(velocities) != self.actuator_count.count:
            return {"ok": False, "err": f"Expected {self.actuator_count.count} velocities, got {len(velocities)}"}

        # 3) 确保命令帧准备好
        self._ensure_base_command_ready()

        # 4) 刷新一次反馈，准备把 position 跟随到测量（安全）
        fb = self.base_cyclic_client.RefreshFeedback()

        # 5) 写入每个关节的速度；位置跟随测量值（防跟随误差）
        for i in range(self.actuator_count.count):
            v = float(velocities[i])
            if speed_cap is not None:
                # cap = abs(float(speed_cap))
                cap = abs(float(20))
                v = max(-cap, min(cap, v))
            self.base_command.actuators[i].position = fb.actuators[i].position  # 跟随测量
            self.base_command.actuators[i].velocity = v
            self.base_command.actuators[i].flags = 1
            self.base_command.actuators[i].command_id = (getattr(self, "_frame_id", 0) + 1) & 0xFFFF

        # 6) 帧号自增（16位回绕）
        self._frame_id = (getattr(self, "_frame_id", 0) + 1) & 0xFFFF
        self.base_command.frame_id = self._frame_id
        MAX_RETRIES = 3
    
        for attempt in range(MAX_RETRIES):
        # 7) 发送一帧
            try:
                self.base_feedback = self.base_cyclic_client.Refresh(self.base_command)
                return {"ok": True}
            except KServerException as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"[WARN] Attempt {attempt+1} failed, retrying: {e}")
                    time.sleep(0.001)  # 短暂延迟后重试
                    continue
                else:
                    return {"ok": False, "err": f"Failed after {MAX_RETRIES} attempts: {e}"}
            except Exception as e:
                return {"ok": False, "err": f"{e}"}



    def set_gripper_position_udp(
        self,
        position,
        *,
        kp: float = 2.0,            # 比例增益：%/s per %
        vel_cap_pct: float = 20.0, # 速度上限（百分比）
        tol_pct: float = 1.5,       # 到位阈值（百分比）
        force_pct: float | None = None
    ):
        """
        使用UDP BaseCyclic设置gripper位置 (低级控制模式)
        
        Args:
            position: gripper position (0.0 to 1.0 or actual position value)
        """
        if not hasattr(self, 'in_low_level_mode') or not self.in_low_level_mode:
            print("Warning: Not in low-level mode. Entering low-level mode...")
            self.enter_low_level_mode()
        grip_cmd = self.base_command.interconnect.gripper_command
        if len(grip_cmd.motor_cmd) == 0:
            m = grip_cmd.motor_cmd.add()
        else:
            m = grip_cmd.motor_cmd[0]

        def to_percent(x):
            x = float(x)
            open_pos = getattr(self, 'gripper_open_pos', None)
            close_pos = getattr(self, 'gripper_close_pos', None)
            if open_pos is not None and close_pos is not None and abs(open_pos - close_pos) > 1e-6:
                lo, hi = (open_pos, close_pos) if open_pos < close_pos else (close_pos, open_pos)
                alpha = (x - lo) / (hi - lo)
                return max(0.0, min(100.0, alpha * 100.0))
            if 0.0 <= x <= 1.2:
                return max(0.0, min(100.0, x * 100.0))
            return max(0.0, min(100.0, x))
        
        def fb_to_percent(v):
            v = float(v)
            return v * 100.0 if 0.0 <= v <= 1.2 else v
        try:
            tgt_pct = to_percent(position)
            fb = self.base_cyclic_client.RefreshFeedback()
            cur_raw = fb.interconnect.gripper_feedback.motor[0].position if len(fb.interconnect.gripper_feedback.motor) > 0 else 0.0
            cur_pct = fb_to_percent(cur_raw)#反馈的夹爪位置0到1,转成百分比
            err = tgt_pct - cur_pct
            reached = (abs(err) <= tol_pct)
            m.position = float(tgt_pct)
            if hasattr(m, 'velocity'):
                v = min(vel_cap_pct, max(0.0, kp * abs(err)))
                m.velocity = float(v if not reached else 0.0)
            if hasattr(m, 'force'):
                f = force_pct if force_pct is not None else getattr(self, 'gripper_force', 100.0)
                m.force = float(max(0.0, min(100.0, f)))

            # 4) 帧号递增 + 刷新一帧
            self._frame_id = (getattr(self, "_frame_id", 0) + 1) & 0xFFFF
            self.base_command.frame_id = self._frame_id
            self.base_feedback = self.base_cyclic_client.Refresh(self.base_command)

            return {"ok": True, "reached": reached, "err_pct": float(err)}

            
        except Exception as e:
            print(f"ERROR setting gripper position via UDP: {e}")
            return {"ok": False, "reached": False, "err_pct": None}

    def _check_for_end_or_abort(self, event):
        """Callback function to check for action completion"""
        def check(notification, e=event):
            print(f"Action Event: {Base_pb2.ActionEvent.Name(notification.action_event)}")
            if notification.action_event == Base_pb2.ACTION_END:
                print("Action completed successfully")
                e.set()
            elif notification.action_event == Base_pb2.ACTION_ABORT:
                print("Action aborted")
                e.set()
        return check
    
    def home_robot(self):
        """Move robot to home position"""
        print("Starting robot homing sequence...")
        
        # Get home action
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base_client.ReadAllActions(action_type)
        
        home_action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                home_action_handle = action.handle
                break
        
        if home_action_handle is None:
            print("ERROR: Home action not found!")
            return False
        
        # Set up event and notification
        completion_event = threading.Event()
        notification_handle = self.base_client.OnNotificationActionTopic(
            self._check_for_end_or_abort(completion_event),
            Base_pb2.NotificationOptions()
        )
        
        print("Executing home action...")
        self.base_client.ExecuteActionFromReference(home_action_handle)
        
        # Wait for completion
        finished = completion_event.wait(TIMEOUT_DURATION)
        self.base_client.Unsubscribe(notification_handle)
        
        if finished:
            print("Robot homing completed successfully!")
            return True
        else:
            print("ERROR: Robot homing timed out!")
            return False
    

    def home_gripper(self):
        """Home the gripper and determine max/min positions"""
        
        print("Starting gripper homing sequence...")
        
        # Create gripper command and request objects
        gripper_command = Base_pb2.GripperCommand()
        gripper_request = Base_pb2.GripperRequest()
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        
        # Open gripper to find max position
        print("Opening gripper to find maximum position...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = 0.1  # Positive speed opens gripper
        self.base_client.SendGripperCommand(gripper_command)
        
        # Wait for gripper to stop (speed = 0)
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base_client.GetMeasuredGripperMovement(gripper_request)
            if len(gripper_measure.finger):
                current_speed = gripper_measure.finger[0].value
                print(f"Current gripper speed: {current_speed}")
                if abs(current_speed) < 0.01:  # Speed close to 0
                    break
            else:
                break
            time.sleep(0.1)
        
        # Get max position
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base_client.GetMeasuredGripperMovement(gripper_request)
        if len(gripper_measure.finger):
            self.gripper_open_pos = gripper_measure.finger[0].value
            print(f"Gripper maximum position: {self.gripper_open_pos}")
        
        # Close gripper to find min position
        print("Closing gripper to find minimum position...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = -0.1  # Negative speed closes gripper
        self.base_client.SendGripperCommand(gripper_command)
        
        # Wait for gripper to stop
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base_client.GetMeasuredGripperMovement(gripper_request)
            if len(gripper_measure.finger):
                current_speed = gripper_measure.finger[0].value
                print(f"Current gripper speed: {current_speed}")
                if abs(current_speed) < 0.01:  # Speed close to 0
                    break
            else:
                break
            time.sleep(0.1)
        
        # Get min position
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base_client.GetMeasuredGripperMovement(gripper_request)
        if len(gripper_measure.finger):
            self.gripper_close_pos = gripper_measure.finger[0].value
            print(f"Gripper minimum position: {self.gripper_close_pos}")
        
        # Return gripper to max (open) position
        print("Returning gripper to open position...")
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.value = self.gripper_open_pos
        self.base_client.SendGripperCommand(gripper_command)
        
        print("Gripper homing completed successfully!")
        print(f"Gripper range: {self.gripper_close_pos} to {self.gripper_open_pos}")
    
    def set_joint_positions(self, positions, reverse_order=False):
        """
        Set joint positions
        
        Args:
            positions: numpy array of joint positions (in degrees)
            reverse_order: if True, reverse the order of joint assignment
        """
        if len(positions) != self.actuator_count.count:
            print(f"ERROR: Expected {self.actuator_count.count} positions, got {len(positions)}")
            return False
        
        print(f"Setting joint positions: {positions}")
        
        # Create action
        action = Base_pb2.Action()
        action.name = "Set Joint Positions"
        action.application_data = ""
        
        # Set joint angles
        for joint_id in range(len(positions)):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            
            if reverse_order:
                joint_angle.joint_identifier = self.actuator_count.count - 1 - joint_id
            else:
                joint_angle.joint_identifier = joint_id
                
            joint_angle.value = float(positions[joint_id])
        
        # Execute action (non-blocking)
        self.base_client.ExecuteAction(action)
        print("Joint position command sent")
        return True
    
    def get_joint_positions(self):
        """
        Get current joint positions
        
        Returns:
            numpy array of current joint positions (in degrees)
        """
        try:
            feedback = self.base_cyclic_client.RefreshFeedback()
            positions = []
            
            for actuator in feedback.actuators:
                deg=actuator.position
                if deg >= 180.0:
                    deg -= 360.0
                positions.append(deg)
            
            return np.array(positions)
        
        except Exception as e:
            print(f"ERROR getting joint positions: {e}")
            return np.array([])
        
    def get_joint_speeds(self):
        """
        Get current joint velocities

        Returns:
            numpy array of current joint velocities (in degrees/second)
        """
        try:
            feedback = self.base_cyclic_client.RefreshFeedback()
            speeds = []
            for actuator in feedback.actuators:
                speeds.append(actuator.velocity)  # deg/s from BaseCyclic feedback
            return np.array(speeds, dtype=float)
        except Exception as e:
            print(f"ERROR getting joint speeds: {e}")
            return np.array([])
    def get_tool_pose(self):
        """
        Get current tool pose
        
        Returns:
            numpy array [x, y, z, theta_x, theta_y, theta_z]
        """
        try:
            feedback = self.base_cyclic_client.RefreshFeedback()
            
            pose = np.array([
                feedback.base.tool_pose_x,      # meters
                feedback.base.tool_pose_y,      # meters  
                feedback.base.tool_pose_z,      # meters
                feedback.base.tool_pose_theta_x, # degrees
                feedback.base.tool_pose_theta_y, # degrees
                feedback.base.tool_pose_theta_z  # degrees
            ])
            
            return pose
        
        except Exception as e:
            print(f"ERROR getting tool pose: {e}")
            return np.array([])
    
    def set_single_joint_position(self, joint_id, position):
        """
        Set position of a single joint
        
        Args:
            joint_id: ID of the joint (0 to actuator_count-1)
            position: desired position (in degrees)
        """
        if joint_id < 0 or joint_id >= self.actuator_count.count:
            print(f"ERROR: Invalid joint ID {joint_id}. Valid range: 0 to {self.actuator_count.count-1}")
            return False
        
        print(f"Setting joint {joint_id} to position {position} degrees")
        
        # Get current positions
        current_positions = self.get_joint_positions()
        if len(current_positions) == 0:
            return False
        
        # Update single joint
        current_positions[joint_id] = position
        
        # Set all joint positions
        return self.set_joint_positions(current_positions)
    
    def get_single_joint_position(self, joint_id):
        """
        Get position of a single joint
        
        Args:
            joint_id: ID of the joint (0 to actuator_count-1)
            
        Returns:
            current position of the joint (in degrees)
        """
        if joint_id < 0 or joint_id >= self.actuator_count.count:
            print(f"ERROR: Invalid joint ID {joint_id}. Valid range: 0 to {self.actuator_count.count-1}")
            return None
        
        positions = self.get_joint_positions()
        if len(positions) == 0:
            return None
        
        return positions[joint_id]
    
    def get_gripper_open_pos(self):
        return self.gripper_open_pos
    
    def get_gripper_close_pos(self):
        return self.gripper_close_pos
    
    def set_gripper_position(self, position):
        """
        Set gripper position
        
        """
        try:
            position = float(position)
        except:
            print("invalid gripper position"); return False
        
        # Clamp position between actual gripper limits
        if self.gripper_open_pos==None or self.gripper_close_pos==None:
            position = max(0.0, min(1.0, position))
        else:
            lo, hi = sorted([self.gripper_close_pos, self.gripper_open_pos])
            position = max(lo, min(hi, position))
        
        print(f"Setting gripper position to {position}")
        print(f"(Range: {self.gripper_close_pos} to {self.gripper_open_pos})")
        
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        finger.finger_identifier = 1
        
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        finger.value = position
        
        self.base_client.SendGripperCommand(gripper_command)
        print("Gripper position command sent")
        
    def get_gripper_position(self):
        """
        Get current gripper position
        
        Returns:
            current gripper position (0.0 = fully open, 1.0 = fully closed)
        """
        try:
            gripper_request = Base_pb2.GripperRequest()
            gripper_request.mode = Base_pb2.GRIPPER_POSITION
            
            gripper_measure = self.base_client.GetMeasuredGripperMovement(gripper_request)
            
            if len(gripper_measure.finger):
                return gripper_measure.finger[0].value
            else:
                print("ERROR: No gripper feedback available")
                return None
                
        except Exception as e:
            print(f"ERROR getting gripper position: {e}")
            return None
    
    def close(self):
        """Close the robot connection"""
        print("Closing robot connection...")
        if getattr(self, "in_velocity_mode", False):
            try:
                self.exit_velocity_control()
            except Exception as e:
                print(f"[WARN] exit_velocity_control failed: {e}")


        if hasattr(self, 'in_low_level_mode') and self.in_low_level_mode:
            self.exit_low_level_mode()
        
        try:
            if hasattr(self, 'tcp_session_manager'):
                router_options = RouterClientSendOptions()
                router_options.timeout_ms = 1000
                self.tcp_session_manager.CloseSession(router_options)
                print("Session for tcp closed")
        except Exception as e:
            print(f"Error closing session for tcp: {e}")

        try:
            if hasattr(self, 'udp_session_manager'):
                router_options = RouterClientSendOptions()
                router_options.timeout_ms = 1000
                self.udp_session_manager.CloseSession(router_options)
                print("Session for udp closed")
        except Exception as e:
            print(f"Error closing session for udp: {e}")
        
        try:
            if hasattr(self, 'tcp_transport'):
                self.tcp_transport.disconnect()
                print("TCP transport disconnected")
        except Exception as e:
            print(f"Error disconnecting TCP transport: {e}")
        
        try:
            if hasattr(self, 'udp_transport'):
                self.udp_transport.disconnect() 
                print("UDP transport disconnected")
        except Exception as e:
            print(f"Error disconnecting UDP transport: {e}")
        
        print("Robot connection closed successfully")
    
    def __del__(self):
        """Destructor - ensure connections are closed"""
        self.close()


# Example usage
if __name__ == "__main__":
    try:
        # Create robot controller
        robot = KortexRobotController()
        
        # Home the robot
        robot.home_robot()
        
        # Home the gripper
        robot.home_gripper()
        
        # Get current positions
        positions = robot.get_joint_positions()
        print(f"Current joint positions: {positions}")
        
        # Get tool pose
        pose = robot.get_tool_pose()
        print(f"Current tool pose: {pose}")
        
        # Test gripper
        print("Testing gripper...")
        robot.set_gripper_position(0.5)  # Half closed
        time.sleep(2)
        gripper_pos = robot.get_gripper_position()
        print(f"Gripper position: {gripper_pos}")
        
        # Test single joint movement
        print("Testing single joint movement...")
        robot.set_single_joint_position(0, 10.0)  # Move first joint 10 degrees
        time.sleep(2)
        joint_pos = robot.get_single_joint_position(0)
        print(f"Joint 0 position: {joint_pos}")
        
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'robot' in locals():
            robot.close()
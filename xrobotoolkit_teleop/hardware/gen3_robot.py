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

from kortex_api.autogen.messages import Session_pb2, Base_pb2, BaseCyclic_pb2
from kortex_api.Exceptions.KServerException import KServerException

# 连接参数
ROBOT_IP = "169.254.105.199"
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
        self.gripper_open_pos=None
        self.gripper_close_pos=None
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
                positions.append(actuator.position)
            
            return np.array(positions)
        
        except Exception as e:
            print(f"ERROR getting joint positions: {e}")
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
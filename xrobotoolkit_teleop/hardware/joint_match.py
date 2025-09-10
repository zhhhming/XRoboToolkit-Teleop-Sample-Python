#!/usr/bin/env python3

import time
import webbrowser
import placo
import numpy as np
import json
import os
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)
from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController


class InteractiveJointMatcher:
    def __init__(self, urdf_path, config_file="joint_mapping.json"):
        """
        Interactive joint matcher for mapping robot joints to Placo model joints
        
        Args:
            urdf_path: Path to the URDF file
            config_file: JSON file to save/load joint mapping configuration
        """
        print("=" * 60)
        print("Interactive Joint Matcher")
        print("=" * 60)
        
        self.urdf_path = urdf_path
        self.config_file = config_file
        
        # Initialize robot controller
        print("Initializing robot controller...")
        self.controller = KortexRobotController()
        
        # Initialize Placo robot
        print("Loading Placo robot model...")
        self.placo_robot = placo.RobotWrapper(urdf_path)
        
        # Setup visualization
        print("Setting up visualization...")
        self.placo_viz = robot_viz(self.placo_robot)
        time.sleep(0.5)
        meshcat_url = self.placo_viz.viewer.url()
        print(f"Opening visualization at: {meshcat_url}")
        webbrowser.open(meshcat_url)
        
        # Initialize state
        self.initial_placo_q = self.placo_robot.state.q.copy()
        self.initial_robot_pos = None
        
        # Joint mapping: placo_joint_id -> robot_joint_id
        self.joint_mapping = {}
        self.joint_ranges = {}  # Store tested ranges for each joint
        self.joint_directions = {}  # Store direction info (1 or -1)
        
        # Robot info
        self.num_placo_joints = len(self.placo_robot.state.q)
        self.num_robot_joints = self.controller.actuator_count.count
        
        print(f"Placo model has {self.num_placo_joints} joints")
        print(f"Robot has {self.num_robot_joints} joints")
        
        # Display joint names
        print("\nPlaco joint names:")
        for i, name in enumerate(self.placo_robot.model.names):
            print(f"  {i}: {name}")
        
        print(f"\nRobot joint IDs: 0 to {self.num_robot_joints - 1}")
        
        # Home robot and display initial positions
        self._initialize_robot()
        
        # Load existing mapping if available
        self._load_mapping()
        
        print("\n" + "=" * 60)
        print("Ready! Type 'help' for available commands")
        print("=" * 60)

    def _initialize_robot(self):
        """Initialize robot to home position"""
        print("\nHoming robot...")
        if not self.controller.home_robot():
            print("WARNING: Failed to home robot")
        
        print("Homing gripper...")
        self.controller.home_gripper()
        
        # Get initial positions
        self.initial_robot_pos = self.controller.get_joint_positions()
        print(f"Robot home position: {self.initial_robot_pos}")
        
        # Display initial Placo state
        self.placo_viz.display(self.placo_robot.state.q)

    def _save_mapping(self):
        """Save current joint mapping to file"""
        config = {
            "joint_mapping": self.joint_mapping,
            "joint_ranges": self.joint_ranges,
            "joint_directions": self.joint_directions,
            "num_placo_joints": self.num_placo_joints,
            "num_robot_joints": self.num_robot_joints,
            "urdf_path": self.urdf_path
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            print(f"Mapping saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving mapping: {e}")

    def _load_mapping(self):
        """Load joint mapping from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Convert string keys back to integers
                self.joint_mapping = {int(k): int(v) for k, v in config.get("joint_mapping", {}).items()}
                self.joint_ranges = {int(k): v for k, v in config.get("joint_ranges", {}).items()}
                self.joint_directions = {int(k): v for k, v in config.get("joint_directions", {}).items()}
                
                print(f"Loaded existing mapping from {self.config_file}")
                self._display_current_mapping()
                
            except Exception as e:
                print(f"Error loading mapping: {e}")
        else:
            print("No existing mapping file found")

    def _display_current_mapping(self):
        """Display current joint mapping"""
        print("\nCurrent Joint Mapping:")
        print("Placo Joint -> Robot Joint (Direction, Range)")
        print("-" * 45)
        
        for placo_id in range(self.num_placo_joints):
            if placo_id in self.joint_mapping:
                robot_id = self.joint_mapping[placo_id]
                direction = self.joint_directions.get(placo_id, "unknown")
                range_info = self.joint_ranges.get(placo_id, "unknown")
                print(f"  {placo_id:2d} -> {robot_id:2d} (dir: {direction:2}, range: {range_info})")
            else:
                print(f"  {placo_id:2d} -> NOT MAPPED")

    def _reset_to_home(self):
        """Reset both robot and Placo to home positions"""
        print("Resetting to home positions...")
        
        # Reset Placo
        self.placo_robot.state.q = self.initial_placo_q.copy()
        self.placo_robot.update_kinematics()
        self.placo_viz.display(self.placo_robot.state.q)
        
        # Reset robot
        if self.initial_robot_pos is not None:
            self.controller.set_joint_positions(self.initial_robot_pos)
            time.sleep(2)  # Wait for movement to complete

    def test_placo_joint(self, placo_joint_id, angle_deg=20, direction=1):
        """
        Test a specific Placo joint by moving it and observing the visualization
        
        Args:
            placo_joint_id: ID of the Placo joint to test
            angle_deg: Angle in degrees to move
            direction: 1 for positive, -1 for negative direction
        """
        if placo_joint_id < 0 or placo_joint_id >= self.num_placo_joints:
            print(f"Invalid Placo joint ID. Must be 0-{self.num_placo_joints-1}")
            return
        
        # Reset to home first
        self._reset_to_home()
        time.sleep(1)
        
        # Move Placo joint
        angle_rad = np.deg2rad(angle_deg * direction)
        self.placo_robot.state.q[placo_joint_id] += angle_rad
        self.placo_robot.update_kinematics()
        self.placo_viz.display(self.placo_robot.state.q)
        
        print(f"Moved Placo joint {placo_joint_id} by {angle_deg * direction}° ({angle_rad:.3f} rad)")
        print("Observe the visualization and note which part moved")

    def test_robot_joint(self, robot_joint_id, angle_deg=20, direction=1):
        """
        Test a specific robot joint by moving it
        
        Args:
            robot_joint_id: ID of the robot joint to test
            angle_deg: Angle in degrees to move
            direction: 1 for positive, -1 for negative direction
        """
        if robot_joint_id < 0 or robot_joint_id >= self.num_robot_joints:
            print(f"Invalid robot joint ID. Must be 0-{self.num_robot_joints-1}")
            return
        
        # Get current position
        current_pos = self.controller.get_joint_positions()
        
        # Move specific joint
        new_pos = current_pos.copy()
        new_pos[robot_joint_id] += angle_deg * direction
        
        print(f"Moving robot joint {robot_joint_id} by {angle_deg * direction}°")
        print(f"From {current_pos[robot_joint_id]:.1f}° to {new_pos[robot_joint_id]:.1f}°")
        
        self.controller.set_joint_positions(new_pos)

    def match_joints(self, placo_joint_id, robot_joint_id, direction=1):
        """
        Create a mapping between a Placo joint and a robot joint
        
        Args:
            placo_joint_id: ID of the Placo joint
            robot_joint_id: ID of the robot joint
            direction: 1 for same direction, -1 for opposite direction
        """
        if placo_joint_id < 0 or placo_joint_id >= self.num_placo_joints:
            print(f"Invalid Placo joint ID. Must be 0-{self.num_placo_joints-1}")
            return
        
        if robot_joint_id < 0 or robot_joint_id >= self.num_robot_joints:
            print(f"Invalid robot joint ID. Must be 0-{self.num_robot_joints-1}")
            return
        
        self.joint_mapping[placo_joint_id] = robot_joint_id
        self.joint_directions[placo_joint_id] = direction
        
        print(f"Mapped Placo joint {placo_joint_id} -> Robot joint {robot_joint_id} (direction: {direction})")

    def test_mapping(self, test_angle=15):
        """
        Test the current joint mapping by moving all mapped joints simultaneously
        
        Args:
            test_angle: Angle in degrees to test with
        """
        if not self.joint_mapping:
            print("No joint mapping defined yet!")
            return
        
        print(f"Testing mapping with {test_angle}° movements...")
        
        # Reset to home
        self._reset_to_home()
        time.sleep(2)
        
        # Prepare movements
        placo_q = self.initial_placo_q.copy()
        robot_pos = self.initial_robot_pos.copy()
        
        # Apply test movements based on mapping
        for placo_id, robot_id in self.joint_mapping.items():
            direction = self.joint_directions.get(placo_id, 1)
            
            # Move Placo joint
            placo_q[placo_id] += np.deg2rad(test_angle)
            
            # Move corresponding robot joint
            robot_pos[robot_id] += test_angle * direction
        
        # Execute movements
        print("Moving Placo model...")
        self.placo_robot.state.q = placo_q
        self.placo_robot.update_kinematics()
        self.placo_viz.display(self.placo_robot.state.q)
        
        print("Moving robot...")
        self.controller.set_joint_positions(robot_pos)
        
        print("Both should move in sync! Check if they match.")

    def measure_joint_range(self, robot_joint_id, step_size=10, max_range=180):
        """
        Measure the range of motion for a robot joint
        
        Args:
            robot_joint_id: ID of the robot joint to test
            step_size: Step size in degrees
            max_range: Maximum range to test in each direction
        """
        if robot_joint_id < 0 or robot_joint_id >= self.num_robot_joints:
            print(f"Invalid robot joint ID. Must be 0-{self.num_robot_joints-1}")
            return
        
        print(f"Measuring range for robot joint {robot_joint_id}")
        print("This will move the joint to find its limits. Press Ctrl+C to stop if needed.")
        
        # Start from home position
        self._reset_to_home()
        time.sleep(2)
        
        home_pos = self.controller.get_joint_positions()[robot_joint_id]
        current_pos = self.controller.get_joint_positions()
        
        # Test positive direction
        print("Testing positive direction...")
        max_positive = home_pos
        for angle in range(step_size, max_range + 1, step_size):
            target_pos = current_pos.copy()
            target_pos[robot_joint_id] = home_pos + angle
            
            try:
                self.controller.set_joint_positions(target_pos)
                time.sleep(1)
                
                # Check if movement was successful
                actual_pos = self.controller.get_joint_positions()[robot_joint_id]
                if abs(actual_pos - (home_pos + angle)) > 5:  # 5 degree tolerance
                    print(f"Reached limit at +{angle - step_size}°")
                    break
                max_positive = home_pos + angle
                print(f"  +{angle}° OK")
                
            except Exception as e:
                print(f"Error at +{angle}°: {e}")
                break
        
        # Return to home
        current_pos[robot_joint_id] = home_pos
        self.controller.set_joint_positions(current_pos)
        time.sleep(2)
        
        # Test negative direction
        print("Testing negative direction...")
        max_negative = home_pos
        for angle in range(step_size, max_range + 1, step_size):
            target_pos = current_pos.copy()
            target_pos[robot_joint_id] = home_pos - angle
            
            try:
                self.controller.set_joint_positions(target_pos)
                time.sleep(1)
                
                # Check if movement was successful
                actual_pos = self.controller.get_joint_positions()[robot_joint_id]
                if abs(actual_pos - (home_pos - angle)) > 5:  # 5 degree tolerance
                    print(f"Reached limit at -{angle - step_size}°")
                    break
                max_negative = home_pos - angle
                print(f"  -{angle}° OK")
                
            except Exception as e:
                print(f"Error at -{angle}°: {e}")
                break
        
        # Store range information
        range_info = f"[{max_negative:.1f}, {max_positive:.1f}]"
        
        # Find corresponding Placo joint
        placo_joint_id = None
        for p_id, r_id in self.joint_mapping.items():
            if r_id == robot_joint_id:
                placo_joint_id = p_id
                break
        
        if placo_joint_id is not None:
            self.joint_ranges[placo_joint_id] = range_info
        
        print(f"Range for robot joint {robot_joint_id}: {range_info}")
        
        # Return to home
        self._reset_to_home()

    def run_interactive_mode(self):
        """Run the interactive command-line interface"""
        print("\nStarting interactive mode...")
        
        while True:
            try:
                command = input("\njoint_matcher> ").strip().lower()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0]
                
                if cmd in ['quit', 'exit', 'q']:
                    break
                    
                elif cmd == 'help' or cmd == 'h':
                    self._show_help()
                    
                elif cmd == 'status' or cmd == 's':
                    self._display_current_mapping()
                    
                elif cmd == 'reset' or cmd == 'r':
                    self._reset_to_home()
                    
                elif cmd == 'save':
                    self._save_mapping()
                    
                elif cmd == 'load':
                    self._load_mapping()
                    
                elif cmd == 'tp':  # test placo
                    if len(parts) >= 2:
                        placo_id = int(parts[1])
                        angle = int(parts[2]) if len(parts) > 2 else 20
                        direction = int(parts[3]) if len(parts) > 3 else 1
                        self.test_placo_joint(placo_id, angle, direction)
                    else:
                        print("Usage: tp <placo_joint_id> [angle] [direction]")
                        
                elif cmd == 'tr':  # test robot
                    if len(parts) >= 2:
                        robot_id = int(parts[1])
                        angle = int(parts[2]) if len(parts) > 2 else 20
                        direction = int(parts[3]) if len(parts) > 3 else 1
                        self.test_robot_joint(robot_id, angle, direction)
                    else:
                        print("Usage: tr <robot_joint_id> [angle] [direction]")
                        
                elif cmd == 'map' or cmd == 'm':
                    if len(parts) >= 3:
                        placo_id = int(parts[1])
                        robot_id = int(parts[2])
                        direction = int(parts[3]) if len(parts) > 3 else 1
                        self.match_joints(placo_id, robot_id, direction)
                    else:
                        print("Usage: map <placo_joint_id> <robot_joint_id> [direction]")
                        
                elif cmd == 'test' or cmd == 't':
                    angle = int(parts[1]) if len(parts) > 1 else 15
                    self.test_mapping(angle)
                    
                elif cmd == 'range':
                    if len(parts) >= 2:
                        robot_id = int(parts[1])
                        step = int(parts[2]) if len(parts) > 2 else 10
                        max_range = int(parts[3]) if len(parts) > 3 else 180
                        self.measure_joint_range(robot_id, step, max_range)
                    else:
                        print("Usage: range <robot_joint_id> [step_size] [max_range]")
                        
                elif cmd == 'clear':
                    self.joint_mapping.clear()
                    self.joint_directions.clear()
                    self.joint_ranges.clear()
                    print("Cleared all mappings")
                    
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")

    def _show_help(self):
        """Display help information"""
        help_text = """
Available Commands:
==================

Basic Commands:
  help, h          - Show this help message
  status, s        - Show current joint mapping
  reset, r         - Reset robot and Placo to home positions
  save             - Save current mapping to file
  load             - Load mapping from file
  clear            - Clear all mappings
  quit, exit, q    - Exit the program

Testing Commands:
  tp <id> [angle] [dir]     - Test Placo joint (dir: 1 or -1)
  tr <id> [angle] [dir]     - Test robot joint (dir: 1 or -1)
  map <p_id> <r_id> [dir]   - Map Placo joint to robot joint
  test [angle]              - Test current mapping with all joints
  range <r_id> [step] [max] - Measure robot joint range

Examples:
  tp 0              - Move Placo joint 0 by 20°
  tr 1 30 -1        - Move robot joint 1 by -30°
  map 0 1 1         - Map Placo joint 0 to robot joint 1 (same direction)
  map 2 3 -1        - Map Placo joint 2 to robot joint 3 (opposite direction)
  test 10           - Test mapping with 10° movements
  range 0 5 90      - Measure robot joint 0 range (5° steps, max 90°)

Workflow:
1. Use 'tp <id>' to test each Placo joint and see which part moves
2. Use 'tr <id>' to test each robot joint and see which part moves
3. Use 'map <p_id> <r_id>' to create mappings between joints
4. Use 'test' to verify the mapping works correctly
5. Use 'range <id>' to measure joint limits
6. Use 'save' to save your configuration
"""
        print(help_text)

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'controller'):
                self.controller.close()
        except:
            pass


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python joint_match.py <urdf_path>")
        print("Example: python joint_match.py /path/to/robot.urdf")
        sys.exit(1)
    
    urdf_path = sys.argv[1]
    
    try:
        matcher = InteractiveJointMatcher(urdf_path)
        matcher.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Goodbye!")
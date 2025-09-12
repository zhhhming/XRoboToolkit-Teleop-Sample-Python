#!/usr/bin/env python3

import numpy as np
import threading
import time
from collections import deque
from typing import Optional, Tuple, List
from ruckig import InputParameter, OutputParameter, Ruckig, Result

class RuckigTrajectoryPlanner:
    """
    Ruckig-based trajectory planner for smooth robot motion control.
    Handles real-time target updates with separate waypoint management.
    """
    
    def __init__(
        self,
        dof: int = 7,
        max_velocity: List[float] = None,
        max_acceleration: List[float] = None,
        max_jerk: List[float] = None,
        control_cycle: float = 0.001,  # 1ms for low-level control
        waypoint_buffer_size: int = 30,
        velocity_filter_tau: float = 0.05,  # Time constant for velocity filtering
        simulation_mode: bool = False,  # If True, use computed values as feedback
    ):
        """
        Initialize the Ruckig trajectory planner.
        
        Args:
            dof: Degrees of freedom (number of joints)
            max_velocity: Maximum velocity for each joint (deg/s)
            max_acceleration: Maximum acceleration for each joint (deg/s^2)
            max_jerk: Maximum jerk for each joint (deg/s^3)
            control_cycle: Control cycle time in seconds
            waypoint_buffer_size: Maximum number of waypoints to keep in buffer
            velocity_filter_tau: Time constant for exponential moving average filter
            simulation_mode: If True, use predicted values as actual feedback
        """
        self.dof = dof
        self.control_cycle = control_cycle
        self.waypoint_buffer_size = waypoint_buffer_size
        self.velocity_filter_tau = velocity_filter_tau
        self.simulation_mode = simulation_mode
        
        # Set default limits if not provided
        if max_velocity is None:
            max_velocity = [50.0] * dof  # 50 deg/s default
        if max_acceleration is None:
            max_acceleration = [100.0] * dof  # 100 deg/s^2 default  
        if max_jerk is None:
            max_jerk = [500.0] * dof  # 500 deg/s^3 default
            
        self.max_velocity = np.array(max_velocity)
        self.max_acceleration = np.array(max_acceleration)
        self.max_jerk = np.array(max_jerk)
        
        # Initialize Ruckig
        self.otg = Ruckig(dof, control_cycle)
        self.input_param = InputParameter(dof)
        self.output_param = OutputParameter(dof)
        
        # Set constraints
        self.input_param.max_velocity = self.max_velocity.tolist()
        self.input_param.max_acceleration = self.max_acceleration.tolist()
        self.input_param.max_jerk = self.max_jerk.tolist()
        
        # Waypoint management with thread safety
        self.waypoint_queue = deque(maxlen=waypoint_buffer_size)
        self.waypoint_lock = threading.Lock()
        
        # State tracking - current and previous states for estimation
        self.current_position = np.zeros(dof)
        self.current_velocity = np.zeros(dof)
        self.current_acceleration = np.zeros(dof)
        
        # Previous states for velocity/acceleration estimation
        self.prev_position = np.zeros(dof)
        self.prev_velocity = np.zeros(dof)
        self.last_update_time = time.time()
        
        # Filtered target velocity for smooth pass-through
        self.filtered_target_velocity = np.zeros(dof)
        
        # Simulation mode state (used when not connected to real robot)
        self.sim_position = np.zeros(dof)
        self.sim_velocity = np.zeros(dof)
        self.sim_acceleration = np.zeros(dof)
        
        # Statistics
        self.waypoints_enqueued = 0
        self.trajectory_steps = 0
        
    def update_constraints(
        self,
        max_velocity: Optional[List[float]] = None,
        max_acceleration: Optional[List[float]] = None,
        max_jerk: Optional[List[float]] = None
    ):
        """Update motion constraints dynamically."""
        if max_velocity is not None:
            self.max_velocity = np.array(max_velocity)
            self.input_param.max_velocity = max_velocity
        if max_acceleration is not None:
            self.max_acceleration = np.array(max_acceleration)
            self.input_param.max_acceleration = max_acceleration
        if max_jerk is not None:
            self.max_jerk = np.array(max_jerk)
            self.input_param.max_jerk = max_jerk
    
    def add_waypoint(self, target_position: np.ndarray):
        """
        Add a new waypoint to the trajectory queue.
        This should be called from a separate high-frequency thread.
        
        Args:
            target_position: Target joint positions (degrees)
        """
        with self.waypoint_lock:
            waypoint = {
                'position': target_position.copy(),
                'timestamp': time.time()
            }
            self.waypoint_queue.append(waypoint)
            self.waypoints_enqueued += 1
    
    def get_latest_waypoint(self) -> Optional[np.ndarray]:
        """Get the most recent waypoint from the queue."""
        with self.waypoint_lock:
            if not self.waypoint_queue:
                return None
            # Always return the newest waypoint for best tracking
            return self.waypoint_queue[-1]['position']
    
    def estimate_target_velocity(self) -> np.ndarray:
        """
        Estimate target velocity based on waypoint history.
        Uses the trajectory of recent waypoints to predict desired velocity.
        """
        with self.waypoint_lock:
            if len(self.waypoint_queue) < 2:
                return np.zeros(self.dof)
            
            # Use a window of recent waypoints (e.g., last 10-20)
            window_size = min(20, len(self.waypoint_queue))
            
            # Get first and last waypoints in window
            oldest_wp = self.waypoint_queue[-window_size]
            newest_wp = self.waypoint_queue[-1]
            
            # Calculate time difference
        dt = newest_wp['timestamp'] - oldest_wp['timestamp']
            
        if dt < 0.001:  # Avoid division by very small numbers
            return self.filtered_target_velocity
            
        # Calculate instantaneous velocity
        position_diff = newest_wp['position'] - oldest_wp['position']
        inst_velocity = position_diff / dt
            
        # Apply exponential moving average filter
        # v_filtered(k) = (1-β)*v_filtered(k-1) + β*v_inst(k)
        # β = 1 - exp(-Δt/τ)
        beta = 1.0 - np.exp(-self.control_cycle / self.velocity_filter_tau)
        self.filtered_target_velocity = (
            (1 - beta) * self.filtered_target_velocity + 
             beta * inst_velocity
        )
            
        # Clamp to velocity limits
        self.filtered_target_velocity = np.clip(
            self.filtered_target_velocity,
            -self.max_velocity * 0.5,  # Use 50% of max for target velocity
            self.max_velocity * 0.5
        )
        threshold = 0.5  # deg/s，可以按实际需要调整
        self.filtered_target_velocity[np.abs(self.filtered_target_velocity) < threshold] = 0.0
            
        return self.filtered_target_velocity
    
    def update_current_state(
        self,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None
    ):
        """
        Update the current robot state with proper state tracking.
        
        Args:
            position: Current joint positions (degrees)
            velocity: Current joint velocities (deg/s), if available
            acceleration: Current joint accelerations (deg/s^2), if available
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # Save previous states before updating
        self.prev_position = self.current_position.copy()
        self.prev_velocity = self.current_velocity.copy()
        
        # Update position
        self.current_position = position.copy()
        
        # Update or estimate velocity
        if velocity is not None:
            self.current_velocity = velocity.copy()
        else:
            # Estimate velocity from position change
            if dt > 0.001:  # Avoid division by very small dt
                self.current_velocity = (self.current_position - self.prev_position) / dt
            else:
                # Keep previous velocity if dt is too small
                self.current_velocity = self.prev_velocity.copy()
        
        # Update or estimate acceleration
        if acceleration is not None:
            self.current_acceleration = acceleration.copy()
        else:
            # Estimate acceleration from velocity change
            if dt > 0.001:
                self.current_acceleration = (self.current_velocity - self.prev_velocity) / dt
            else:
                self.current_acceleration = self.current_acceleration.copy()
        
        self.last_update_time = current_time
    
    def compute_trajectory_step(
        self,
        current_position: np.ndarray,
        current_velocity: Optional[np.ndarray] = None,
        current_acceleration: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Compute one step of the trajectory.
        
        Args:
            current_position: Current joint positions (degrees)
            current_velocity: Current joint velocities (deg/s)
            current_acceleration: Current joint accelerations (deg/s^2)
            
        Returns:
            (target_velocity, target_position, reached_target)
        """
        # In simulation mode, use simulated feedback
        if self.simulation_mode:
            current_position = self.sim_position.copy()
            current_velocity = self.sim_velocity.copy()
            current_acceleration = self.sim_acceleration.copy()
        
        # Update current state with proper tracking
        self.update_current_state(current_position, current_velocity, current_acceleration)
        
        # Get latest waypoint
        target_position = self.get_latest_waypoint()
        
        if target_position is None:
            # No waypoint, maintain current position with zero velocity
            return np.zeros(self.dof), self.current_position, True
        
        # Set current state for Ruckig
        self.input_param.current_position = self.current_position.tolist()
        self.input_param.current_velocity = self.current_velocity.tolist()
        self.input_param.current_acceleration = self.current_acceleration.tolist()
        
        # Set target position
        self.input_param.target_position = target_position.tolist()
        
        # Estimate target velocity based on waypoint history for smooth pass-through
        target_velocity = self.estimate_target_velocity()
        self.input_param.target_velocity = target_velocity.tolist()
        
        # Target acceleration is typically zero
        self.input_param.target_acceleration = [0.0] * self.dof
        
        # Perform trajectory calculation
        result = self.otg.update(self.input_param, self.output_param)
        
        # Check if we reached the target
        reached = result == Result.Finished
        
        # Extract new state from output
        new_position = np.array(self.output_param.new_position)
        new_velocity = np.array(self.output_param.new_velocity)
        new_acceleration = np.array(self.output_param.new_acceleration)
        
        # In simulation mode, update simulated state
        if self.simulation_mode:
            self.sim_position = new_position.copy()
            self.sim_velocity = new_velocity.copy()
            self.sim_acceleration = new_acceleration.copy()
        
        self.trajectory_steps += 1
        
        return new_velocity, new_position, reached
    
    def reset(self):
        """Reset the trajectory planner."""
        with self.waypoint_lock:
            self.waypoint_queue.clear()
        
        self.current_velocity = np.zeros(self.dof)
        self.current_acceleration = np.zeros(self.dof)
        self.prev_position = self.current_position.copy()
        self.prev_velocity = np.zeros(self.dof)
        self.filtered_target_velocity = np.zeros(self.dof)
        
        # Reset simulation state
        self.sim_position = self.current_position.copy()
        self.sim_velocity = np.zeros(self.dof)
        self.sim_acceleration = np.zeros(self.dof)
        
        # Reset Ruckig
        self.otg = Ruckig(self.dof, self.control_cycle)
        
        self.trajectory_steps = 0
    
    def set_simulation_mode(self, enabled: bool):
        """Enable or disable simulation mode."""
        self.simulation_mode = enabled
        if enabled:
            print("Simulation mode enabled - using predicted values as feedback")
            # Initialize simulation state with current state
            self.sim_position = self.current_position.copy()
            self.sim_velocity = self.current_velocity.copy()
            self.sim_acceleration = self.current_acceleration.copy()
        else:
            print("Simulation mode disabled - using real robot feedback")
    
    def get_status(self) -> dict:
        """Get current planner status."""
        with self.waypoint_lock:
            num_waypoints = len(self.waypoint_queue)
            
        return {
            'num_waypoints': num_waypoints,
            'current_velocity_norm': np.linalg.norm(self.current_velocity),
            'current_acceleration_norm': np.linalg.norm(self.current_acceleration),
            'filtered_target_velocity_norm': np.linalg.norm(self.filtered_target_velocity),
            'waypoints_processed': self.waypoints_enqueued,
            'trajectory_steps': self.trajectory_steps,
            'simulation_mode': self.simulation_mode
        }


class RuckigControlInterface:
    """
    Interface to integrate Ruckig planner with the robot controller.
    Manages waypoint collection and trajectory execution in separate threads.
    """
    
    def __init__(
        self,
        robot_controller,
        planner: RuckigTrajectoryPlanner,
        control_rate_hz: float = 1000.0,  # 1kHz for trajectory execution
        waypoint_rate_hz: float = 100.0,   # 100Hz for waypoint collection
    ):
        """
        Initialize the control interface.
        
        Args:
            robot_controller: The KortexRobotController instance
            planner: The RuckigTrajectoryPlanner instance
            control_rate_hz: Control loop frequency in Hz
            waypoint_rate_hz: Waypoint collection frequency in Hz
        """
        self.robot = robot_controller
        self.planner = planner
        self.control_rate_hz = control_rate_hz
        self.waypoint_rate_hz = waypoint_rate_hz
        self.control_dt = 1.0 / control_rate_hz
        self.waypoint_dt = 1.0 / waypoint_rate_hz
        
        # Thread control
        self._stop_event = threading.Event()
        self._control_thread = None
        self._waypoint_thread = None
        
        # Shared state for IK targets
        self._target_position_lock = threading.Lock()
        self._latest_ik_target = None
        
        # Performance tracking
        self.control_loop_times = deque(maxlen=100)
        self.waypoint_loop_times = deque(maxlen=100)
        
    def set_ik_target(self, target_position: np.ndarray):
        """
        Called by IK thread to set new target position.
        """
        with self._target_position_lock:
            self._latest_ik_target = target_position.copy()
    
    def _waypoint_collection_loop(self):
        """Separate thread for collecting waypoints from IK at higher frequency."""
        print(f"Starting waypoint collection loop at {self.waypoint_rate_hz}Hz...")
        
        last_target = None
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            try:
                # Check for new IK target
                with self._target_position_lock:
                    if self._latest_ik_target is not None:
                        new_target = self._latest_ik_target.copy()
                        self._latest_ik_target = None
                    else:
                        new_target = None
                
                # Add waypoint if we have a new target
                if new_target is not None:
                    # Only add if significantly different from last target
                    if last_target is None or np.linalg.norm(new_target - last_target) > 0.05:
                        self.planner.add_waypoint(new_target)
                        last_target = new_target
                
            except Exception as e:
                print(f"Error in waypoint collection: {e}")
            
            # Maintain loop rate
            elapsed = time.time() - loop_start
            self.waypoint_loop_times.append(elapsed)
            
            sleep_time = self.waypoint_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("Waypoint collection loop stopped")
    
    def _control_loop(self):
        """Main control loop for trajectory execution."""
        print(f"Starting Ruckig control loop at {self.control_rate_hz}Hz...")
        
        # Ensure robot is in low-level mode
        if not self.planner.simulation_mode and not self.robot.in_low_level_mode:
            self.robot.enter_low_level_mode()
        
        status_print_interval = 2.0  # Print status every 2 seconds
        last_status_time = time.time()
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            try:
                # Get current robot state
                if not self.planner.simulation_mode:
                    current_pos = self.robot.get_joint_positions()
                    if len(current_pos) < 7:
                        print("Warning: Invalid joint position reading")
                        time.sleep(self.control_dt)
                        continue
                else:
                    # In simulation mode, use the planner's simulated position
                    current_pos = self.planner.sim_position
                
                # Compute trajectory step
                target_velocity, target_position, reached = self.planner.compute_trajectory_step(
                    current_pos
                )
                
                # Send commands to robot (only if not in simulation mode)
                if not self.planner.simulation_mode:
                    # Send velocity command via UDP
                    result = self.robot.set_joint_positions_udp(
                        target_position,  # For position tracking
                        kp=0.0,  # Pure velocity control
                        vel_cap=np.max(np.abs(target_velocity)),
                        tol=0.5
                    )
                
                # Print status periodically
                current_time = time.time()
                if current_time - last_status_time > status_print_interval:
                    status = self.planner.get_status()
                    avg_control_time = np.mean(list(self.control_loop_times)) * 1000 if self.control_loop_times else 0
                    avg_waypoint_time = np.mean(list(self.waypoint_loop_times)) * 1000 if self.waypoint_loop_times else 0
                    
                    print(f"Status: WP={status['num_waypoints']}, "
                          f"Vel={status['current_velocity_norm']:.1f}°/s, "
                          f"TargetVel={status['filtered_target_velocity_norm']:.1f}°/s, "
                          f"Steps={status['trajectory_steps']}, "
                          f"Control={avg_control_time:.1f}ms, "
                          f"WP_collect={avg_waypoint_time:.1f}ms")
                    
                    last_status_time = current_time
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                import traceback
                traceback.print_exc()
            
            # Track loop timing
            elapsed = time.time() - loop_start
            self.control_loop_times.append(elapsed)
            
            # Maintain control rate
            sleep_time = self.control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif elapsed > self.control_dt * 2:
                print(f"Warning: Control loop overrun ({elapsed*1000:.1f}ms)")
        
        # Exit low-level mode when stopping (if not in simulation)
        if not self.planner.simulation_mode and self.robot.in_low_level_mode:
            self.robot.exit_low_level_mode()
        
        print("Ruckig control loop stopped")
    
    def start(self):
        """Start both waypoint collection and control threads."""
        if self._control_thread is not None and self._control_thread.is_alive():
            print("Control threads already running")
            return
        
        self._stop_event.clear()
        
        # Start waypoint collection thread
        self._waypoint_thread = threading.Thread(
            target=self._waypoint_collection_loop,
            name="WaypointCollectionThread"
        )
        self._waypoint_thread.daemon = True
        self._waypoint_thread.start()
        
        # Start control thread
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="RuckigControlThread"
        )
        self._control_thread.daemon = True
        self._control_thread.start()
        
        print("Ruckig control interface started")
    
    def stop(self):
        """Stop all threads."""
        print("Stopping Ruckig control interface...")
        self._stop_event.set()
        
        # Wait for threads to stop
        for thread, name in [(self._waypoint_thread, "Waypoint"), 
                             (self._control_thread, "Control")]:
            if thread is not None:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    print(f"Warning: {name} thread did not stop gracefully")
        
        print("Ruckig control interface stopped")
    
    def test_without_robot(self, duration: float = 10.0):
        """
        Test the planner in simulation mode without robot hardware.
        
        Args:
            duration: Test duration in seconds
        """
        print(f"Testing in simulation mode for {duration} seconds...")
        
        # Enable simulation mode
        self.planner.set_simulation_mode(True)
        
        # Start threads
        self.start()
        
        # Generate test waypoints
        start_time = time.time()
        waypoint_interval = 0.5  # New waypoint every 500ms
        last_waypoint_time = start_time
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # Generate new waypoint periodically
                if current_time - last_waypoint_time >= waypoint_interval:
                    # Generate sinusoidal test pattern
                    t = current_time - start_time
                    test_position = np.array([
                        20 * np.sin(0.5 * t),
                        15 * np.cos(0.3 * t),
                        10 * np.sin(0.7 * t),
                        0, 0, 0, 0
                    ])
                    self.set_ik_target(test_position)
                    last_waypoint_time = current_time
                    print(f"Test waypoint at t={t:.1f}s: [{test_position[0]:.1f}, {test_position[1]:.1f}, {test_position[2]:.1f}, ...]")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Test interrupted")
        
        # Stop and report
        self.stop()
        status = self.planner.get_status()
        print(f"\nTest completed:")
        print(f"  Waypoints processed: {status['waypoints_processed']}")
        print(f"  Trajectory steps: {status['trajectory_steps']}")
        print(f"  Final velocity: {status['current_velocity_norm']:.2f}°/s")
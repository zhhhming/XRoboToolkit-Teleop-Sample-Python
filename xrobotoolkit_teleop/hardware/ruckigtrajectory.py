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
        beta = 1.0 - 0.9
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
    #set的时候要顺便能够设置一下当前位置
    def set_simulation_mode(self, enabled: bool, sim_position=None):
        """Enable or disable simulation mode."""
        self.simulation_mode = enabled
        print("Simulation mode enabled - using predicted values as feedback")

        # 1) 解析/推断 sim_position
        import numpy as np
        dof = int(getattr(self, "dof", 7))

        if sim_position is None:
            base_pos = getattr(self, "current_position", None)
            if base_pos is None or np.size(base_pos) < dof:
                base_pos = np.zeros(dof, dtype=float)
            else:
                base_pos = np.asarray(base_pos, dtype=float).reshape(-1)[:dof]
        else:
            base_pos = np.asarray(sim_position, dtype=float).reshape(-1)
            if base_pos.size < dof:
                raise ValueError(f"sim_position length {base_pos.size} < dof {dof}")
            base_pos = base_pos[:dof]

        self.sim_position = base_pos.copy()
        self.sim_velocity = np.zeros(dof, dtype=float)
        self.sim_acceleration = np.zeros(dof, dtype=float)
    
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

    
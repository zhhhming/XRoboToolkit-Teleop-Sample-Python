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
        control_cycle: float = 0.003,  # 1ms for low-level control
        waypoint_buffer_size: int = 30,
        velocity_filter_tau: float = 0.05,  # Time constant for velocity filtering
        simulation_mode: bool = False,  # If True, use computed values as feedback
        enable_waypoint_filter: bool = True,
        waypoint_filter_alpha: float = 0.15,  # 滤波强度 (0-1), 越小越平滑
        waypoint_filter_cutoff_hz: float = None,  # 可选：用截止频率计算alpha
        waypoint_filter_deadband: float = 0.01,  # 死区：变化太小时不更新滤波器
        waypoint_blend_beta: float = 1.0,  # 输出waypoint与当前位置的融合系数
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
            enable_waypoint_filter: Whether to enable low-pass filtering on waypoints
            waypoint_blend_beta: Blend factor between filtered waypoint and
            current joint position (0-1, closer to 1 favors the waypoint)
        """
        self.dof = dof
        self.control_cycle = control_cycle
        self.waypoint_buffer_size = waypoint_buffer_size
        self.velocity_filter_tau = velocity_filter_tau
        self.simulation_mode = simulation_mode

         # 低通滤波器配置
        self.enable_waypoint_filter = enable_waypoint_filter
        self.waypoint_filter_deadband = waypoint_filter_deadband
        self.waypoint_blend_beta = float(np.clip(waypoint_blend_beta, 0.0, 1.0))

        # 计算滤波系数
        if waypoint_filter_cutoff_hz is not None:
            # 从截止频率计算alpha: α = 1 - exp(-2π * fc * dt)
            dt_filter = 1.0 / 1000.0  # 假设1kHz调用频率
            self.waypoint_filter_alpha = 1.0 - np.exp(-2.0 * np.pi * waypoint_filter_cutoff_hz * dt_filter)
            self.waypoint_filter_alpha = np.clip(self.waypoint_filter_alpha, 0.01, 0.99)
        else:
            self.waypoint_filter_alpha = waypoint_filter_alpha
        
         # 低通滤波器状态变量
        self.filtered_waypoint = None  # 当前滤波输出
        self.last_raw_waypoint = None  # 上一次原始输入
        self.waypoint_filter_initialized = False
        self.waypoint_filter_lock = threading.Lock()

        # Set default limits if not provided
        if max_velocity is None:
            max_velocity = [50.0] * dof  # 50 deg/s default
        if max_acceleration is None:
            max_acceleration = [1000.0] * dof  # 100 deg/s^2 default  
        if max_jerk is None:
            max_jerk = [1000.0] * dof  # 500 deg/s^3 default
            
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

    def normalize_angle_deg(self, angle_deg):
        """将任意角度归一化到 [-180, 180] 范围"""
        angle = angle_deg % 360
        if angle > 180:
            angle -= 360
        return angle

    def to_nearest_equivalent_angle(self, target_deg, current_deg):
        """
        将目标角度调整为离当前角度最近的等效角度
        例如：current=10, target=350 -> 返回 -10 (而不是 350)
        """
        diff = target_deg - current_deg
        # 归一化差值到 [-180, 180]
        diff = self.normalize_angle_deg(diff)
        # 返回最近的等效角度
        return current_deg + diff
    
    def _apply_waypoint_filter(self, raw_waypoint: np.ndarray) -> np.ndarray:
        """
        对原始waypoint应用低通滤波
        
        Args:
            raw_waypoint: 原始目标位置
            
        Returns:
            filtered_waypoint: 滤波后的目标位置
        """
        if not self.enable_waypoint_filter:
            return raw_waypoint.copy()
            
        with self.waypoint_filter_lock:
            # 首次初始化
            if not self.waypoint_filter_initialized:
                self.filtered_waypoint = raw_waypoint.copy()
                self.last_raw_waypoint = raw_waypoint.copy()
                self.waypoint_filter_initialized = True
                print(f"[WAYPOINT_FILTER] Initialized with: {self.filtered_waypoint}")
                return self.filtered_waypoint.copy()
            
            # 计算变化量（考虑角度连续性）
            if self.filtered_waypoint is not None:
                adjusted_target = np.array([
                    self.to_nearest_equivalent_angle(raw_waypoint[i], self.filtered_waypoint[i])
                    for i in range(self.dof)
                ])
            else:
                adjusted_target = raw_waypoint.copy()
            
            # 计算变化量
            change = adjusted_target - self.filtered_waypoint
            max_change = np.max(np.abs(change))
            
            # 死区处理：变化太小时不更新
            if max_change < self.waypoint_filter_deadband:
                return self.filtered_waypoint.copy()
            
            # 应用指数移动平均滤波
            # filtered = (1-α) * filtered_old + α * raw
            alpha = self.waypoint_filter_alpha
            self.filtered_waypoint = (1.0 - alpha) * self.filtered_waypoint + alpha * adjusted_target
            
            # 记录输入
            self.last_raw_waypoint = raw_waypoint.copy()
            
            return self.filtered_waypoint.copy()
    
    def set_waypoint_filter_params(
        self, 
        alpha: Optional[float] = None,
        cutoff_hz: Optional[float] = None,
        deadband: Optional[float] = None,
        enabled: Optional[bool] = None,
        blend_beta: Optional[float] = None
    ):
        """
        动态调整滤波器参数
        
        Args:
            alpha: 滤波系数
            cutoff_hz: 截止频率 (Hz)
            deadband: 死区阈值
            enabled: 是否启用滤波
            blend_beta: waypoint与当前关节位置融合系数
        """
        with self.waypoint_filter_lock:
            if enabled is not None:
                self.enable_waypoint_filter = bool(enabled)
                
            if deadband is not None:
                self.waypoint_filter_deadband = deadband
                
            if cutoff_hz is not None:
                dt_filter = 1.0 / 1000.0  # 1kHz
                self.waypoint_filter_alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt_filter)
                self.waypoint_filter_alpha = np.clip(self.waypoint_filter_alpha, 0.01, 0.99)
                print(f"[WAYPOINT_FILTER] Set cutoff to {cutoff_hz}Hz, alpha={self.waypoint_filter_alpha:.3f}")
            elif alpha is not None:
                self.waypoint_filter_alpha = np.clip(alpha, 0.01, 0.99)
                print(f"[WAYPOINT_FILTER] Set alpha to {self.waypoint_filter_alpha:.3f}")
            if blend_beta is not None:
                self.waypoint_blend_beta = float(np.clip(blend_beta, 0.0, 1.0))

    def reset_waypoint_filter(self, initial_position: Optional[np.ndarray] = None):
        """
        重置滤波器状态
        
        Args:
            initial_position: 初始位置，如果为None则使用当前filtered_waypoint
        """
        with self.waypoint_filter_lock:
            if initial_position is not None:
                self.filtered_waypoint = initial_position.copy()
                self.last_raw_waypoint = initial_position.copy()
                self.waypoint_filter_initialized = True
                print(f"[WAYPOINT_FILTER] Reset to position: {initial_position}")
            else:
                self.waypoint_filter_initialized = False
                self.filtered_waypoint = None
                self.last_raw_waypoint = None
                print("[WAYPOINT_FILTER] Reset to uninitialized state")

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
    
    def get_latest_waypoint(self, current_position: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Get the most recent waypoint from the queue with optional low-pass filtering.
        
        Returns:
            Filtered target position, or None if no waypoints available
        """
        with self.waypoint_lock:
            if not self.waypoint_queue:
                return None
            # Get the newest raw waypoint
            raw_waypoint = self.waypoint_queue[-1]['position']
        
        # Apply low-pass filter
        filtered_waypoint = self._apply_waypoint_filter(raw_waypoint)
        beta = self.waypoint_blend_beta
        if beta < 1.0:
            if current_position is None:
                return filtered_waypoint.copy()
            else:
                current = np.array(current_position, dtype=float)
            filtered_waypoint = beta * filtered_waypoint + (1.0 - beta) * current

        return filtered_waypoint.copy()
    
    def estimate_target_velocity(self) -> np.ndarray:#waypoint线程要快一点，500吧
        """
        Estimate target velocity based on waypoint history.
        Uses the trajectory of recent waypoints to predict desired velocity.
        """
        cur_pos = self.current_position.copy()
        with self.waypoint_lock:
            if len(self.waypoint_queue) < 2:
                return np.zeros(self.dof)
            
            # Use a window of recent waypoints (e.g., last 10-20)
            window_size = min(5, len(self.waypoint_queue))
            
            # Get first and last waypoints in window
            oldest_wp = self.waypoint_queue[-window_size]
            newest_wp = self.waypoint_queue[-1]
            
            # Calculate time difference
            dt = newest_wp['timestamp'] - oldest_wp['timestamp']
            
            if dt < 0.001:  # Avoid division by very small numbers
                print("[ESTIMATE]时间间隔太小")
                return self.filtered_target_velocity
                
            # Calculate instantaneous velocity
            position_diff = newest_wp['position'] - oldest_wp['position']
            inst_velocity = position_diff / dt
            adjusted_target = np.array([
                self.to_nearest_equivalent_angle(newest_wp['position'][i], cur_pos[i])
                for i in range(self.dof)
            ])
            desired_dir = adjusted_target - cur_pos
            # 小距离死区：距离很小时（例如 < 0.5°）认为不需要速度
            dist_deadband = 0.1
            near_mask = np.abs(desired_dir) < dist_deadband
            desired_dir_sign = np.sign(desired_dir)  # -1, 0, +1
        inst_velocity = np.asarray(inst_velocity, dtype=float)
        inst_velocity = np.nan_to_num(inst_velocity, nan=0.0, posinf=0.0, neginf=0.0)
        # Apply exponential moving average filter
        # v_filtered(k) = (1-β)*v_filtered(k-1) + β*v_inst(k)
        # β = 1 - exp(-Δt/τ)
        beta = 0.3
        self.filtered_target_velocity = (
            (1 - beta) * self.filtered_target_velocity + 
             beta * inst_velocity
        )
        v = self.filtered_target_velocity.copy()
        # 不一致：sign(v) 与 desired_dir_sign 不同，且距离不在近目标死区之外
        sign_mismatch = (np.sign(v) * desired_dir_sign) < 0
        v[sign_mismatch] = 0.0
        v[near_mask] = 0.0
        # Clamp to velocity limits
        v = np.clip(v, -self.max_velocity, self.max_velocity)
        threshold = 0.05  # deg/s，可以按实际需要调整
        v[np.abs(v) < threshold] = 0.0
        self.filtered_target_velocity = v
        print(f"[VELOCITY DEBUG] position_error: {np.max(np.abs(desired_dir)):.3f}°")
        print(f"[VELOCITY DEBUG] inst_velocity: {np.max(np.abs(inst_velocity)):.3f}°/s") 
        print(f"[VELOCITY DEBUG] filtered_velocity: {np.max(np.abs(v)):.3f}°/s")
    
        
        # return self.filtered_target_velocity
        return np.array([0.0]*7,dtype=float)
    
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
    def normalize_angle_deg(self, angle_deg):
        """将任意角度归一化到 [-180, 180] 范围"""
        angle = angle_deg % 360
        if angle > 180:
            angle -= 360
        return angle

    def to_nearest_equivalent_angle(self, target_deg, current_deg):
        """
        将目标角度调整为离当前角度最近的等效角度
        例如：current=10, target=350 -> 返回 -10 (而不是 350)
        """
        diff = target_deg - current_deg
        # 归一化差值到 [-180, 180]
        diff = self.normalize_angle_deg(diff)
        # 返回最近的等效角度
        return current_deg + diff
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
                    (target_velocity, target_position, reached_target, ok)
        ok=False 表示本周期 Ruckig 解算失败（需要上层 fallback）
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
            return np.zeros(self.dof), self.current_position, True, True
        adjusted_target = np.array([
            self.to_nearest_equivalent_angle(target_position[i], self.current_position[i])
            for i in range(self.dof)
        ])
        print(f"[COMPUTE STEP]adjusted_target:{adjusted_target}")
        print(f"[COMPUTE STEP]current_position:{current_position}")
        # Set current state for Ruckig
        self.input_param.current_position = self.current_position.tolist()
        self.input_param.current_velocity = self.current_velocity.tolist()
        self.input_param.current_acceleration = self.current_acceleration.tolist()
        
        # Set target position
        self.input_param.target_position = adjusted_target.tolist()
        
        # Estimate target velocity based on waypoint history for smooth pass-through
        target_velocity = self.estimate_target_velocity()
        self.input_param.target_velocity = target_velocity.tolist()
        print(f"[COMPUTE]target_velocity:{target_velocity}")
        
        # Target acceleration is typically zero
        self.input_param.target_acceleration = [0.0] * self.dof
        
        # Perform trajectory calculation
        try:
            result = self.otg.update(self.input_param, self.output_param)
        except Exception as e:
            # 求解抛异常：ok=False，交给上层 fallback
            # 用当前速度当作“保持”更安全
            print(f"[Ruckig ERROR] Exception in update: {repr(e)}")
            hold_vel = self.current_velocity.copy()
            return hold_vel, self.current_position, False, False

        
        
        if result in (Result.Working, Result.Finished):
            new_position     = np.array(self.output_param.new_position)
            new_velocity     = np.array(self.output_param.new_velocity)
            new_acceleration = np.array(self.output_param.new_acceleration)

            if self.simulation_mode:
                self.sim_position     = new_position.copy()
                self.sim_velocity     = new_velocity.copy()
                self.sim_acceleration = new_acceleration.copy()

            self.trajectory_steps += 1
            print(f"[COMPUTER] new velocity:{new_velocity}")
            return new_velocity, new_position, (result == Result.Finished), True
        else:
            # Ruckig 报错：ok=False，上层 fallback
            hold_vel = self.current_velocity.copy()
            return hold_vel, self.current_position, False, False
        
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

    
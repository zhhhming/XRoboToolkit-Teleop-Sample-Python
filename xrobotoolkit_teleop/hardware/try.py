#!/usr/bin/env python3
import time
import threading
import webbrowser
import numpy as np
import placo
import meshcat.transformations as tf

from placo_utils.visualization import (frame_viz, robot_frame_viz, robot_viz)
from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController

# =========================
# 基本参数
# =========================
URDF = "/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-7DOF.urdf"
EE_LINK = "bracelet_link"
CTRL_HZ = 200.0
CTRL_DT = 1.0 / CTRL_HZ
TARGET_UPDATE_HZ = 1.0             # 每秒更新一次 target
TARGET_STEP_M = 0.01                # 每次 +1cm（沿 x 轴）
SOLVER_LAMBDA = True                # solver.solve(True) 走带夹持/正则步
VEL_KP = 2.0                        # 低层比例（deg/s per deg）
VEL_CAP = 0.6                       # 低层速度上限（deg/s）
POS_TOL = 0.5                       # 到位阈值（deg）

# =========================
# 初始化 Placo 模型与可视化
# =========================
placo_robot = placo.RobotWrapper(URDF)
print("Joint names in the Placo model:")
for i, joint_name in enumerate(placo_robot.model.names):
    print(f"  {i}: {joint_name}")

print("Initializing Placo visualization...")
placo_vis = robot_viz(placo_robot)
webbrowser.open(placo_vis.viewer.url())
placo_vis.display(placo_robot.state.q)

# 关节名到“机器人关节序号”的映射（按你的代码）
joint_name_to_robot_index = {
    "joint_1": 0, "joint_2": 1, "joint_3": 2,
    "joint_4": 3, "joint_5": 4, "joint_6": 5, "joint_7": 6,
}

# =========================
# 一些工具函数（保持你的风格）
# =========================
def _idxq_nq(joint_name: str):
    jid = placo_robot.model.getJointId(joint_name)
    j = placo_robot.model.joints[jid]
    return j.idx_q, j.nq

def _read_joint_rad(joint_name: str) -> float:
    i0, nq = _idxq_nq(joint_name)
    return float(placo_robot.state.q[i0])

def _write_joint_rad(joint_name: str, value_rad: float) -> None:
    i0, nq = _idxq_nq(joint_name)
    q = placo_robot.state.q.copy()
    q[i0] = float(value_rad)
    placo_robot.state.q = q

def _placo_to_robot_deg_vector() -> np.ndarray:
    if not joint_name_to_robot_index:
        raise RuntimeError("joint_name_to_robot_index is empty")
    n = max(joint_name_to_robot_index.values()) + 1
    robot_deg = np.full(n, np.nan, dtype=float)
    for name, idx in joint_name_to_robot_index.items():
        v_rad = _read_joint_rad(name)
        robot_deg[idx] = np.degrees(v_rad)
    if np.isnan(robot_deg).any():
        missing = np.where(np.isnan(robot_deg))[0].tolist()
        raise RuntimeError(f"Failed to read joint values for indices: {missing}")
    return robot_deg

def _robot_deg_to_placo(robot_deg: np.ndarray):
    if robot_deg is None or len(robot_deg) < 7:
        raise ValueError("robot_deg must have 7 elements.")
    print(f"placo_get_degree:{robot_deg}")
    for name, idx in joint_name_to_robot_index.items():
        _write_joint_rad(name, np.radians(float(robot_deg[idx])))
        print(f"{name}get{np.radians(float(robot_deg[idx]))}")

def _get_link_pose(link_name: str):
    T_world_link = placo_robot.get_T_world_frame(link_name)
    pos = T_world_link[:3, 3]
    quat = tf.quaternion_from_matrix(T_world_link)
    return pos, quat
def normalize_angle_deg(angle_deg):
        """将任意角度归一化到 [-180, 180] 范围"""
        angle = angle_deg % 360
        if angle > 180:
            angle -= 360
        return angle

def to_nearest_equivalent_angle(target_deg, current_deg):
    """
    将目标角度调整为离当前角度最近的等效角度
    例如：current=10, target=350 -> 返回 -10 (而不是 350)
    """
    diff = target_deg - current_deg
    # 归一化差值到 [-180, 180]
    diff =self.normalize_angle_deg(diff)
    # 返回最近的等效角度
    return current_deg + diff

# =========================
# 连接实机 & 初始对齐
# =========================
robot = KortexRobotController()
solver = placo.KinematicsSolver(placo_robot)
solver.dt = CTRL_DT
solver.mask_fbase(True)
solver.add_kinetic_energy_regularization_task(1e-6)
robot_positions=robot.get_joint_positions()
robot_positions_normalized = np.array([normalize_angle_deg(angle) for angle in robot_positions])

# 从实机读取当前关节 → 同步到 Placo
_robot_deg_to_placo(robot_positions_normalized)

placo_robot.update_kinematics()


# =========================
# 主流程：启动线程，等待 Ctrl+C 退出
# =========================
if __name__ == "__main__":  
    while True:
        robot_positions=robot.get_joint_positions()
        robot_positions_normalized = np.array([normalize_angle_deg(angle) for angle in robot_positions])

        # 从实机读取当前关节 → 同步到 Placo
        _robot_deg_to_placo(robot_positions_normalized)
        print(f"postion:{robot_positions_normalized}")
        placo_vis.display(placo_robot.state.q.copy())
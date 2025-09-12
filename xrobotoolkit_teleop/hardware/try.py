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

# =========================
# 连接实机 & 初始对齐
# =========================
robot = KortexRobotController()
robot.home_robot()  # 可按需保留/去掉
deg=robot.get_joint_positions()
deg[0]-=70
robot.set_joint_positions(deg)
time.sleep(5)
solver = placo.KinematicsSolver(placo_robot)
solver.dt = CTRL_DT
solver.mask_fbase(True)
solver.add_kinetic_energy_regularization_task(1e-6)

# 从实机读取当前关节 → 同步到 Placo
_robot_deg_to_placo(robot.get_joint_positions())
placo_robot.update_kinematics()

# 基于当前 EE 姿态建立初始目标（先 +1cm）
ee_xyz, ee_quat = _get_link_pose(EE_LINK)
print(f"ee_xyz:{ee_xyz}  ee_quat:{ee_quat}")
target_T = tf.quaternion_matrix(ee_quat)
print(f"target_T:{target_T}")
target_T[:3, 3] = ee_xyz + np.array([TARGET_STEP_M, 0.0, 0.0])  # 初次加 1cm
print(f"target_T:{target_T}     ee_xyz:{ee_xyz + np.array([TARGET_STEP_M, 0.0, 0.0]) }")


# 创建末端任务 & 可选任务
effector = solver.add_frame_task(EE_LINK, target_T)
effector.configure("gen3", "soft", 1.0)
manipulability = solver.add_manipulability_task(EE_LINK, "both", 1.0)
manipulability.configure("manipulability", "soft", 1e-2)

# 切低层
robot.enter_low_level_mode()
robot.enter_velocity_control_mode()

# 共享状态：目标与同步锁
target_lock = threading.Lock()
stop_event = threading.Event()

# =========================
# 目标更新线程：每 1s 沿 x 轴 +1cm
# =========================
def target_update_thread():
    global target_T
    try:
        while not stop_event.is_set():
            with target_lock:
                T = target_T.copy()
                T[:3, 3] += np.array([TARGET_STEP_M, 0.0, 0.0])  # 沿 x 轴前移
                target_T = T
            time.sleep(1.0 / TARGET_UPDATE_HZ)
    except Exception as e:
        print(f"[TARGET] thread error: {e}")

# =========================
# 控制线程：50Hz 解 IK & 低层发命令
# =========================
def control_thread():
    last_joint_cmd_deg = None
    try:
        while not stop_event.is_set():
            t0 = time.time()

            # 1) 读实机 → 同步 Placo 状态
            robot_deg = np.asarray(robot.get_joint_positions(), dtype=float)
            if robot_deg.size < 7:
                print("[WARN] get_joint_positions() 数量不足")
                time.sleep(CTRL_DT); continue
            _robot_deg_to_placo(robot_deg)
            placo_robot.update_kinematics()

            # 2) 读取当前 target（加锁）
            with target_lock:
                effector.T_world_frame = target_T.copy()

            # 3) 解 IK
            try:
                solver.solve(SOLVER_LAMBDA)
            except RuntimeError as e:
                print(f"[IK] solver failed: {e}")
                time.sleep(CTRL_DT); continue

            placo_robot.update_kinematics()

            # 4) 关节发送（低层：位置=反馈，速度=KP*误差 限幅）
            try:
                joint_cmd_deg = _placo_to_robot_deg_vector()

                # NaN 防护
                if np.isnan(joint_cmd_deg).any():
                    if last_joint_cmd_deg is not None and not np.isnan(last_joint_cmd_deg).any():
                        print("[WARN] NaN in joint target, fallback to last valid cmd.")
                        joint_cmd_deg = last_joint_cmd_deg.copy()
                    else:
                        print("[WARN] NaN and no fallback, skip sending.")
                        time.sleep(CTRL_DT); continue

                robot.set_joint_positions_udp(
                    joint_cmd_deg.tolist(),
                )
                last_joint_cmd_deg = joint_cmd_deg.copy()

            except Exception as e:
                print(f"[CTRL] send command failed: {e}")

            # 5) 维持 50Hz
            dt = time.time() - t0
            if (sleep := (CTRL_DT - dt)) > 0:
                time.sleep(sleep)
    except Exception as e:
        print(f"[CTRL] thread error: {e}")

# =========================
# 主流程：启动线程，等待 Ctrl+C 退出
# =========================
if __name__ == "__main__":
    print("\n--- Start: target updater @1Hz, controller @50Hz ---")
    th_target = threading.Thread(target=target_update_thread, daemon=True)
    th_control = threading.Thread(target=control_thread, daemon=True)
    th_target.start()
    th_control.start()

    try:
        while True:
            # 可视化（低频即可）
            placo_vis.display(placo_robot.state.q)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[MAIN] Stopping...")
        stop_event.set()
        th_target.join(timeout=1.0)
        th_control.join(timeout=1.0)
        # 可选：发一帧零速度（安全停车）
        try:
            cur = robot.get_joint_positions()
            robot.set_joint_positions_udp(cur, kp=0.0, vel_cap=0.0, tol=POS_TOL)
        except Exception:
            pass
        print("[MAIN] Done.")

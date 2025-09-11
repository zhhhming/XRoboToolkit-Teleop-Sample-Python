import time
import webbrowser
import placo
import meshcat.transformations as tf
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)
from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController
robot_urdf_path="/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-7DOF.urdf"
placo_robot = placo.RobotWrapper(robot_urdf_path)

print("Joint names in the Placo model:")
for i, joint_name in enumerate(placo_robot.model.names):
    print(f"  {i}: {joint_name}")
print("Initializing Placo visualization...")
print(placo_robot.state.q[7:])
placo_vis = robot_viz(placo_robot)
webbrowser.open(placo_vis.viewer.url())
placo_vis.display(placo_robot.state.q)
joint_name_to_robot_index = {
                "joint_1": 0, 
                "joint_2": 1, 
                "joint_3": 2,
                "joint_4": 3, 
                "joint_5": 4, 
                "joint_6": 5, 
                "joint_7": 6,
            }

import numpy as np

# 你自己定义的标准顺序（按 URDF 里的命名）
PLAC0_JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"]

# Kortex 端的目标顺序（joint_identifier）。常见是 0..6 对应 joint_1..joint_7
KORTEX_ORDER = [0,1,2,3,4,5,6]   # 如果你的机器人编号反了，就改成 [6,5,4,3,2,1,0] 等

def _idxq_nq(joint_name: str):
    """返回该关节在 q 中的起始 idx 和维度 nq"""
    jid = placo_robot.model.getJointId(joint_name)
    j = placo_robot.model.joints[jid]
    return j.idx_q, j.nq

def _read_joint_rad(joint_name: str) -> float | None:
    """从 state.q 精确读取某个转动关节弧度值"""
    i0, nq = _idxq_nq(joint_name)
    return float(placo_robot.state.q[i0])

def _write_joint_rad(joint_name: str, value_rad: float) -> bool:
    """把标量弧度写回 state.q 的正确槽位；非标量关节直接跳过"""
    i0, nq = _idxq_nq(joint_name)
    q = placo_robot.state.q.copy()
    q[i0] = float(value_rad)
    placo_robot.state.q = q
    return True

def _placo_to_robot_deg_vector() -> np.ndarray:
    """
    从 Placo 精确读取 7 个关节转robot 索引放入数组。
    """
    # 计算目标长度
    if not joint_name_to_robot_index:
        raise RuntimeError("joint_name_to_robot_index is empty")
    n = max(joint_name_to_robot_index.values()) + 1

    robot_deg = np.full(n, np.nan, dtype=float)  # 先用 NaN 填充，便于发现缺失

    for name, idx in joint_name_to_robot_index.items():
        v_rad = _read_joint_rad(name) 
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


def _robot_deg_to_placo( robot_deg: np.ndarray):
    """
    把机器人测得的 7 个关节角写回 Placo 的 state.q（rad）。
    """
    if robot_deg is None or len(robot_deg) < 7:
        raise ValueError("robot_deg must have 7 elements.")
    for name, idx in joint_name_to_robot_index.items():
        _write_joint_rad(name, np.radians(float(robot_deg[idx])))
        print(f"joint:{name} get {np.radians(float(robot_deg[idx]))}")
def debug_print_joint_layout(model):
    print("\n[DEBUG] Joint layout (idx_q / nq):")
    for jid, name in enumerate(model.names):
        j = model.joints[jid]
        print(f"  {jid:2d} {name:25s} idx_q={j.idx_q:2d} nq={j.nq}")
def _get_link_pose(link_name: str):
    """Get current world pose for a given link name from Placo"""
    T_world_link = placo_robot.get_T_world_frame(link_name)
    pos = T_world_link[:3, 3]
    quat = tf.quaternion_from_matrix(T_world_link)
    return pos, quat

debug_print_joint_layout(placo_robot.model)
robot=KortexRobotController()
robot.home_robot()
solver = placo.KinematicsSolver(placo_robot)
solver.dt = 1.0 / 50.0
solver.mask_fbase(True)
solver.add_kinetic_energy_regularization_task(1e-6)
ee_xyz, ee_quat = _get_link_pose("bracelet_link")
ee_target = tf.quaternion_matrix(ee_quat)
ee_target[:3, 3] = ee_xyz
effector_task={}
effector_task["gen3"] = solver.add_frame_task("bracelet_link", ee_target)
print(f"Created pose task for gen3 ")
effector_task["gen3"].configure("gen3", "soft", 1.0)
manipulability = solver.add_manipulability_task("bracelet_link", "both", 1.0)
manipulability.configure("manipulability", "soft", 1e-2)
_robot_deg_to_placo(robot.get_joint_positions())
placo_robot.update_kinematics()
placo_vis.display(placo_robot.state.q)
robot._set_single_level_servoing()
curren=robot.get_joint_positions()
curren[0]-=70
robot.set_joint_positions(curren)
time.sleep(10)
_robot_deg_to_placo(robot.get_joint_positions())
placo_robot.update_kinematics()
placo_vis.display(placo_robot.state.q)
AXIS = "x"               # "x" 或 "y"
STEP_M = 0.05            # 每步 5 cm
TRAVEL_LIMIT = 0.20      # 往返总幅度 ±0.20 m，可按需要调小
CTRL_HZ = 50.0
DT = 1.0 / CTRL_HZ

offset_val = 0.0         # 累积位移（米）
direction = 1.0          # 方向：+1 / -1
last_joint_cmd_deg = None  # 保护性回退

STEP_DEG =-3.0         # 每次 +1°

last_cmd = None
print("\n--- Start incremental EE translation IK loop ---")
while True:
    loop_start = time.time()

    
    robot_deg = np.asarray(robot.get_joint_positions(), dtype=float)
    if robot_deg.size < 7:
        print("[WARN] get_joint_positions() 返回数量不足，跳过本帧")
        time.sleep(DT)
        continue

    _robot_deg_to_placo(robot_deg)
    placo_robot.update_kinematics()

    # 2) 基于“当前 EE 姿态”做增量目标：每次沿 X/Y 加 5cm
    cur_xyz, cur_quat = _get_link_pose("bracelet_link")

    # 计算本步要增加的位移
    offset_val += direction * STEP_M
    if abs(offset_val) > TRAVEL_LIMIT:
        # 到达幅度上限就反向
        direction *= -1.0
        offset_val = np.clip(offset_val, -TRAVEL_LIMIT, TRAVEL_LIMIT)

    if AXIS.lower() == "x":
        delta = np.array([STEP_M * direction, 0.0, 0.0])
    else:
        delta = np.array([0.0, STEP_M * direction, 0.0])

    target_xyz = cur_xyz + delta
    target_quat = cur_quat  # 不改变姿态，只平移

    target_T = tf.quaternion_matrix(target_quat)
    target_T[:3, 3] = target_xyz

    # 3) 把目标位姿给 IK 的 effector task
    effector_task["gen3"].T_world_frame = target_T
    try:
        solver.solve(True)   # True: 通常表示 clamped / 带正则，避免过大步长
    except RuntimeError as e:
        print(f"[IK] solver failed: {e}")
        time.sleep(DT)
        continue

    placo_robot.update_kinematics()
    placo_vis.display(placo_robot.state.q)
    try:
        joint_cmd_deg = _placo_to_robot_deg_vector()

        # 防 NaN（映射缺失或求解异常时使用上一帧）
        if np.isnan(joint_cmd_deg).any():
            if last_joint_cmd_deg is not None and not np.isnan(last_joint_cmd_deg).any():
                print("[WARN] NaN in joint target, fallback to last valid cmd.")
                joint_cmd_deg = last_joint_cmd_deg.copy()
            else:
                print("[WARN] NaN and no fallback, skip sending this frame.")
                time.sleep(DT)
                continue

        robot.set_joint_positions(joint_cmd_deg.tolist())
        last_joint_cmd_deg = joint_cmd_deg.copy()

        # 可选：打印一行简短调试
        print(f"EE→ {AXIS}+{STEP_M*100:.0f}mm | joints(deg)={np.round(joint_cmd_deg,1)}")

    except Exception as e:
        print(f"[CTRL] send command failed: {e}")

    # 6) 循环节拍
    elapsed = time.time() - loop_start
    sleep_time = DT - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
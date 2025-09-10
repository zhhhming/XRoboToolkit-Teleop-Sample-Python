import webbrowser
import placo
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)
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

import numpy as np

# 你自己定义的标准顺序（按 URDF 里的命名）
PLAC0_JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"]

# Kortex 端的目标顺序（joint_identifier）。常见是 0..6 对应 joint_1..joint_7
KORTEX_ORDER = [0,1,2,3,4,5,6]   # 如果你的机器人编号反了，就改成 [6,5,4,3,2,1,0] 等

def _joint_scalar_from_q(model, q, joint_name):
    """
    在 Pinocchio/placo 模型里，精确取某个关节在 q 向量中的标量位置（支持 free-flyer 情况）
    """
    jid = model.getJointId(joint_name)
    i0  = model.joints[jid].idx_q
    nqj = model.joints[jid].nq
    if nqj != 1:
        raise ValueError(f"{joint_name} has nq={nqj}, expected 1.")
    return float(q[i0])

def _write_joint_scalar_to_q(model, q, joint_name, value):
    """
    把一个标量关节位置写回 q 向量正确的位置
    """
    jid = model.getJointId(joint_name)
    i0  = model.joints[jid].idx_q
    nqj = model.joints[jid].nq
    if nqj != 1:
        raise ValueError(f"{joint_name} has nq={nqj}, expected 1.")
    q[i0] = value

def placo_joints_rad(placo_robot, joint_names=PLAC0_JOINT_NAMES):
    """
    从 placo 的 state.q 中拿到 7 个真实关节的角度（单位：rad；顺序：joint_names）
    无论是否有 free-flyer 都成立
    """
    q = placo_robot.state.q
    return np.array([_joint_scalar_from_q(placo_robot.model, q, name) for name in joint_names], dtype=float)

def placo_to_kortex_deg(placo_robot):
    """
    把 Placo 的 7 个关节（rad）转换成发给 Kortex 的角度数组（deg），
    且按照 KORTEX_ORDER 排好序（默认 0..6）
    """
    q_rad = placo_joints_rad(placo_robot, PLAC0_JOINT_NAMES)     # [j1..j7] rad
    q_deg = np.degrees(q_rad)                                    # 转成 deg
    # 如果 KORTEX_ORDER 与 joint_1..7 的顺序一致，这里就是原样返回
    # 若不一致，可用一个“重排映射”把 q_deg 对应到 Kortex 的编号顺序
    # 这里假设 joint_i -> identifier i-1，故直接返回：
    return q_deg.astype(float)

def kortex_deg_to_placo(placo_robot, kortex_deg):
    """
    把 Kortex 反馈的 7 个关节角（deg，按 0..6）写回 placo 的 state.q（rad）
    """
    if len(kortex_deg) != 7:
        raise ValueError(f"Expect 7 angles from Kortex, got {len(kortex_deg)}")

    # 如果你的 KORTEX_ORDER 与 joint_1..7 一一对应（0->joint_1, 1->joint_2,...）
    # 那就直接按顺序使用；若不一致，需要根据实际映射先重排 kortex_deg
    q_rad = np.radians(np.asarray(kortex_deg, dtype=float))

    q_full = placo_robot.state.q.copy()
    for i, name in enumerate(PLAC0_JOINT_NAMES):
        _write_joint_scalar_to_q(placo_robot.model, q_full, name, q_rad[i])
    placo_robot.state.q = q_full
    return q_rad
def debug_print_joint_layout(model):
    print("\n[DEBUG] Joint layout (idx_q / nq):")
    for jid, name in enumerate(model.names):
        j = model.joints[jid]
        print(f"  {jid:2d} {name:25s} idx_q={j.idx_q:2d} nq={j.nq}")
debug_print_joint_layout(placo_robot.model)
while True:
    pass
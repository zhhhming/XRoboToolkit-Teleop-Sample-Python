# mj_placo_demo.py
# Requirements:
#   pip install mujoco placo numpy
#   (建议也装 mujoco-python-viewer，用于可视化: pip install mujoco-python-viewer)
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

import placo
from placo_utils import tf  # 用来处理4x4变换矩阵

# ==== 路径与配置 ====
MJCF_XML_PATH = r"D:/xrobotics/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/scene_gen3.xml"
URDF_PATH     = r"D:/xrobotics/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/GEN3-6DOF.urdf"

# MJCF 里的 keyframe：可以用名字，也可以用索引。两者二选一，优先用名字。
KEYFRAME_NAME = "home"      # 如果你在 XML 的 <keyframe><key name="home" .../></keyframe> 中定义了名字
KEYFRAME_INDEX = 0          # 如果没名字，就用第 0 个 keyframe

# Placo 里作为末端的 frame（URDF 里的连杆/Frame 名）
EE_FRAME_NAME = "bracelet_link"  # 也可以用 "end_effector_link"

# 给末端一个随便的 Task：在世界坐标系里 z 方向抬高 5cm
EE_TARGET_OFFSET_WORLD = np.array([0.0, 0.0, 0.05])  # meter

# 伺服（位置->力矩）的 PD 参数与停止阈值
KP = 120.0
KD = 4.0
POS_TOL = 1e-3  # rad
MAX_STEPS = 2000


# ==== 一些工具函数 ====
def apply_keyframe(model, data, keyframe_name=None, keyframe_index=0):
    """将数据重置到指定 keyframe。优先按名字找，否则按索引。"""
    key_id = None
    if keyframe_name:
        try:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        except Exception:
            key_id = -1
    if key_id is None or key_id < 0:
        key_id = int(keyframe_index)
    # 应用 keyframe
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)  # 让派生量一致


def get_scalar_joint_addr(model, j_id):
    """返回（qpos_index, qvel_index, dof_size）。只处理标量关节（hinge/slide）。"""
    jtype = model.jnt_type[j_id]
    if jtype == mujoco.mjtJoint.mjJNT_HINGE or jtype == mujoco.mjtJoint.mjJNT_SLIDE:
        qadr = model.jnt_qposadr[j_id]
        vadr = model.jnt_dofadr[j_id]
        return qadr, vadr, 1
    return None, None, 0  # 非标量（如 free/ball）这里不处理


def build_actuator_joint_map(model):
    """
    返回一个列表，每个元素是 dict:
      {"act_id": i, "act_name": name, "j_id": jid, "j_name": jname, "qadr": qadr, "vadr": vadr}
    假设 actuator 与 joint 同名（你的 XML 就是这样）。
    """
    m = []
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        # 通过名字找到对应关节
        try:
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, act_name)
        except Exception:
            j_id = -1
        if j_id >= 0:
            qadr, vadr, dof = get_scalar_joint_addr(model, j_id)
            if dof == 1:
                j_name = act_name
                m.append({
                    "act_id": i, "act_name": act_name,
                    "j_id": j_id, "j_name": j_name,
                    "qadr": qadr, "vadr": vadr
                })
    return m


def clamp(x, lo, hi):
    return float(np.minimum(np.maximum(x, lo), hi))


def ctrl_saturate(model, act_id, u):
    lo, hi = model.actuator_ctrlrange[act_id]
    return clamp(u, lo, hi)


# ==== 主流程 ====
def main():
    # 1) 载入 MuJoCo
    if not os.path.exists(MJCF_XML_PATH):
        raise FileNotFoundError(MJCF_XML_PATH)
    model = mujoco.MjModel.from_xml_path(MJCF_XML_PATH)
    data = mujoco.MjData(model)

    # 应用 keyframe 作为初始状态
    apply_keyframe(model, data, KEYFRAME_NAME, KEYFRAME_INDEX)

    # 做一个 actuator->joint 的映射（只含标量关节）
    mapping = build_actuator_joint_map(model)
    if not mapping:
        raise RuntimeError("未找到任何与关节同名的 actuator，请检查 XML。")

    # 2) 载入 PlaCo（URDF）
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(URDF_PATH)
    robot = placo.RobotWrapper(URDF_PATH)         # 加载 URDF（可以直接传文件路径）
    solver = placo.KinematicsSolver(robot)         # 逆解求解器
    solver.mask_fbase(True)                        # 固定基座（桌面机械臂）

    # 3) 将 MuJoCo 当前关节角同步到 PlaCo
    #    按名字同步，只同步 mapping 中出现的关节
    for item in mapping:
        q_now = float(data.qpos[item["qadr"]])
        try:
            robot.set_joint(item["j_name"], q_now)
        except Exception:
            # URDF 里可能名字不一致，忽略此项
            pass
    robot.update_kinematics()

    # 4) 在 PlaCo 上添加末端位姿任务，并设置一个目标（当前姿态基础上 +z 5cm）
    #    先取当前末端位姿
    T_world_ee = robot.get_T_world_frame(EE_FRAME_NAME)
    # 目标 = 当前 * 平移(0,0,0.05)
    T_target = T_world_ee @ tf.translation_matrix(EE_TARGET_OFFSET_WORLD)

    ee_task = solver.add_frame_task(EE_FRAME_NAME, np.eye(4))
    ee_task.configure("ee_task", "soft", 1.0, 1.0)
    ee_task.T_world_frame = T_target

    # 5) 求解 IK（一次），把解反映到 robot.state.q
    robot.update_kinematics()
    solver.solve(True)      # True = 把解积分进 q
    robot.update_kinematics()

    # 6) 从 PlaCo 读取目标关节角，作为 MuJoCo 位置伺服的参考
    q_target_dict = {}
    for item in mapping:
        try:
            q_target_dict[item["j_name"]] = float(robot.get_joint(item["j_name"]))
        except Exception:
            pass

    # 7) 打开 MuJoCo viewer，循环 PD 伺服直到到位
    print("启动 MuJoCo viewer：按下 ESC 关闭窗口")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        for step in range(MAX_STEPS):
            # 计算每个执行器的 PD 力矩
            total_err2 = 0.0
            for item in mapping:
                j = item["j_name"]
                if j not in q_target_dict:
                    continue
                qdes = q_target_dict[j]
                q    = float(data.qpos[item["qadr"]])
                qd   = float(data.qvel[item["vadr"]])
                err  = qdes - q
                tau  = KP * err - KD * qd
                tau  = ctrl_saturate(model, item["act_id"], tau)
                data.ctrl[item["act_id"]] = tau
                total_err2 += err * err

            # 前进一步
            mujoco.mj_step(model, data)

            # 同步可视化
            if step % 5 == 0:
                viewer.sync()

            # 到位判据（均方根误差）
            rmse = np.sqrt(total_err2 / max(1, len(q_target_dict)))
            if rmse < POS_TOL:
                print(f"到位：RMSE={rmse:.6f} rad, steps={step}, sim_t={data.time:.3f}s, wall_t={time.time()-t0:.2f}s")
                # 到位后让它多跑一会儿稳定一下
                for _ in range(200):
                    mujoco.mj_step(model, data)
                    if _ % 5 == 0:
                        viewer.sync()
                break

        # 关闭前再同步一下
        viewer.sync()


if __name__ == "__main__":
    main()

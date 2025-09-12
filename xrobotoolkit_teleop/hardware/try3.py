#!/usr/bin/env python3
import time
import threading
import numpy as np
from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController

# ===== 用户可调参数 =====
CTRL_HZ = 50.0                   # 控制发送频率
CTRL_DT = 1.0 / CTRL_HZ
TARGET_STEP_DEG = 5.0            # 每秒 +5°
TARGET_UPDATE_HZ = 1.0           # 目标更新频率（1Hz）
VEL_KP = 2.0                     # set_joint_positions_udp 的 kp（deg/s per deg）
VEL_CAP = 1.0                    # 每关节速度上限（deg/s）
POS_TOL = 0.5                    # 到位阈值（deg）
HOMING = True                  # 是否先做一次 home

# ===== 共享状态 =====
target_lock = threading.Lock()
stop_event = threading.Event()
target_joint1_deg = None         # 仅 joint1 的目标角
fixed_pose_deg = None            # 其余 6 个关节的固定角 + joint1 初始角（用于构造整向量）
last_send_deg = None             # 打印/调试

def target_update_thread():
    global target_joint1_deg
    try:
        while not stop_event.is_set():
            with target_lock:
                target_joint1_deg += TARGET_STEP_DEG
                print(f"[TARGET] joint1 target -> {target_joint1_deg:.2f} deg")
            time.sleep(1.0 / TARGET_UPDATE_HZ)
    except Exception as e:
        print(f"[TARGET] thread error: {e}")

def control_thread(robot: KortexRobotController):
    global last_send_deg
    try:
        while not stop_event.is_set():
            t0 = time.time()
            with target_lock:
                # 构造 7 维目标数组：只有 joint1 用递增目标，其余保持初始固定角
                cmd = fixed_pose_deg.copy()
                cmd[0] = target_joint1_deg

            # 发送（低层速度控制：函数内部会用反馈位置 + 速度指令）
            info = robot.set_joint_positions_udp(
                cmd.tolist(), kp=VEL_KP, vel_cap=VEL_CAP, tol=POS_TOL
            )

            # 可选：打印简要信息
            if last_send_deg is None or abs(cmd[0] - last_send_deg) >= 5.0:
                print(f"[CTRL] send joint1={cmd[0]:.2f}° | ok={info.get('ok')} "
                      f"reached={info.get('reached')} max_err={info.get('max_err'):.3f}")
                last_send_deg = cmd[0]

            # 维持 50Hz
            dt = time.time() - t0
            if (sleep := CTRL_DT - dt) > 0:
                time.sleep(sleep)
    except Exception as e:
        print(f"[CTRL] thread error: {e}")

def main():
    global target_joint1_deg, fixed_pose_deg

    robot = KortexRobotController()
    if HOMING:
        robot.home_robot()
    deg=robot.get_joint_positions()
    deg[0]-=70
    robot.set_joint_positions(deg)
    time.sleep(5)
    # 进入低层
    robot.enter_low_level_mode()

    # 读取当前关节角（度）
    cur_deg = np.asarray(robot.get_joint_positions(), dtype=float)
    if cur_deg.size < 7:
        raise RuntimeError(f"Expected 7 joints, got {cur_deg.size}")

    # 初始化共享状态
    fixed_pose_deg = cur_deg.copy()         # 其余关节固定
    target_joint1_deg = float(cur_deg[0])   # joint1 目标从当前角度开始
    print(f"[INIT] joint1 start at {target_joint1_deg:.2f} deg")
    print(f"[INIT] other joints fixed at: {np.round(fixed_pose_deg[1:], 2)} deg")

    # 启动线程
    th_target = threading.Thread(target=target_update_thread, daemon=True)
    th_ctrl = threading.Thread(target=control_thread, args=(robot,), daemon=True)
    th_target.start()
    th_ctrl.start()

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[MAIN] Stopping...")
        stop_event.set()
        th_target.join(timeout=1.0)
        th_ctrl.join(timeout=1.0)
        # 安全停车：发一帧“零速度”  (kp=0, vel_cap=0)
        try:
            robot.set_joint_positions_udp(fixed_pose_deg.tolist(), kp=0.0, vel_cap=0.0, tol=POS_TOL)
        except Exception:
            pass
        print("[MAIN] Done.")

if __name__ == "__main__":
    main()

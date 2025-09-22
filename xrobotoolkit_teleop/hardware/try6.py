from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController
import time
import numpy as np
import signal
import sys
import csv
import matplotlib.pyplot as plt
RATE_HZ = 300.0
DT = 1.0 / RATE_HZ
JOINT_DOF = 7
CSV_PATH = "gen3_log.csv"   # 结束后会导出：t, pos[0..6], vel[0..6]

robot = KortexRobotController()
# 日志缓存
t_log = []
pos_log = []  # (N, 7)
vel_log = []  # (N, 7)

running = True
def handle_sigint(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handle_sigint)
target_pos=robot.get_joint_positions()[2]
print(target_pos)
print("开始循环：按 Ctrl+C 结束并绘图…")
last_time = time.time()
try:
    while running:
        loop_start = time.time()
        
        
        # 读取状态
        position = np.array(robot.get_joint_positions(), dtype=float)  # deg
        speed    = np.array(robot.get_joint_speeds(), dtype=float)     # deg/s

        # 安全检查（有些情况下读取失败可能返回长度不足）
        if position.shape[0] < JOINT_DOF or speed.shape[0] < JOINT_DOF:
            # 轻微等待以维持频率
            time.sleep(DT)
            continue

        # 你的需求：把第 3 关节（index=2）目标设为 19 度

        cmd_pos = position.copy()
        cmd_pos[2]-=0.4
        robot.send_joint_positions_udp(cmd_pos)

        # 记录
        t_now = time.time()
        t_log.append(t_now)
        pos_log.append(position.tolist())
        vel_log.append(speed.tolist())

        # 维持 100 Hz
        elapsed = time.time() - loop_start
        sleep_time = DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

finally:
    # ============== 数据整理与导出 ==============
    if len(t_log) == 0:
        print("无有效数据，直接退出。")
        sys.exit(0)

    t0 = t_log[0]
    t_rel = np.array(t_log) - t0             # 相对时间（秒）
    pos_arr = np.array(pos_log)              # (N, 7)
    vel_arr = np.array(vel_log)              # (N, 7)

    # 导出 CSV
    header = (["t_s"] +
              [f"pos_deg_j{i+1}" for i in range(JOINT_DOF)] +
              [f"vel_degps_j{i+1}" for i in range(JOINT_DOF)])
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(len(t_rel)):
            writer.writerow([t_rel[k]] + pos_arr[k].tolist() + vel_arr[k].tolist())
    print(f"数据已保存：{CSV_PATH}")

    # ============== 绘图：位置曲线 ==============
    plt.figure()
    for j in range(JOINT_DOF):
        plt.plot(t_rel, pos_arr[:, j], label=f"J{j+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (deg)")
    plt.title("Gen3 Joint Positions (deg)")
    plt.legend(loc="best")
    plt.grid(True)

    # ============== 绘图：速度曲线 ==============
    plt.figure()
    for j in range(JOINT_DOF):
        plt.plot(t_rel, vel_arr[:, j], label=f"J{j+1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (deg/s)")
    plt.title("Gen3 Joint Velocities (deg/s)")
    plt.legend(loc="best")
    plt.grid(True)

    plt.show()

#joint1:80deg/s 4deg
#joint2:80deg/s 4deg
#joint3:80deg/s 4deg 0.5deg:20deg/s 0.2:5deg/s 1:60dedg/s
#joint4:80deg/s 4deg
#joint5:150deg/s 5deg
#joint6:150deg/s 5deg
#joint7:140deg/s 5deg

#joint1 >0.1
#joint2 >0.02
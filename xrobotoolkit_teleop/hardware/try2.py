# check_velocity_mode.py
import time
from xrobotoolkit_teleop.hardware.gen3_robot import KortexRobotController

from kortex_api.autogen.messages import Base_pb2, Common_pb2
from kortex_api.autogen.messages import ActuatorConfig_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2

# ---- 1) 连接 & 进入低级模式 ----
robot = KortexRobotController()
print("active_state:", robot.base_client.GetArmState())

# 进入 LOW_LEVEL_SERVOING
smi = Base_pb2.ServoingModeInformation()
smi.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
robot.base_client.SetServoingMode(smi)
print("servoing mode:", robot.base_client.GetServoingMode().servoing_mode)

# ---- 2) 打印可用 ControlMode 枚举 ----
cm_names = [v.name for v in ActuatorConfig_pb2.ControlMode.DESCRIPTOR.values]
print("ActuatorConfig.ControlMode enums:", cm_names)

# 选择速度 / 位置枚举名（不同 SDK 可能叫 VELOCITY vs JOINT_VELOCITY / POSITION vs JOINT_POSITION）
def pick(name_opts):
    for n in name_opts:
        if n in cm_names: return n
    raise RuntimeError(f"None of {name_opts} found in {cm_names}")

VEL_NAME = pick(["VELOCITY", "JOINT_VELOCITY"])
POS_NAME = pick(["POSITION", "JOINT_POSITION"])
vel_mode = ActuatorConfig_pb2.ControlMode.Value(VEL_NAME)
pos_mode = ActuatorConfig_pb2.ControlMode.Value(POS_NAME)

# ---- 3) 用 DeviceManager 拿到每个执行器的 deviceId ----
handles = robot.device_manager_client.ReadAllDevices().device_handle
actuators = [h for h in handles
             if h.device_type == Common_pb2.DeviceTypes.Value("ACTUATOR")]
act_ids = [h.device_identifier for h in actuators]
print("Actuator deviceIds:", act_ids)

# ---- 4) 切到关节速度模式（关键：用 deviceId=xxx 传参）----
info_vel = ActuatorConfig_pb2.ControlModeInformation(control_mode=vel_mode)
for aid in act_ids:
    robot.actuator_config_client.SetControlMode(info_vel, deviceId=int(aid))
print(f"All actuators -> ControlMode.{VEL_NAME}")

# ---- 5) 给关节1发 1s 的小速度（必须同时把 position 填成反馈）----
t_end = time.time() + 1.0
while time.time() < t_end:
    fb = robot.base_cyclic_client.RefreshFeedback()
    cmd = BaseCyclic_pb2.Command()
    cmd.frame_id = (fb.frame_id + 1) & 0xFFFF

    # 组 7 个执行器命令: 位置=反馈，速度=仅对关节1给一点点
    for i in range(len(fb.actuators)):
        a = cmd.actuators.add()
        a.position   = float(fb.actuators[i].position)   # 手册要求：速度/力矩控制时 position 用反馈值
        a.velocity   = 5.0 if i == 0 else 0.0            # 关节1给 5 deg/s，其余 0
        a.command_id = int(fb.actuators[i].command_id) + 1

    robot.base_cyclic_client.Refresh(cmd)
    time.sleep(0.002)  # 约 500 Hz，实验足够

# ---- 6) 切回关节位置模式，收尾 ----
info_pos = ActuatorConfig_pb2.ControlModeInformation(control_mode=pos_mode)
for aid in act_ids:
    robot.actuator_config_client.SetControlMode(info_pos, deviceId=int(aid))
print(f"All actuators -> ControlMode.{POS_NAME}")
print("Done.")

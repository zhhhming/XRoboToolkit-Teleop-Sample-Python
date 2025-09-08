#!/usr/bin/env python3
"""
快速测试joint_1是否能正常控制
"""

import mujoco
import numpy as np

def quick_test():
    # 加载模型
    try:
        model = mujoco.MjModel.from_xml_path("/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/scene_gen3.xml")
        data = mujoco.MjData(model)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 重置
    mujoco.mj_resetData(model, data)
    
    # 获取joint_1的控制器索引
    joint_1_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_1")
    joint_1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_1")
    
    print(f"Joint 1 actuator ID: {joint_1_actuator_id}")
    print(f"Joint 1 joint ID: {joint_1_joint_id}")
    
    # 检查控制范围
    ctrl_range = model.actuator_ctrlrange[joint_1_actuator_id]
    joint_range = model.jnt_range[joint_1_joint_id]
    print(f"控制范围: [{ctrl_range[0]:.3f}, {ctrl_range[1]:.3f}]")
    print(f"关节范围: [{joint_range[0]:.3f}, {joint_range[1]:.3f}]")
    
    # 测试控制
    print(f"\n初始位置: {data.qpos[joint_1_joint_id]:.3f}")
    
    # 设置目标位置
    target = 1.0
    data.ctrl[joint_1_actuator_id] = target
    
    # 仿真
    for i in range(2000):
        mujoco.mj_step(model, data)
        if i % 500 == 0:
            pos = data.qpos[joint_1_joint_id]
            vel = data.qvel[joint_1_joint_id]
            force = data.actuator_force[joint_1_actuator_id]
            print(f"Step {i:4d}: pos={pos:6.3f}, vel={vel:6.3f}, force={force:6.3f}")
    
    final_pos = data.qpos[joint_1_joint_id]
    print(f"\n最终位置: {final_pos:.3f}")
    print(f"目标位置: {target:.3f}")
    print(f"误差: {abs(final_pos - target):.3f}")
    
    if abs(final_pos - target) < 0.1:
        print("✓ Joint 1 工作正常!")
    else:
        print("✗ Joint 1 存在问题!")

if __name__ == "__main__":
    quick_test()
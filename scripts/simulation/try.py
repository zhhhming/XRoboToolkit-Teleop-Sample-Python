import mujoco
import mujoco.viewer

def main():
    # 载入你的 XML 文件
    xml_path = "D:\\xrobotics\\XRoboToolkit-Teleop-Sample-Python\\assets\\arx\\Gen\\scene_gen3.xml"   # 替换成你的路径
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 输出关节名称
    print("Joint names:")
    for i in range(model.njnt):
        print(f"  {i}: {model.joint(i).name}")

    # 输出body名称
    print("\nBody names:")
    for i in range(model.nbody):
        print(f"  {i}: {model.body(i).name}")

    # 启动viewer展示模型
    print("\nLaunching Mujoco viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()

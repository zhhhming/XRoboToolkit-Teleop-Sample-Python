from urdfpy import URDF
import meshcat

# 1. 读取 urdf
robot = URDF.load("/home/ming/xrrobotics_new/XRoboToolkit-Teleop-Sample-Python/assets/arx/Gen/gen3_gripper.urdf")

# 2. 打开 MeshCat 可视化
vis = meshcat.Visualizer().open()

# 3. 显示模型
robot.show(vis)

print("打开浏览器 http://127.0.0.1:7000 可以查看机器人模型")

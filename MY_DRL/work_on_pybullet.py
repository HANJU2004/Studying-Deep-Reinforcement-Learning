import pybullet as p
# import pybullet_envs.baselines.enjoy_pybullet_racecar
import numpy as np
import math
from time import sleep


class Controller:  # 自定义控制器，用来玩遥控车的以及测试关节
    def __init__(self, object=None):
        # 控制器缓存相机方向向量，键码等信息
        self.temp = np.array([0, -2, 0])
        self.key_w = ord('w')
        self.key_a = ord('a')
        self.key_s = ord('s')
        self.key_d = ord('d')
        self.object = object
        self.action_gap=6 #每k帧请求一次动作
        self.i=0

    def camera_update(self, object, distance):  # 更新相机位置，每帧调用一次
        position_o = np.array(p.getBasePositionAndOrientation(object)[0])
        position_c = self.temp
        line = position_o - position_c
        angle = math.atan(-line[0] / line[1]) / math.pi * 180
        if line[0] >= 0 and line[1] <= 0:
            angle += -180
        elif line[0] <= 0 and line[1] <= 0:
            angle += 180

        p.resetDebugVisualizerCamera(distance, angle, -30, position_o)
        line[2] = 0
        line2d = line / np.linalg.norm(line)
        self.temp = position_o - 2.7 * line2d

    def SendKeyboardCommand(self, object):
        key_dict = p.getKeyboardEvents()
        if p.B3G_UP_ARROW in key_dict:
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[0,1,5,3],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[70,70,70,70],
                                        forces=[1,1,1,1])

        if p.B3G_DOWN_ARROW in key_dict:
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[0,1,5,3],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[-50,-50,-50,-50],
                                        forces=[1,1,1,1])

        if p.B3G_LEFT_ARROW in key_dict:
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[4,2],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[10,10],
                                        forces=[0.5, 0.5])

        if p.B3G_RIGHT_ARROW in key_dict:
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[4,2],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[-10,-10],
                                        forces=[0.5, 0.5])

        if key_dict=={}:
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[0, 1,3,5],
                                        controlMode=p.VELOCITY_CONTROL,
                                        forces=[0, 0, 0, 0])
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[2,4],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0,0],
                                        forces=[0.5, 0.5])

    def SetTarget(self,object):
        self.object=object

    def step(self,action):
        state=[]
        if action==0:#up
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[2, 4],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0, 0],
                                        forces=[0.5, 0.5])
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[0, 1, 5, 3],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[70, 70, 70, 70],
                                        forces=[1, 1, 1, 1])
        if action==1:#down
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[2, 4],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0, 0],
                                        forces=[0.5, 0.5])
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[0, 1, 5, 3],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[-50, -50, -50, -50],
                                        forces=[1, 1, 1, 1])
        if action==2:#left
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[4, 2],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[10, 10],
                                        forces=[0.5, 0.5])
        if action==3:#right
            p.setJointMotorControlArray(bodyUniqueId=object,
                                        jointIndices=[4, 2],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[-10, -10],
                                        forces=[0.5, 0.5])

        if action==4:#nope
            p.setJointMotorControlArray(bodyUniqueId=self.object,
                                        jointIndices=[0, 1, 3, 5],
                                        controlMode=p.VELOCITY_CONTROL,
                                        forces=[0, 0, 0, 0])
            p.setJointMotorControlArray(bodyUniqueId=self.object,
                                        jointIndices=[2, 4],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0, 0],
                                        forces=[0.5, 0.5])


        velocity=p.getLinkState(self.object,0)[6]+p.getLinkState(self.object,0)[7]#线速度角速度列表


        return state#,reward,over



physicsClient = p.connect(p.GUI)  # 开始显示界面

# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)


p.setGravity(0, 0, -10)
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
plane = p.loadURDF("plane/plane.urdf")
# bike = p.loadURDF("C:/Users/lenovo/PycharmProjects/pytorch_test/venv/Lib/site-packages/pybullet_data/bicycle/bike.urdf",cubeStartPos, cubeStartOrientation)
RC = p.loadURDF("RC_car.urdf", [3, 3, 1], cubeStartOrientation)
quadruped=p.loadURDF("../pybullet_data/quadruped/spirit40.urdf", [-3, -3, 1], cubeStartOrientation)
slope=p.loadURDF("slope.urdf", [0, 0, 0.16], p.getQuaternionFromEuler([0.3, 0, 0]))
# human=p.loadURDF("pybullet_data/humanoid/humanoid.urdf", [-3, -3, 1], cubeStartOrientation)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(bike)

useRealTimeSimulation = 1
c = Controller()

active_joint_id = [i for i in range(p.getNumJoints(quadruped)) if p.getJointInfo(quadruped, i)[2] != p.JOINT_FIXED]
print("活动关节ID: ", active_joint_id)

p.setJointMotorControlArray(bodyUniqueId=RC,
                            jointIndices=[0,1,2,3,4,5],
                            controlMode=p.VELOCITY_CONTROL,
                            forces=[0,0,0,0,0,0])

p.setJointMotorControlArray(bodyUniqueId=quadruped,
                            jointIndices=[0,1,2,4,5,6,8,9,10,12,13,14],
                            controlMode=p.VELOCITY_CONTROL,
                            forces=[0,0,0,0,0,0,0,0,0,0,0,0])

p.addUserDebugText('TARGET',[1,1,1],[0,0,1],2,0)


if useRealTimeSimulation:
    p.setRealTimeSimulation(1)

while 1:
    if useRealTimeSimulation:
        # p.setGravity(0, 0, -10)
        sleep(0.01)  # Time in seconds.
        c.camera_update(RC, 1.6)
        c.SendKeyboardCommand(RC)


    else:
        p.stepSimulation()


import pybullet as p
import numpy as np
import math
import random
# from time import sleep

class Camera:  # 自定义控制器，用来玩遥控车的以及测试关节
    def __init__(self, object=None):
        # 控制器缓存相机方向向量，键码等信息
        self.temp = np.array([0, -2, 0])
        self.object = object
        self.positionACU=np.array([0.0,0.0,0.0])
        self.i=0

    def camera_update(self, object, distance):  # 更新相机位置，每帧调用一次
        position_o = np.array(p.getBasePositionAndOrientation(object)[0])
        self.positionACU+=0.3*(position_o-self.positionACU)
        position_c = self.temp
        line = position_o - position_c
        angle = math.atan(-line[0] / line[1]) / math.pi * 180
        if line[0] >= 0 and line[1] <= 0:
            angle += -180
        elif line[0] <= 0 and line[1] <= 0:
            angle += 180

        p.resetDebugVisualizerCamera(distance, angle, -30, self.positionACU)
        line[2] = 0
        line2d = line / np.linalg.norm(line)
        self.temp = position_o - 2.7 * line2d



class RC_ENV():
    #小车训练环境，目标：使小车行驶至设定地点
    def __init__(self,maxstep=500,gap=10):
        self.physicsClient = p.connect(p.GUI)  # 开始显示界面
        p.setGravity(0, 0, -10)
        self.plane = p.loadURDF("plane/plane.urdf", useMaximalCoordinates=True)
        self.RC = p.loadURDF("./RC_car.urdf", [3, 3, 1], p.getQuaternionFromEuler([0, 0, 0]))
        self.targetpos = np.array([random.randrange(-15,15),random.randrange(-15,15),0.06])
        self.target=p.addUserDebugText('TARGET',self.targetpos,[0,0,1],2,0)
        self.vector=np.array([1,0,0])
        self.step_number=0
        self.max_step=maxstep
        self.action_gap=gap
        self.test_mode=False
        self.camera=Camera()

    def reset(self):
        state = []
        self.step_number=0
        #重置环境
        p.removeUserDebugItem(self.target)
        p.resetBasePositionAndOrientation(self.RC,[0,0,0.3],p.getQuaternionFromEuler([0,0,0]))
        self.targetpos = np.array([random.randrange(-15,15),random.randrange(-15,15),0.03])
        self.target=self.target=p.addUserDebugText('TARGET',self.targetpos,[0,0,1],1,0)


        p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                    jointIndices=[0, 1, 3, 5],
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0, 0, 0, 0])
        p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                    jointIndices=[2, 4],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0, 0],
                                    forces=[0.5, 0.5])

        direction_vector = p.rotateVector(p.getBasePositionAndOrientation(self.RC)[1], np.array([1, 0, 0]))
        velocity = p.getBaseVelocity(self.RC)[0]
        to_target_vector = self.targetpos - p.getBasePositionAndOrientation(self.RC)[0]
        R_v = np.linalg.norm(velocity)
        R_t = np.linalg.norm(self.targetpos - p.getBasePositionAndOrientation(self.RC)[0])
        theta_v = math.acos(np.dot(direction_vector, velocity) / (R_v + 1e-5))
        if np.cross(direction_vector, velocity)[2] <= 0:
            theta_v = -theta_v
        theta_t = math.acos(np.dot(direction_vector, to_target_vector) / (R_t + 1e-5))  # 求当前向量与正对方向夹角，该夹角不一定沿xy平面
        if np.cross(direction_vector, to_target_vector)[2] <= 0:
            theta_t = -theta_t

        angle_velocity = p.getBaseVelocity(self.RC)[1]

        state.append(R_v)
        state.append(theta_v)
        state.append(R_t)
        state.append(theta_t)
        state += angle_velocity  # 角速度  #length=7

        return state

    def sample(self):
        return random.randrange(0,5,1)

    def step(self,action):
        state = []
        over=False

        ###ACTION
        if action == 0:  # up
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[2, 4],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0, 0],
                                        forces=[0.5, 0.5])
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[0, 1, 5, 3],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[70, 70, 70, 70],
                                        forces=[1, 1, 1, 1])
        if action == 1:  # down
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[2, 4],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0, 0],
                                        forces=[0.5, 0.5])
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[0, 1, 5, 3],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[-50, -50, -50, -50],
                                        forces=[1, 1, 1, 1])
        if action == 2:  # left
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[4, 2],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[10, 10],
                                        forces=[0.5, 0.5])
        if action == 3:  # right
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[4, 2],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[-10, -10],
                                        forces=[0.5, 0.5])

        if action == 4:  # nope
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[0, 1, 3, 5],
                                        controlMode=p.VELOCITY_CONTROL,
                                        forces=[0, 0, 0, 0])
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[2, 4],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0, 0],
                                        forces=[0.5, 0.5])

        ###STATE
        self.step_number += 1
        if self.step_number >= self.max_step and self.test_mode == False:
            over = True
            self.step_number = 0

        if self.test_mode == False:
            for i in range(self.action_gap):
                p.stepSimulation()

        # 状态：车头正对方向为基准，当前速度与正对方向的夹角（左正右负），距离（正）
        # 车与目标点连线向量与正对方向夹角，距离 # p.rotateVector()
        direction_vector = p.rotateVector(p.getBasePositionAndOrientation(self.RC)[1], np.array([1, 0, 0]))
        velocity=p.getBaseVelocity(self.RC)[0]
        to_target_vector=self.targetpos-p.getBasePositionAndOrientation(self.RC)[0]
        R_v=np.linalg.norm(velocity)
        R_t=np.linalg.norm(self.targetpos-p.getBasePositionAndOrientation(self.RC)[0])
        theta_v = math.acos(np.dot(direction_vector,velocity)/(R_v+1e-5))
        if np.cross(direction_vector,velocity)[2]<=0:
            theta_v=-theta_v
        #下面这行防止崩溃
        cos_theta_t=np.dot(direction_vector,to_target_vector)/(R_t+1e-5)
        if cos_theta_t<=-1:
            cos_theta_t=-1
        elif cos_theta_t>=1:
            cos_theta_t=1

        theta_t = math.acos(cos_theta_t)# 求当前向量与正对方向夹角，该夹角不一定沿xy平面
        if np.cross(direction_vector,to_target_vector)[2]<=0:
            theta_t=-theta_t

        angle_velocity = p.getBaseVelocity(self.RC)[1]


        ###REWARD
        temp1=(self.targetpos-p.getBasePositionAndOrientation(self.RC)[0])/R_t
        temp2=p.getBaseVelocity(self.RC)[0]#/(R_v+1e-10)
        # reward=(np.dot(temp1,temp2)/5)-0.4
        if -0.4<=theta_t<=0.4 and np.dot(temp1,temp2)/5-0.6>=0:
            reward=1
        else:
            reward=-1

        if self.test_mode==True:
            # debug模式使用
            self.camera.camera_update(self.RC, 2)
            p.addUserDebugText("RT="+str(round(R_t, 3)), [0, 0, 2], [0.5, 0.5, 1], 2, 0.08)
            p.addUserDebugText(str(round(reward, 3)), [0, 0, 2.5], [0.2, 0.5, 1], 2, 0.08)
            p.addUserDebugLine(p.getBasePositionAndOrientation(self.RC)[0], self.targetpos, [0.6, 0.2, 0.1],
                               lifeTime=0.08)


        state.append(R_v)
        state.append(theta_v)
        state.append(R_t / 20)
        state.append(theta_t)
        state += angle_velocity  # 角速度  #length=7


        ###OVER
        if(R_t<=0.8):
            # over=True
            p.removeUserDebugItem(self.target)
            self.targetpos = np.array([random.randrange(-15, 15), random.randrange(-15, 15), 0.03])
            self.target = self.target = p.addUserDebugText('TARGET', self.targetpos, [0, 0, 1], 1, 0)
            reward=0
            self.step_number = 0
        if(R_t>=30):
            # over = True
            reward = -400
            self.step_number = 0





        return state,reward,over


    def set_test(self):
        self.test_mode=True
        p.setRealTimeSimulation(1)









class QUADRUPED_ENV():
    #四足机器人训练环境，目标：1根据指令选择激活或者休眠状态，2激活状态时行走至目标地点并保持机身直立
    def __init__(self, maxstep=700, gap=9):
        self.physicsClient = p.connect(p.GUI)  # 开始显示界面
        p.setGravity(0, 0, -10)
        self.plane = p.loadURDF("pybullet_data/plane.urdf", useMaximalCoordinates=True)
        self.quadruped = p.loadURDF("pybullet_data/quadruped/spirit40.urdf", [0, 0, 1], p.getQuaternionFromEuler([0, 0, 0]))
        self.targetpos = np.array([random.randrange(-15, 15), random.randrange(-15, 15), 0.06])
        self.target = p.addUserDebugText('TARGET', self.targetpos, [0, 0, 1], 2, 0)
        self.vector = np.array([1, 0, 0])
        self.step_number = 0
        self.max_step = maxstep
        self.action_gap = gap
        self.test_mode = False

    def reset(self):
        state = []
        self.step_number = 0
        # 重置环境
        p.removeUserDebugItem(self.target)
        p.resetBasePositionAndOrientation(self.cheetah, [0, 0, 1], p.getQuaternionFromEuler([0, 0, 0]))
        self.targetpos = np.array([random.randrange(-15, 15), random.randrange(-15, 15), 0.03])
        self.target = self.target = p.addUserDebugText('TARGET', self.targetpos, [0, 0, 1], 1, 0)

        p.setJointMotorControlArray(bodyUniqueId=self.cheetah,
                                    jointIndices=[0, 1, 3, 5],
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0, 0, 0, 0])
        p.setJointMotorControlArray(bodyUniqueId=self.cheetah,
                                    jointIndices=[2, 4],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[0, 0],
                                    forces=[0.5, 0.5])

        direction_vector = p.rotateVector(p.getBasePositionAndOrientation(self.RC)[1], np.array([1, 0, 0]))
        velocity = p.getBaseVelocity(self.RC)[0]
        to_target_vector = self.targetpos - p.getBasePositionAndOrientation(self.RC)[0]
        R_v = np.linalg.norm(velocity)
        R_t = np.linalg.norm(self.targetpos - p.getBasePositionAndOrientation(self.RC)[0])
        theta_v = math.acos(np.dot(direction_vector, velocity) / (R_v + 1e-5))
        if np.cross(direction_vector, velocity)[2] <= 0:
            theta_v = -theta_v
        theta_t = math.acos(np.dot(direction_vector, to_target_vector) / (R_t + 1e-5))  # 求当前向量与正对方向夹角，该夹角不一定沿xy平面
        if np.cross(direction_vector, to_target_vector)[2] <= 0:
            theta_t = -theta_t

        angle_velocity = p.getBaseVelocity(self.RC)[1]

        state.append(R_v)
        state.append(theta_v)
        state.append(R_t)
        state.append(theta_t)
        state += angle_velocity  # 角速度  #length=7

        return state

    def sample(self):
        return random.randrange(0, 5, 1)

    def step(self, action):
        state = []
        over = False

        ###ACTION
        if action == 0:  # up
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[0, 1, 5, 3],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[70, 70, 70, 70],
                                        forces=[1, 1, 1, 1])
        if action == 1:  # down
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[0, 1, 5, 3],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[-50, -50, -50, -50],
                                        forces=[1, 1, 1, 1])
        if action == 2:  # left
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[4, 2],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[10, 10],
                                        forces=[0.5, 0.5])
        if action == 3:  # right
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[4, 2],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[-10, -10],
                                        forces=[0.5, 0.5])

        if action == 4:  # nope
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[0, 1, 3, 5],
                                        controlMode=p.VELOCITY_CONTROL,
                                        forces=[0, 0, 0, 0])
            p.setJointMotorControlArray(bodyUniqueId=self.RC,
                                        jointIndices=[2, 4],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=[0, 0],
                                        forces=[0.5, 0.5])

        ###STATE
        self.step_number += 1
        if self.step_number >= self.max_step and self.test_mode == False:
            over = True
            self.step_number = 0

        if self.test_mode == False:
            for i in range(self.action_gap):
                p.stepSimulation()

        # 状态：车头正对方向为基准，当前速度与正对方向的夹角（左正右负），距离（正）
        # 车与目标点连线向量与正对方向夹角，距离 # p.rotateVector()
        direction_vector = p.rotateVector(p.getBasePositionAndOrientation(self.RC)[1], np.array([1, 0, 0]))
        velocity = p.getBaseVelocity(self.RC)[0]
        to_target_vector = self.targetpos - p.getBasePositionAndOrientation(self.RC)[0]
        R_v = np.linalg.norm(velocity)
        R_t = np.linalg.norm(self.targetpos - p.getBasePositionAndOrientation(self.RC)[0])
        theta_v = math.acos(np.dot(direction_vector, velocity) / (R_v + 1e-5))
        if np.cross(direction_vector, velocity)[2] <= 0:
            theta_v = -theta_v
        theta_t = math.acos(np.dot(direction_vector, to_target_vector) / (R_t + 1e-5))  # 求当前向量与正对方向夹角，该夹角不一定沿xy平面
        if np.cross(direction_vector, to_target_vector)[2] <= 0:
            theta_t = -theta_t

        angle_velocity = p.getBaseVelocity(self.RC)[1]

        state.append(R_v)
        state.append(theta_v)
        state.append(R_t / 20)
        state.append(theta_t)
        state += angle_velocity  # 角速度  #length=7

        ###REWARD
        temp1 = (self.targetpos - p.getBasePositionAndOrientation(self.RC)[0]) / R_t
        temp2 = p.getBaseVelocity(self.RC)[0]  # /(R_v+1e-10)
        reward = (np.dot(temp1, temp2) / 5) - 0.4
        if -0.6 <= theta_t <= 0.6:
            reward += 0.8
        else:
            reward += -1

        ###OVER
        if (R_t <= 0.8):
            over = True
            reward = 400
            self.step_number = 0
        if (R_t >= 30):
            over = True
            reward = -400
            self.step_number = 0

        if self.test_mode == True:
            # debug模式使用
            p.addUserDebugText("RT=" + str(round(R_t, 3)), [0, 0, 2], [0.5, 0.5, 1], 2, 0.08)
            p.addUserDebugText(str(round(reward, 3)), [0, 0, 2.5], [0.2, 0.5, 1], 2, 0.08)
            p.addUserDebugLine(p.getBasePositionAndOrientation(self.RC)[0], self.targetpos, [0.6, 0.2, 0.1],
                               lifeTime=0.08)

        return state, reward, over

    def set_test(self):
        self.test_mode = True
        p.setRealTimeSimulation(1)









class ROBOT_ENV():
    #人形机器人训练环境，目标：给定身体朝向，移动方向，移动速度，正确行走。
    def __init__(self):
        pass
import random
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
#import pygame
#from IPython import display

#DQN倒立摆很基本的很基本的basicbasic

#神经网络
class DRL_net(nn.Module):
    def __init__(self):
        super(DRL_net,self).__init__()
        self.in_to_out=nn.Sequential(
            nn.Linear(4,50),
            nn.Sigmoid(),
            nn.Linear(50,50),
            nn.Sigmoid(),
            nn.Linear(50,2),
        )

    def forward(self,x):
        output=self.in_to_out(x)
        return output

#游戏环境
class GYM_ENV(gym.Wrapper):
    def __init__(self):
        #env = gym.make("CartPole-v1",render_mode="human")
        env = gym.make("CartPole-v1")
        super().__init__(env)
        self.env=env
        self.step_number=0

    def step(self, action):
        state,reward,terminated,truncated,info=self.env.step(action)
        over=terminated or truncated

        #此处可设置reward和终止条件
        if 2>=state[3]>=-2:
            reward=1
        else:
            reward=-1

        self.step_number+=1
        if self.step_number>=200:
            over=True
            self.step_number=0

        return state,reward,over




#数据池
class Pool:
    def __init__(self):
        self.pool=[]

    def update(self):
        old_len=len(self.pool)
        while len(self.pool)-old_len<200:
            self.pool.extend(play()[0])

        self.pool=self.pool[-1_0000:]

    def sample(self):
        return random.choice(self.pool)


#pool中的函数
def play():
    data=[]
    reward_sum=0
    state,rrr=env.reset()#重写
    over=False

    while not over:
        action=net(torch.FloatTensor(state).resize(1,4)).argmax().item()
        if random.random()<0.1:
            action=env.action_space.sample()

        obs=env.step(action)
        data.append((state,action,obs[1],obs[0],obs[2]))
        reward_sum+=obs[1]
        state=obs[0]
        over=obs[2]



    return data,reward_sum

def train_net(epo):
    criterion=nn.MSELoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=0.0008)

    for epoch in range(epo):
        pool.update()#玩一次得到200条

        for i in range(200):
            state,action,reward,next_state,over=pool.sample()
            value=torch.gather(input=net(torch.FloatTensor(state)),dim=0,index=torch.tensor(action,dtype=torch.int64))

            with torch.no_grad():
                target=net(torch.FloatTensor(next_state))
            target=target.max(dim=0)[0].reshape(-1,1)
            target=target*0.99*(1-over)+reward#目标值=此时刻奖励+下一时刻预估价值*discount系数

            loss=criterion(value.resize(1,1),target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch%10==0:
            test_result=sum([play()[-1]for j in range(10)])/10
            print('epoch=',epoch,'pool length=',len(pool.pool),'mean reward=',test_result)

            if test_result>=199.0:
                break


net=DRL_net()
env=GYM_ENV()
state=env.reset()
pool=Pool()

# for i in range(100):
#     action=env.action_space.sample()
#     observation=env.step(action)#返回字典
#     env.render()
#     print(observation,type(observation))
#     if observation[2]==True:
#         env.reset()
#         continue
#
#
# env.close()

train_net(1000)

# PATH="Cart_Pole_with_DQN.pth"
# torch.save(net.state_dict(),PATH)

# net.load_state_dict(torch.load(PATH))
# play()
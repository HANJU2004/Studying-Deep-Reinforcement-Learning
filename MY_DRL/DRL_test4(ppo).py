import math
import random
import gym
import torch
import torch.nn as nn
import numpy as np
import keyboard as kb

#memory具体如何工作，4个网络的更新方式？
#c改进学习率太小估不准
#采样过少
class AC_NET(nn.Module):#actor_critic
    def __init__(self,state_dim,action_dim):
        super(AC_NET, self).__init__()
        #actor
        self.in_to_out = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, action_dim),
            nn.Softmax(dim=0)
        )
        #critic
        self.in_to_value=nn.Sequential(
            nn.Linear(state_dim,10),
            nn.Tanh(),
            nn.Linear(10,10),
            nn.Tanh(),
            nn.Linear(10,1)
        )

    def forward(self, x):
        outputA = self.in_to_out(x)
        outputB =self.in_to_value(x)
        return outputA,outputB

    def act(self,state,memory):
        state = torch.from_numpy(state)
        action_probs = self.in_to_out(state).tolist()
        action = random.choices(range(4), weights=action_probs, k=1)[0]  # 采样

        memory.state.append(state)
        memory.action.append(action)
        memory.log_prob.append(math.log(action_probs[action] + 1e-8, 2))

        return action


class Memory():
    #memory是1个由列表组成的列表[[sarnldsarnld],[sarnldsarnldsarnld],[].....]
    def __init__(self):
        self.memory_data=[]

    def clear(self):
        del self.memory_data[:]


class GYM_ENV(gym.Wrapper):
    def __init__(self,game,max_step,render=False):
        if render==True:
            env = gym.make(game,render_mode="human")
        else:
            env = gym.make(game)

        super().__init__(env)
        self.env=env
        self.step_number=0
        self.max_step=max_step


    def step(self, action):
        state,reward,terminated,truncated,info=self.env.step(action)
        over=terminated or truncated
        self.step_number+=1
        if self.step_number >= self.max_step:
            over = True
            self.step_number = 0
        #此段重写

        return state,reward,over

    def reset(self):
        return self.env.reset()[0]


class PPO():#这里面包括以上的class,训练方法类，返回网络参数（或模型）
    def __init__(self,env,state_dim,action_dim):
        self.memory=Memory()
        self.env=env
        self.ac_net=AC_NET(state_dim,action_dim)
        self.ac_net_old = AC_NET(state_dim,action_dim)
        self.ac_net_old.load_state_dict(self.ac_net.state_dict())
        self.optimizer=torch.optim.Adam(self.ac_net.parameters(),lr=0.004)
        self.criterion=nn.MSELoss()
        self.state_dim=state_dim
        self.action_dim=action_dim


    def act(self,state):#使用action_old根据state返回action及所有动作的概率，在play中调用，act后接env.step,(之后可以删了)
        state=torch.from_numpy(state)
        action_probs=self.ac_net_old.in_to_out(state).tolist()
        action = random.choices(range(self.action_dim), weights=action_probs, k=1)[0]#采样

        return action,action_probs

    def play(self):#做一次完整的游戏周期，并将其保存至memory中，返回一个轨迹data列表
        data = []  # [state,action,reward,next_state,over],[],[],[]......
        reward_sum = 0
        state = env.reset()
        env.step_number = 0
        over = False

        while not over:
            action,action_probs=self.act(state)#选动作并存储到记忆中
            if random.random() < 0.001:
                action = env.action_space.sample()

            obs = env.step(action)
            reward = obs[1]
            next_state=obs[0]
            log_prob=math.log(action_probs[action]+1e-8)
            over = obs[2]
            data.append([state, action, reward, next_state, log_prob, over])
            reward_sum += reward
            state = obs[0]

        data.append(reward_sum)
        self.memory.memory_data.append(data)

        return data, reward_sum

    def get_dis_reward(self,memory):#输入一条轨迹，输出该轨迹的每个状态对应的折扣价值列表
        length=len(memory)-1
        reward=[]
        for i in range(length):
            reward.append(memory[i][2])

        dis_r = []
        for i in range(len(reward)):
            s = 0
            for j in range(i, len(reward)):
                s += reward[j] * 0.92 ** (j - i)  # 折扣价值，gamma=0.98
            dis_r.append(s)
        return dis_r

    def train_C(self,memory,discounted_reward):#只使用400条里的1条作为输入，反复训练10次
        for i in range(10):#一条数据反复训练10次
            loss=torch.zeros(1)
            length = len(memory) - 1
            value=torch.zeros(length)
            target=torch.zeros(length)
            for j in range(length):
                cu_state = memory[j][0]
                cu_action = memory[j][1]
                cu_reward = memory[j][2]
                cu_nextstate = memory[j][3]
                cu_logprob = memory[j][4]

                value[j]=(self.ac_net.in_to_value(torch.FloatTensor(cu_state))[0])

                target[j]=(discounted_reward[j])
                # target=cu_reward+0.95*self.ac_net.in_to_value(torch.tensor(cu_nextstate))
                # loss+=self.criterion(value,target)
                # loss += self.criterion(value.resize(1), torch.FloatTensor([discounted_reward[j]]).resize(1))
            # loss=loss/length
            # print(value)
            loss=self.criterion(value,target)
            # loss.backward()
            # self.optimizer.step()
            # self.optimizer.zero_grad()
            return loss


    def train_A(self,memory,discounted_reward):#只使用400条里的1条作为输入，更新1次
        loss=torch.zeros(1)
        length=len(memory)-1
        for j in range(length):
            cu_state = memory[j][0]
            cu_action = memory[j][1]
            cu_reward = memory[j][2]
            cu_nextstate = memory[j][3]
            cu_logprob = memory[j][4]
            reward_sum=memory[-1]


            state_value=self.ac_net.in_to_value(torch.FloatTensor(cu_state))
            this_logprob=torch.log(self.ac_net.in_to_out(torch.FloatTensor(cu_state)).gather(dim=0,index=torch.tensor(cu_action,dtype=torch.int64)))
            # loss+=-this_logprob*discounted_reward[j]
            # surr1=torch.exp(this_logprob-torch.FloatTensor([cu_logprob]))*(reward_sum-state_value)
            # surr2=torch.clamp(torch.exp(this_logprob-torch.FloatTensor([cu_logprob])),0.8,1.2)*(reward_sum-state_value)
            surr1 = torch.exp(this_logprob - torch.FloatTensor([cu_logprob])) * (discounted_reward[j] - state_value)
            surr2 = torch.clamp(torch.exp(this_logprob - torch.FloatTensor([cu_logprob])), 0.8, 1.2) * (discounted_reward[j] - state_value)
            loss+=-torch.min(surr1,surr2)

        loss=loss/length
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        return loss

    def train_net(self,epoch,numbers=400):
        for epo in range(epoch):
            self.memory.clear()
            for i in range(numbers):
                self.play()



            for j in range(3):
                loss_final=torch.zeros(1)
                for i in range(numbers):
                    discounted_reward=self.get_dis_reward(self.memory.memory_data[i])
                    loss_final+=self.train_C(self.memory.memory_data[i],discounted_reward)
                    loss_final+=self.train_A(self.memory.memory_data[i],discounted_reward)
                loss_final.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.ac_net_old.load_state_dict(self.ac_net.state_dict())
            self.memory.clear()
            test_result = sum([self.play()[-1] for j in range(20)]) / 20
            print('epoch=', epo, 'mean reward=', test_result)

        return self.ac_net.state_dict()

env=GYM_ENV("LunarLander-v2",500)
ppo=PPO(env=env,state_dim=8,action_dim=4)
ppo.train_net(epoch=400,numbers=35)

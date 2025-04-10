import math
import random
from time import sleep
import numpy as np
import torch
import torch.nn as nn


class AC_NET(nn.Module):  # actor_critic
    # 具有离散动作模式（只支持单动作）与连续动作模式（支持多动作）
    def __init__(self, state_dim, action_dim, mode="discrete"):
        super(AC_NET, self).__init__()
        self.mode = mode
        self.state_dim = state_dim
        self.action_dim = action_dim
        # actor，分为离散和连续两种情况
        if self.mode == "discrete":
            self.in_to_out = nn.Sequential(
                nn.Linear(state_dim, 25),
                nn.Tanh(),
                nn.Linear(25, 25),
                nn.Tanh(),
                nn.Linear(25, action_dim),
                nn.Softmax(dim=0)
            )
        elif self.mode == "continuous":
            self.in_to_out = nn.Sequential(
                nn.Linear(state_dim, 60),
                nn.Tanh(),
                nn.Linear(60, 60),
                nn.Tanh(),
            )
            self.to_mu = nn.Sequential(
                nn.Linear(60, action_dim),
                nn.Tanh(),
            )
            self.to_sigma = nn.Sequential(
                nn.Linear(60, action_dim),
                nn.Tanh(),
            )

        # critic
        self.in_to_value = nn.Sequential(
            nn.Linear(state_dim, 15),
            nn.Tanh(),
            nn.Linear(15, 15),
            nn.Tanh(),
            nn.Linear(15, 1)
        )

    def forward(self, x):
        if self.mode == "discrete":
            action_prob = self.in_to_out(x)
            state_value = self.in_to_value(x)
            return action_prob, state_value
        elif self.mode == "continuous":
            temp = self.in_to_out(x)
            mu = self.to_mu(temp)
            sigma = self.to_sigma(temp)
            state_value = self.in_to_value(x)
            return mu, sigma, state_value

    def act(self, state):  # actor网络接受状态参数并返回动作及其当前概率
        # 输入numpy返回tensor
        if self.mode == "discrete":
            state = torch.FloatTensor(state).detach()
            action_probs = self.in_to_out(state)
            action = random.choices(range(self.action_dim), weights=action_probs.tolist(), k=1)[0]  # 采样
            return action, action_probs  # 1,n
        if self.mode == "continuous":
            # 连续动作的act与离散不同，由于可以同时有多个连续动作，所以返回的动作及其概率是等长的列表
            state = torch.FloatTensor(state).detach()
            temp = self.in_to_out(state)
            mu = self.to_mu(temp)  # 均值
            sigma = torch.exp(self.to_sigma(temp))  # 标准差
            # 根据均值与标准差采样动作（-1到1，之后映射到动作空间内即可）
            # action = []
            #
            # for i in range(self.action_dim):
            #     action.append(random.normalvariate(mu=mu[i].item(), sigma=sigma[i].item()))  # 采样得到动作值（-1~1之间）
            distribution = torch.distributions.Normal(mu, sigma)
            action = distribution.sample().tolist()
            action_probs = torch.exp(distribution.log_prob(torch.FloatTensor(action)))  # 列表中记录该动作按照正态分布的概率
            return action, action_probs  # n,n

    def get_new_prob(self, current_state, previous_action):
        # 输入memory中的state和action，用新网络得到记忆中的动作被采样到的概率并返回
        # 输入numpy返回tensor
        if self.mode == "discrete":
            state = torch.FloatTensor(current_state).detach()
            action_probs = self.in_to_out(state)
            action_probs = action_probs[previous_action]
            return action_probs  # 1,n
        if self.mode == "continuous":
            # 连续动作的act与离散不同，由于可以同时有多个连续动作，所以返回的动作及其概率是等长的列表
            state = torch.FloatTensor(current_state).detach()
            temp = self.in_to_out(state)
            mu = self.to_mu(temp)  # 均值
            sigma = torch.exp(self.to_sigma(temp))  # 标准差

            action_probs = torch.exp(torch.distributions.Normal(mu, sigma).log_prob(
                torch.FloatTensor(previous_action)))  # 列表中记录该动作按照正态分布的概率
            return action_probs  # n,n

    def evaluate(self, state):  # critic网络进行状态评估返回价值value
        return self.in_to_value(state)


class Memory():
    # memory是1个由列表组成的列表[[sarnldsarnld],[sarnldsarnldsarnld],[].....]
    def __init__(self):
        self.memory_data = []
        self.state = []
        self.action = []
        self.logprob = []
        self.reward = []
        self.discounted_reward = []
        self.over = []

    def clear(self):
        del self.memory_data[:]


class PPO_discrete():  # 这里面包括以上的class,训练方法类，返回网络参数（或模型）
    def __init__(self, net, env, state_dim, action_dim, LR=0.001, K_epo=3):
        self.memory = Memory()
        self.env = env
        self.ac_net = AC_NET(state_dim, action_dim, mode="discrete")
        if net != None:
            self.ac_net.load_state_dict(net.state_dict())
        self.ac_net_old = AC_NET(state_dim, action_dim, mode="discrete")
        self.ac_net_old.load_state_dict(self.ac_net.state_dict())
        self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=LR)
        self.A_opt = torch.optim.Adam(self.ac_net.in_to_out.parameters(), lr=LR)
        self.C_opt = torch.optim.Adam(self.ac_net.in_to_value.parameters(), lr=5 * LR)
        self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.K_epo = K_epo

    # def act(self, state):  # 使用action_old根据state返回action及所有动作的概率，在play中调用，act后接env.step,(之后可以删了)
    #     state = torch.FloatTensor(state).detach()
    #     action_probs = self.ac_net_old.in_to_out(state).tolist()
    #     action = random.choices(range(self.action_dim), weights=action_probs, k=1)[0]  # 采样
    #     return action, action_probs

    def play(self):  # 做一次完整的游戏周期，并将其保存至memory中，返回一个轨迹data列表
        data = []  # [state,action,reward,next_state,over],[],[],[]......
        reward_sum = 0
        state = self.env.reset()
        over = False
        # for i in range(5):
        #     self.env.step(4)#这个目的是先让机器人落地，避免初始化的时候就翻了

        while not over:
            with torch.no_grad():
                action, action_probs = self.ac_net_old.act(state)  # 选动作并存储到记忆中
            action_logprobs = torch.log(action_probs[action]).tolist()
            if random.random() < 0.002:
                action = self.env.sample()

            obs = self.env.step(action)
            reward = obs[1]
            next_state = obs[0]
            # log_prob = math.log(action_probs[action] + 1e-8)
            over = obs[2]
            data.append([state, action, reward, next_state, action_probs[action], over])
            reward_sum += reward
            state = obs[0]

        data.append(reward_sum)
        self.memory.memory_data.append(data)

        return data, reward_sum

    def test_play(self):
        self.env.set_test()
        reward_sum = 0
        state = self.env.reset()
        self.env.step_number = 0
        over = False

        while not over:
            # sleep(0.03)
            action, action_probs = self.ac_net_old.act(state)  # 选动作并存储到记忆中
            if random.random() < 0.001:
                action = self.env.sample()

            obs = self.env.step(action)
            reward = obs[1]
            over = obs[2]
            reward_sum += reward
            state = obs[0]

        return reward_sum

    def get_dis_reward(self, memory):  # 输入一条轨迹，输出该轨迹的每个状态对应的折扣价值列表
        length = len(memory) - 1
        reward = []
        for i in range(length):
            reward.append(memory[i][2])

        dis_r = []
        for i in range(len(reward)):
            s = 0
            for j in range(i, len(reward)):
                s += reward[j] * 0.96 ** (j - i)  # 折扣价值，gamma=0.85
            dis_r.append(s)
        dis_r = np.array(dis_r)

        return dis_r

    def train_C(self, memory, discounted_reward):  # 只使用400条里的1条作为输入
        loss = torch.zeros(1)
        length = len(memory) - 1
        value = torch.zeros(length)
        target = torch.zeros(length)
        for j in range(length):
            cu_state = memory[j][0]
            cu_action = memory[j][1]
            cu_reward = memory[j][2]
            cu_nextstate = memory[j][3]
            cu_logprob = memory[j][4]

            value[j] = self.ac_net.evaluate(torch.FloatTensor(cu_state).detach())[0]

            target[j] = discounted_reward[j]
            # target=cu_reward+0.95*self.ac_net.in_to_value(torch.tensor(cu_nextstate))
            # loss+=self.criterion(value,target)
            # loss += self.criterion(value.resize(1), torch.FloatTensor([discounted_reward[j]]).resize(1))
        # loss=loss/length
        # print(value)
        loss = self.criterion(value, target)
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        return loss

    def train_A(self, memory, discounted_reward):  # 只使用400条里的1条作为输入，更新1次
        loss = torch.zeros(1)
        length = len(memory) - 1
        for j in range(length):
            cu_state = memory[j][0]
            cu_action = memory[j][1]
            cu_reward = memory[j][2]
            cu_nextstate = memory[j][3]
            old_prob = memory[j][4]
            reward_sum = memory[-1]

            state_value = self.ac_net.evaluate(torch.FloatTensor(cu_state)).detach()

            new_prob = self.ac_net.get_new_prob(cu_state, cu_action)

            # loss+=-this_logprob*reward_sum
            # surr1=torch.exp(this_logprob-torch.FloatTensor([cu_logprob]))*(reward_sum-state_value)
            # surr2=torch.clamp(torch.exp(this_logprob-torch.FloatTensor([cu_logprob])),0.8,1.2)*(reward_sum-state_value)
            surr1 = (new_prob / old_prob + 1e-8) * (discounted_reward[j] - state_value)
            surr2 = torch.clamp((new_prob / old_prob + 1e-8), 0.8, 1.2) * (discounted_reward[j] - state_value)
            loss += -torch.min(surr1, surr2)

        loss = loss / length
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        return loss

    def train_net(self, epoch, numbers=100):
        for epo in range(epoch):
            self.memory.clear()
            reward_sum = 0
            for i in range(numbers):
                reward_sum += self.play()[-1]
            print('epoch=', epo, 'mean reward=', reward_sum / numbers)
            print("本轮数据收集完毕，计算中")
            for j in range(self.K_epo):
                loss_final = torch.zeros(1)
                for i in range(numbers):
                    discounted_reward = self.get_dis_reward(self.memory.memory_data[i])
                    loss_final += self.train_C(self.memory.memory_data[i], discounted_reward)
                    loss_final += self.train_A(self.memory.memory_data[i], discounted_reward)
                print("aaa")
                (loss_final / numbers).backward()
                print("bbb")
                self.A_opt.step()
                self.A_opt.zero_grad()
                self.C_opt.step()
                self.C_opt.zero_grad()

            self.ac_net_old.load_state_dict(self.ac_net.state_dict())
            self.memory.clear()

        return self.ac_net, self.ac_net.in_to_out


# 连续动作训练器
class PPO_continuous():  # 这里面包括以上的class,训练方法类，返回网络参数（或模型）
    def __init__(self, net, env, state_dim, action_dim, LR=0.001, K_epo=3):
        self.memory = Memory()
        self.env = env
        self.ac_net = AC_NET(state_dim, action_dim, mode="continuous")
        if net != None:
            self.ac_net.load_state_dict(net.state_dict())
        self.ac_net_old = AC_NET(state_dim, action_dim, mode="continuous")
        self.ac_net_old.load_state_dict(self.ac_net.state_dict())
        self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=LR)
        self.A_opt = torch.optim.Adam(self.ac_net.in_to_out.parameters(), lr=LR)
        self.C_opt = torch.optim.Adam(self.ac_net.in_to_value.parameters(), lr=2 * LR)
        self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.K_epo = K_epo

    # def act(self, state):  # 使用action_old根据state返回action及所有动作的概率，在play中调用，act后接env.step,(之后可以删了)
    #     #连续动作的act与离散不同，由于可以同时有多个连续动作，所以返回的动作及其概率是等长的列表
    #     state = torch.FloatTensor(state).detach()
    #     temp=self.ac_net_old.in_to_out(state)
    #     mu = self.ac_net_old.to_mu(temp)#均值
    #     sigma=self.ac_net_old.to_sigma(temp)#标准差
    #     #根据均值与标准差采样动作（-1到1，之后映射到动作空间内即可）
    #     action=[]
    #     action_probs=[]
    #     for i in range(self.action_dim):
    #         action.append(random.normalvariate(mu=mu[i].item(),sigma=sigma[i].item()))   # 采样得到动作值（-1~1之间）
    #         action_probs.append(torch.distributions.Normal(mu[i],sigma[i]+1e-8).log_prob(torch.FloatTensor(action).detach()))#列表中记录该动作按照正态分布的概率
    #     return action, action_probs

    def play(self):  # 做一次完整的游戏周期，并将其保存至memory中，返回一个轨迹data列表
        data = []  # [state,action,reward,next_state,over],[],[],[]......
        reward_sum = 0
        state = self.env.reset()
        over = False
        # for i in range(5):
        #     self.env.step(4)  # 这个目的是先让机器人落地，避免初始化的时候就翻了

        while not over:
            with torch.no_grad():
                action, action_probs = self.ac_net_old.act(state)  # 选动作并存储到记忆中
            action_logprobs = torch.log(action_probs).tolist()
            if random.random() < 0.002:
                action = self.env.sample()

            obs = self.env.step(action)
            reward = obs[1]
            next_state = obs[0]
            over = obs[2]
            data.append([state, action, reward, next_state, action_probs, over])
            reward_sum += reward
            state = obs[0]

        data.append(reward_sum)
        self.memory.memory_data.append(data)

        return data, reward_sum

    def test_play(self):
        self.env.set_test()
        reward_sum = 0
        state = self.env.reset()
        self.env.step_number = 0
        over = False

        while not over:
            sleep(0.05)
            action, action_probs = self.ac_net_old.act(state)  # 选动作并存储到记忆中
            if random.random() < 0.001:
                action = self.env.sample()

            obs = self.env.step(action)
            reward = obs[1]
            over = obs[2]
            reward_sum += reward
            state = obs[0]

        return reward_sum

    def get_dis_reward(self, memory):  # 输入一条轨迹，输出该轨迹的每个状态对应的折扣价值列表
        length = len(memory) - 1
        reward = []
        for i in range(length):
            reward.append(memory[i][2])

        dis_r = []
        for i in range(len(reward)):
            s = 0
            for j in range(i, len(reward)):
                s += reward[j] * 0.96 ** (j - i)  # 折扣价值，gamma=0.85
            dis_r.append(s)
        dis_r = np.array(dis_r)

        return dis_r
        # return (dis_r-dis_r.mean())/(dis_r.std()+1e-5)#对折扣奖励进行标准化，这样使得“学习率”不会过于受到奖励设置大小的影响（比如10000的reward使学习率过大）

    def train_C(self, memory, discounted_reward):  # 只使用400条里的1条作为输入
        length = len(memory) - 1
        value = torch.zeros(length)
        target = torch.zeros(length)
        for j in range(length):
            cu_state = memory[j][0]
            cu_action = memory[j][1]
            cu_reward = memory[j][2]
            cu_nextstate = memory[j][3]
            cu_logprob = memory[j][4]

            value[j] = self.ac_net.evaluate(torch.FloatTensor(cu_state).detach())[0]

            target[j] = discounted_reward[j]
            # target=cu_reward+0.95*self.ac_net.in_to_value(torch.tensor(cu_nextstate))
            # loss+=self.criterion(value,target)
            # loss += self.criterion(value.resize(1), torch.FloatTensor([discounted_reward[j]]).resize(1))
        # loss=loss/length

        loss = self.criterion(value, target)
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        return loss

    def train_A(self, memory, discounted_reward):  # 只使用400条里的1条作为输入，更新1次
        loss = torch.zeros(1)
        length = len(memory) - 1
        for j in range(length):
            cu_state = memory[j][0]
            cu_action = memory[j][1]
            cu_reward = memory[j][2]
            cu_nextstate = memory[j][3]
            old_prob = memory[j][4]
            reward_sum = memory[-1]

            state_value = self.ac_net.evaluate(torch.FloatTensor(cu_state)).detach()
            new_prob = self.ac_net.get_new_prob(cu_state, cu_action)

            # loss+=-this_logprob*reward_sum
            # surr1=torch.exp(this_logprob-torch.FloatTensor([cu_logprob]))*(reward_sum-state_value)
            # surr2=torch.clamp(torch.exp(this_logprob-torch.FloatTensor([cu_logprob])),0.8,1.2)*(reward_sum-state_value)

            surr1 = (new_prob / old_prob + 1e-8) * (
                    discounted_reward[j] - state_value)  # .mean()
            surr2 = torch.clamp((new_prob / old_prob + 1e-8), 0.8, 1.2) * (
                    discounted_reward[j] - state_value)  # .mean()
            loss += -torch.min(surr1, surr2).mean()

        loss = loss / length
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        return loss

    def train_net(self, epoch, numbers=100):
        for epo in range(epoch):
            self.memory.clear()
            reward_sum = 0
            for i in range(numbers):
                reward_sum += self.play()[-1]
            print('epoch=', epo, 'mean reward=', reward_sum / numbers)
            print("本轮数据收集完毕，计算中")
            for j in range(self.K_epo):
                loss_final = torch.zeros(1)
                for i in range(numbers):
                    discounted_reward = self.get_dis_reward(self.memory.memory_data[i])
                    loss_final += self.train_C(self.memory.memory_data[i], discounted_reward)
                    loss_final += self.train_A(self.memory.memory_data[i], discounted_reward)
                print("aaa")
                (loss_final / numbers).backward()
                print("bbb")
                self.A_opt.step()
                self.A_opt.zero_grad()
                self.C_opt.step()
                self.C_opt.zero_grad()

            self.ac_net_old.load_state_dict(self.ac_net.state_dict())
            self.memory.clear()

        return self.ac_net, self.ac_net.in_to_out

import random
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import keyboard as kb

#策略梯度法实验验证  连续动作+去基线+AC算法改良lr0.004adam

#神经网络
class DRL_net(nn.Module):#激活函数一定不要用sigmoid，最好用relu！！这是血的教训！！！
    def __init__(self):
        super(DRL_net,self).__init__()
        self.in_to_out=nn.Sequential(
            nn.Linear(8,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,4),
            nn.Softmax(dim=0)
        )

    def forward(self,x):
        output=self.in_to_out(x)
        return output

#游戏环境
class GYM_ENV(gym.Wrapper):
    def __init__(self):
        # env = gym.make("LunarLander-v2",render_mode="human")
        env = gym.make("LunarLander-v2")
        super().__init__(env)
        self.env=env
        self.step_number=0


    def step(self, action):
        state,reward,terminated,truncated,info=self.env.step(action)
        over=terminated or truncated
        self.step_number+=1
        if self.step_number >= 350:
            over = True
            reward=-115
            self.step_number = 0
        #此段重写


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


#PG中就不只是pool中调用了，训练会直接调用来得到一条轨迹
def play():
    data=[]#[state,action,reward,next_state,over],[],[],[]......
    reward_sum=0
    state,rrr=env.reset()
    env.step_number=0
    over=False

    while not over:
        prob=net(torch.FloatTensor(state)).tolist()###
        action=random.choices(range(4),weights=prob,k=1)[0]

        if random.random()<0.001:
            action=env.action_space.sample()

        obs=env.step(action)
        reward=obs[1]
        over = obs[2]
        data.append((state,action,reward,over))
        reward_sum+=obs[1]
        state=obs[0]




    return data,reward_sum

def train_net(epo):

    optimizer=torch.optim.Adam(net.parameters(),lr=0.004)

    for epoch in range(epo):

        loss_for_this_epo=torch.zeros(1)

        i=0
        reward_sum_for_each_play = []
        average_sum_for_each_play = []
        for i in range(5):#一个batch玩n次，没有去基线。
            data,reward_sum=play()
            reward_sum_for_each_play.append(reward_sum)
            state = []
            action = []
            reward = []


            for q in range(len(data)):
                state.append(data[q][0])
                action.append(data[q][1])
                reward.append(data[q][2])


            #下面计算每个state的价值，每个state对应选择的那个的action的概率将增大，如果他们的价值value为正数的话
            value=[]
            for t in range(len(reward)):
                s=0
                for j in range(t,len(reward)):
                    s+=reward[j]*0.99**(j-t)#折扣价值
                value.append(s)
            value=torch.FloatTensor(value).reshape(-1,1)


            #prob是当前选择的动作的概率，这里重新与环境交互，
            for q in range(len(state)):
                prob=net(torch.FloatTensor(np.array(state[q]))).gather(dim=0,index=torch.tensor(action[q],dtype=torch.int64))
                #prob=(prob+1e-8).log()*average_sum_for_each_play[q]#
                loss=(prob+1e-8).log()*value[q]
                # loss = -(prob + 1e-8).log() * reward_sum
                # loss=-prob.mean()
                # loss.backward()
                loss_for_this_epo[0]+=-loss[0]


        #将N次累计的梯度进行更新
        loss_for_this_epo.backward()
        optimizer.step()
        optimizer.zero_grad()


        if epoch%5==0:
            test_result=sum([play()[-1]for j in range(4)])/4
            print('epoch=',epoch,'mean reward=',test_result)
            if test_result>=199.0:
                break


net=DRL_net()
env=GYM_ENV()
state=env.reset()
env.step_number=0
pool=Pool()



# PATH='LunarLander(PG)_NEW.pth'
# net.load_state_dict(torch.load(PATH))
train_net(400)
# torch.save(net.state_dict(),PATH)


# PATH='LunarLander(PG)_NEW.pth'
# net.load_state_dict(torch.load(PATH))
# play()
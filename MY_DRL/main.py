import torch
import PPO_trainer
import gym
import Pybullet_envs

#你妈的我早就该想到这个他妈输入的状态出大问题，改改赶紧的
#target方向向量在坐标系转换时出现问题，尚不知道原因（然而速度方向向量就没事）
#问题：网络结构的计算量翻倍的情况下，运行时间几乎一样，。。。。。
#这说明我应该尝试用矩阵代替循环输入,但暂时不考虑

class GYM_ENV(gym.Wrapper):
    def __init__(self, game, max_step, render=False):
        if render == True:
            env = gym.make(game, render_mode="human")
        else:
            env = gym.make(game)

        super().__init__(env)
        self.env = env
        self.step_number = 0
        self.max_step = max_step

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        over = terminated or truncated
        self.step_number += 1
        if self.step_number >= self.max_step:
            over = True
            self.step_number = 0
        # 此段重写



        return state, reward, over

    def reset(self):
        return self.env.reset()[0]

    def sample(self):
        return self.env.action_space.sample()




PATH= "RC_car_nn(25_15).pth"
quadruped_PATH= "quadruped_nn(60_15).pth"

net= PPO_trainer.AC_NET(7, 5, mode="discrete")
net.load_state_dict(torch.load(PATH))
#
env=Pybullet_envs.RC_ENV(maxstep=2500,gap=16)
# env=GYM_ENV("Pendulum-v1",120,render=False)
# env=GYM_ENV("CartPole-v1",500)
trainer= PPO_trainer.PPO_discrete(net=net, env=env, state_dim=7, action_dim=5, LR=0.002)
# trainer=PPO_trainer.PPO_continuous(net=None,env=env,state_dim=3,action_dim=1,LR=0.0006)

#训练用
# trained_ACnet,_=trainer.train_net(epoch=100,numbers=6)
# torch.save(trained_ACnet.state_dict(),quadruped_PATH)

#测试用
trainer.test_play()
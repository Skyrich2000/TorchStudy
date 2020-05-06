import numpy as np
import threading
import random
import time
import pylab
import gym

import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 멀티쓰레딩을 위한 글로벌 변수
global episode, render
render = True
episode = 0
EPISODES = 8000000
# 환경 생성
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
lock = threading.Lock()

class Net(nn.Module):
    def __init__(self, input_size, cfg):
        super(Net, self).__init__()

        def init_weights(m):
            if m == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        layer = []
        for v in cfg:
            if v == 'ReLU':
                layer += [nn.ReLU(inplace=True)]
            elif v == 'Softmax':
                layer += [nn.Softmax(dim=1)]
            else:
                layer += [nn.Linear(input_size, v)]
                input_size = v

        self.net = nn.Sequential(*layer)
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

# 브레이크아웃에서의 A3CAgent 클래스(글로벌신경망)
class A3CAgent:
    def __init__(self):
        # 상태크기와 행동크기를 갖고옴
        self.state_size = 4
        self.action_size = 2
        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        # 쓰레드의 갯수
        self.threads = 8
        self.scores, self.episodes = [], []

        # 정책신경망과 가치신경망을 생성
        self.actor, self.critic = self.build_model()

        print("load")
        self.actor.load_state_dict(torch.load(
            './save_model/a3c_actor.pth', map_location=torch.device('cpu')))

    # 정책신경망과 가치신경망을 생성
    def build_model(self):
        actor = Net(self.state_size, [24, 'ReLU', self.action_size, 'Softmax']).to(device)
        critic = Net(self.state_size, [24, 'ReLU', 24, 'ReLU', 1]).to(device)
        return actor, critic

    def run(self):
        print("start", threading.currentThread().getName())
        env = gym.make('CartPole-v1')

        step = 0

        while episode < EPISODES:
            done = False

            score = 0
            state = env.reset()

            while not done:
                env.render()
                step += 1
                action  = self.get_action(state)

                # 선택한 행동으로 한 스텝을 실행
                next_state, reward, done, _ = env.step(action)
                   
    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택
    def get_action(self, state):
        state = torch.FloatTensor(state).to(device).view([1, self.state_size])
        #print("action: ", self.local_actor(state), self.local_actor(state)[0].multinomial(1).item())
        return self.actor(state)[0].multinomial(1).item()

if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.run()

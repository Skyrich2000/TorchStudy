from skimage.color import rgb2gray
from skimage.transform import resize
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
import torchvision.transforms as transforms

# 멀티쓰레딩을 위한 글로벌 변수
global episode, render
render = True
episode = 0
EPISODES = 8000000
# 환경 생성
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
lock = threading.Lock()

actor_cfg = [256, 'ReLU', 3, 'Softmax']
critic_cfg = [256, 'ReLU', 1]

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        def init_weights(m):
            if m == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
            torch.nn.ReLU()
            )

        input_size = 9*9*32
        layer = []
        for v in cfg:
            if v == 'ReLU':
                layer += [nn.ReLU(inplace=True)]
            elif v == 'Softmax':
                layer += [nn.Softmax(dim=1)]
            else:
                layer += [nn.Linear(input_size, v)]
                input_size = v

        self.fc = nn.Sequential(*layer)
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 브레이크아웃에서의 A3CAgent 클래스(글로벌신경망)
class A3CAgent:
    def __init__(self):
        global render
        self.render = render
        render = False
        
        self.actor, self.critic = self.build_model()
        
        self.actor.load_state_dict(torch.load('./save_model/breakout_a3c_actor.pth', map_location=torch.device('cpu')))

    # 정책신경망과 가치신경망을 생성
    def build_model(self):
        actor = Net(actor_cfg).to(device)
        critic = Net(critic_cfg).to(device)
        return actor, critic

    def run(self):
        print("start", threading.currentThread().getName())
        env = gym.make('BreakoutDeterministic-v4')

        step = 0

        while episode < EPISODES:
            done = False
            dead = False

            score, start_life = 0, 5
            observe = env.reset()
            next_observe = observe

            # 0~30 상태동안 정지
            for _ in range(random.randint(1, 30)):
                observe = next_observe
                next_observe, _, _, _ = env.step(1)

            state = pre_processing(next_observe, observe)
            history = torch.cat([state, state, state, state], dim=0).view([1, 4, 84, 84])

            while not done:
                if self.render:
                    env.render()
                step += 1
                observe = next_observe
                action = self.get_action(history)
                # 1: 정지, 2: 왼쪽, 3: 오른쪽
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                # 죽었을 때 시작하기 위해 발사 행동을 함
                if dead:
                    action = 0
                    real_action = 1
                    dead = False

                # 선택한 행동으로 한 스텝을 실행
                next_observe, reward, done, info = env.step(real_action)

                # 각 타임스텝마다 상태 전처리
                next_state = pre_processing(next_observe, observe).view([1, 1, 84, 84])
                next_history = torch.cat([next_state, history[:,:3,:,:]], dim=1)

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1, 1)

                if dead:
                    history = torch.cat(
                        [next_state, next_state, next_state, next_state], dim=0).view([1, 4, 84, 84])
                else:
                    history = next_history

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택
    def get_action(self, state):
        #print("action: ", self.local_actor(state), self.local_actor(state)[0].multinomial(1).item())
        out = self.actor(state / 255.)[0]
        print(out)
        return out.multinomial(1).item()


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    processed_observe = transforms.functional.to_tensor(processed_observe).to(device)
    return processed_observe


if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.run()

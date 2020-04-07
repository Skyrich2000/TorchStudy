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
        # 상태크기와 행동크기를 갖고옴
        self.state_size = 4
        self.action_size = 2
        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        # 쓰레드의 갯수
        self.threads = 16
        self.scores, self.episodes = [], []

        # 정책신경망과 가치신경망을 생성
        self.actor, self.critic = self.build_model()
        # 정책신경망과 가치신경망을 업데이트하는 함수 생성
        self.upgrades = [self.actor_upgrade, self.critic_upgrade]
        self.critic_loss = nn.MSELoss().to(device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)

        self.actor.load_state_dict(torch.load(
            './save_model/breakout_a3c_actor.pth'))
        self.critic.load_state_dict(torch.load(
            "./save_model/breakout_a3c_critic.pth"))

    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):
        # 쓰레드 수만큼 Agent 클래스 생성
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic],
                        self.upgrades, self.discount_factor, self.print_)
                  for _ in range(self.threads)]

        # 각 쓰레드 시작
        for agent in agents:
            time.sleep(1)
            agent.start()

        #10분(600초)에 한번씩 모델을 저장

    # 정책신경망과 가치신경망을 생성
    def build_model(self):
        actor = Net(actor_cfg).to(device)
        critic = Net(critic_cfg).to(device)
        return actor, critic

    # 정책신경망을 업데이트하는 함수
    def actor_upgrade(self, state, action, advantage):
        policy = self.actor(state)
        action_prob = torch.zeros([len(state), 1]).to(device)
        for i in range(len(state)):
            action_prob[i] = policy[i][action[i]]

        loss1 = (-torch.log(action_prob) * advantage).sum()

        loss2 = (policy * torch.log(policy + 1e-10)
                 ).sum(dim=1).sum()  # 왜 dim=1 ??

        loss = loss1 + 0.01 * loss2

        with lock:
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

    # 가치신경망을 업데이트하는 함수
    def critic_upgrade(self, state, target):
        value = self.critic(state)
        target = torch.autograd.Variable(target).to(device)

        with lock:
            self.critic_optimizer.zero_grad()
            self.critic_loss(value, target).backward()
            self.critic_optimizer.step()

    def print_(self, score, step, name):
        # 에피소드마다 학습 결과 출력
        with lock:
            global episode
            episode += 1
            if not episode % 100:
                self.scores.append(score)
                self.episodes.append(episode)
                pylab.plot(self.episodes, self.scores, 'b')
                pylab.savefig("./save_graph/breakout_a3c.png")
                torch.save(self.actor.state_dict(),
                           "./save_model/breakout_a3c_actor.pth")
                torch.save(self.critic.state_dict(),
                           "./save_model/breakout_a3c_critic.pth")
            print("episode:", episode, "  score:",
                  score, "  step:", step, " by", name)

# 액터러너 클래스(쓰레드)


class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model,
                 upgrades, discount_factor, show):
        threading.Thread.__init__(self)

        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.upgrades = upgrades
        self.discount_factor = discount_factor
        self.show = show

        global render
        self.render = render
        render = False

        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards = torch.Tensor().to(device), [], []

        # 로컬 모델 생성
        self.local_actor, self.local_critic = self.build_local_model()
        self.update_local_model()

        # 모델 업데이트 주기
        self.t_max = 20
        self.t = 0

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
            history = torch.cat([state, state, state, state],
                                dim=0).view([1, 4, 84, 84])

            while not done:
                if self.render:
                    env.render()
                step += 1
                self.t += 1
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
                next_state = pre_processing(
                    next_observe, observe).view([1, 1, 84, 84])
                next_history = torch.cat(
                    [next_state, history[:, :3, :, :]], dim=1)

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1, 1)

                # 샘플을 저장
                self.append_sample(history, action, reward)

                if dead:
                    history = torch.cat(
                        [next_state, next_state, next_state, next_state], dim=0).view([1, 4, 84, 84])
                else:
                    history = next_history

                # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 0

                if done:
                    self.show(score, step, threading.currentThread().getName())
                    step = 0

    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = torch.zeros(len(rewards)).to(device)
        running_add = 0

        if not done:
            running_add = self.local_critic(
                self.states[-1].view(1, 4, 84, 84))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add

        return discounted_prediction

    # 정책신경망과 가치신경망을 업데이트
    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(
            self.rewards, done).view(-1, 1)

        states = self.states / 255.
        values = self.local_critic(states).detach()

        advantages = discounted_prediction - values

        self.upgrades[0](states, self.actions, advantages)
        self.upgrades[1](states, discounted_prediction)
        self.states, self.actions, self.rewards = torch.Tensor().to(device), [], []

    # 로컬신경망을 생성하는 함수
    def build_local_model(self):
        actor = Net(actor_cfg).to(device)
        critic = Net(critic_cfg).to(device)
        return actor, critic

    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_actor.load_state_dict(self.actor.state_dict())
        self.local_critic.load_state_dict(self.critic.state_dict())

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택
    def get_action(self, state):
        #print("action: ", self.local_actor(state), self.local_actor(state)[0].multinomial(1).item())
        return self.local_actor(state / 255.)[0].multinomial(1).item()

    # 샘플을 저장
    def append_sample(self, state, action, reward):
        self.states = torch.cat([self.states, state])
        self.actions.append(action)
        self.rewards.append(reward)


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    processed_observe = transforms.functional.to_tensor(
        processed_observe).to(device)
    return processed_observe


if __name__ == "__main__":
    global_agent = A3CAgent()
    global_agent.train()

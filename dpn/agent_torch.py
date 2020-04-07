import sys
import gym
import pylab
import random
import numpy as np
import time
from collections import deque

import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc2_1 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.initweight()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_1(x))
        x = self.fc3(x)
        return x
    
    def initweight(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight).to(device)
        torch.nn.init.xavier_uniform_(self.fc2.weight).to(device)
        torch.nn.init.xavier_uniform_(self.fc2_1.weight).to(device)
        torch.nn.init.xavier_uniform_(self.fc3.weight).to(device)

EPISODES = 500
# 카트폴 예제에서의 DQN 에이전트


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = True
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        self.batch_size = 500
        self.train_start = 2000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=3000)

        # 모델과 타깃 모델 생성
        self.model = Net(self.state_size, self.action_size).to(device)
        self.target_model = Net(self.state_size, self.action_size).to(device)

        self.criterion = nn.MSELoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 타깃 모델 초기화
        self.update_target_model()

        #if self.load_model:
        #    self.model.load_weights("./save_model/cartpole_dqn_trained.h5")

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        #weight.data 는 뭐지...

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                # 모델로부터 행동 산출
                state = torch.FloatTensor(state).to(device)
                return self.model(state)[0].argmax()

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        output = self.model(states).gather(1, actions)
        target = self.target_model(next_states).max(1)[0].detach()
        target = rewards + self.discount_factor * dones * target

        self.optimizer.zero_grad()
        loss = self.criterion(output.squeeze(), target)
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    scores, episodes, steps = [], [], []

    for e in range(EPISODES):
        done = False
        score = 0
        step_size = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            # 현재 상태로 행동을 선택
            action = agent.get_action(state).item()
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            #reward = (-abs(state[0][2])*100 + 5)
            # 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or step_size >= 499 else -100

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, not done)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                if agent.render:
                    env.render()
                agent.train_model()

            score += reward
            state = next_state
            step_size += 1

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                #score = score if score == 500 else score + 100
                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                steps.append(step_size)
                pylab.plot(episodes, steps, 'b')
                pylab.savefig("./save_graph/cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon, " step:", step_size)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                #if np.mean(scores[-min(10, len(scores)):]) > 490:
                #    agent.model.save_weights("./save_model/cartpole_dqn.h5")
                #    sys.exit()

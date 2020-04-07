import sys
import gym
import pylab
import numpy as np

import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

EPISODES = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            else :
                layer += [nn.Linear(input_size, v)]
                input_size = v

        self.net = nn.Sequential(*layer)
        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # 정책신경망과 가치신경망 생성
        self.actor = Net(self.state_size, [24, 'ReLU', self.action_size, 'Softmax']).to(device)
        self.critic = Net(self.state_size, [24, 'ReLU', 24, 'ReLU', self.value_size]).to(device)
        self.critic_loss = nn.MSELoss().to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_actor_trained.h5")
            self.critic.load_weights("./save_model/cartpole_critic_trained.h5")

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state)[0].multinomial(1).item()

    # 정책신경망을 업데이트하는 함수
    def actor_upgrade(self, data):
        state = data[0]
        action = data[1]
        advantage = data[2]

        policy = self.actor(state)[0]
        loss = -torch.log(policy[action]) * advantage
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    # 가치신경망을 업데이트하는 함수
    def critic_upgrade(self, data):
        value = data[0]
        target = data[1]
        self.critic_optimizer.zero_grad()
        self.critic_loss(value, target).backward()
        self.critic_optimizer.step()

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        done = torch.FloatTensor([done]).to(device)

        value = self.critic(state)[0]
        next_value = self.critic(next_state)[0].detach()
        
        advantage = (reward + self.discount_factor * done * next_value) - value.detach()
        target = reward + self.discount_factor * done * next_value

        # if done:
        #     advantage = reward - value
        #     target = [reward]
        # else:
        #     advantage = (reward + self.discount_factor * next_value) - value
        #     target = reward + self.discount_factor * next_value

        self.actor_upgrade([state, action, advantage])
        self.critic_upgrade([value, target])

if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, not done)

            score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_a2c.png")
                print("episode:", e, "  score:", score)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                # if np.mean(scores[-min(10, len(scores)):]) > 490:
                #     agent.actor.save_weights("./save_model/cartpole_actor.h5")
                #     agent.critic.save_weights(
                #         "./save_model/cartpole_critic.h5")
                #     sys.exit()

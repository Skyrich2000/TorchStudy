import torch
import torch.optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W], lr=0.01)

nb_epochs = 1000
for epoch in range(1, nb_epochs + 1):
    hypothesis = x_train * W 
    cost = torch.mean((hypothesis - y_train) ** 2) #loss 함수의 기울기를 계산해서 학습가능한 변수들을 개선시키는것.

    optimizer.zero_grad()  # 그레디언트 초기화
    cost.backward()  # 그레디언트 계산
    optimizer.step()  # 개선

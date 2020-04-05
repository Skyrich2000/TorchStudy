import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        # 데이터
        self.x_data =[[73, 80, 75],
                [93, 88, 93],
                [89, 91, 90],
                [96, 98, 100],
                [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x_data[index])
        y = torch.FloatTensor(self.y_data[index])

        return x, y

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


dataloader = DataLoader(CustomDataset(), batch_size=2, shuffle=True)

# 모델 초기화
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 3000
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 20번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

import torch

x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x)
print(x.size())
result = torch.Tensor(5, 3)
torch.add(x + y, out=result)

print(result)

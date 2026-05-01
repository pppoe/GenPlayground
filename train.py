import torch
from torch.utils.data import DataLoader, TensorDataset
from gen import PointFlowModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load('mnist_2d_pc.pt')
data = ckpt['data']                                          # [10000, 128, 2]
loader = DataLoader(TensorDataset(data), batch_size=256, shuffle=True)

model = PointFlowModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

epochs = 200
for epoch in range(epochs):
    loss_sum = 0.0
    for (x0,) in loader:
        x0 = x0.to(device)                                  # [B, 128, 2]
        x1 = torch.randn_like(x0)                           # noise
        t  = torch.rand(x1.shape[0], device=device)         # [B]
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        pred_vel = model(xt, t)
        gt_vel = (x1 - x0)
        loss = (pred_vel - gt_vel).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    print(f'epoch {epoch+1:>{len(str(epochs))}}  loss {loss_sum / len(loader):.4f}')

torch.save(model.state_dict(), 'model.pt')
print('saved model.pt')

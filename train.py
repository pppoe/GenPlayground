import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from gen import PointFlowModel

parser = argparse.ArgumentParser()
parser.add_argument('out_dir', help='folder to create and save outputs into')
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load('digit_fonts_pc.pt')
data = ckpt['data']                                          # [N, 128, 2]
loader = DataLoader(TensorDataset(data), batch_size=1024, shuffle=True)

model = PointFlowModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

epochs = 100
log = []
train_start = time.time()

epoch_bar = tqdm(range(1, epochs + 1), desc='training', unit='epoch')
for epoch in epoch_bar:
    epoch_start = time.time()
    loss_sum = 0.0

    for (x0,) in loader:
        x0 = x0.to(device)
        x1 = torch.randn_like(x0)
        t  = torch.rand(x1.shape[0], device=device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        pred_vel = model(xt, t)
        gt_vel = (x1 - x0)
        loss = (pred_vel - gt_vel).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()

    epoch_loss = loss_sum / len(loader)
    epoch_time = time.time() - epoch_start
    elapsed    = time.time() - train_start

    epoch_bar.set_postfix(loss=f'{epoch_loss:.4f}', epoch_s=f'{epoch_time:.1f}s')
    log.append({'epoch': epoch, 'loss': epoch_loss,
                'epoch_time_s': round(epoch_time, 2),
                'elapsed_s': round(elapsed, 2)})

total_time = time.time() - train_start
print(f'total training time: {total_time:.1f}s')

torch.save(model.state_dict(), out_dir / 'model.pt')
print(f'saved {out_dir}/model.pt')

with open(out_dir / 'train_log.json', 'w') as f:
    json.dump({'total_time_s': round(total_time, 2), 'epochs': log}, f, indent=2)
print(f'saved {out_dir}/train_log.json')

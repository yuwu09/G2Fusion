import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR

class IRVSDataset(Dataset):
    def __init__(self, ir_dir, vis_dir, size=256):
        self.ir_list = sorted([os.path.join(ir_dir, f) for f in os.listdir(ir_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.vis_list = sorted([os.path.join(vis_dir, f) for f in os.listdir(vis_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        assert len(self.ir_list) == len(self.vis_list), f"Image count mismatch: IR={len(self.ir_list)}, VIS={len(self.vis_list)}"
        self.transform = self.make_transform(size)

    def make_transform(self, resize_size: int = 256):
        return v2.Compose([
            v2.ToImage(),
            v2.Resize((resize_size, resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.ir_list)

    def __getitem__(self, idx):
        ir_img = Image.open(self.ir_list[idx]).convert('RGB')
        vis_img = Image.open(self.vis_list[idx]).convert('RGB')
        return self.transform(ir_img), self.transform(vis_img)


class DINOv3Backbone(nn.Module):
    def __init__(self, model_dir, train_last_blocks=2):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        for param in self.model.parameters():
            param.requires_grad = False
        if hasattr(self.model, 'layer'):
            layers = self.model.layer
            total_layers = len(layers)
            for i in range(total_layers - train_last_blocks, total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
        else:
            raise AttributeError(" self.model.layer not found")

    def forward(self, x):
        out = self.model(x) # x: [B, 3, H, W]
        last_hidden_state = out.last_hidden_state  
        patch_tokens = last_hidden_state[:, 5:, :]  # [B, 196, 768]
        B, N, D = patch_tokens.shape
        H_feat = x.shape[2] // 16
        W_feat = x.shape[3] // 16
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)
        return patch_tokens

def alignment_loss(ir_feat, vis_feat):
    ir_norm = F.normalize(ir_feat, dim=1)
    vis_norm = F.normalize(vis_feat, dim=1)
    cos_sim = (ir_norm * vis_norm).sum(dim=1)  # [B, H, W]
    return 1.0 - cos_sim.mean()

def train_dinov3(ir_dir, vis_dir, model_dir, save_dir, epochs=10, batch_size=8, lr=2e-5, device='cuda'):
    dataset = IRVSDataset(ir_dir, vis_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = DINOv3Backbone(model_dir, train_last_blocks=2).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler()
    os.makedirs(save_dir, exist_ok=True)
    print(f"Starting fine-tuning: Epochs={epochs}, BatchSize={batch_size}, LR={lr}")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for ir, vis in dataloader:
            ir, vis = ir.to(device), vis.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device):
                ir_f = model(ir)
                vis_f = model(vis)
                loss = alignment_loss(ir_f, vis_f)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.7f}")
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(save_dir, f'dinov3_aligned_ep{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune the last DINOv3 blocks for IR-visible alignment.')
    parser.add_argument('--ir-dir', required=True)
    parser.add_argument('--vis-dir', required=True)
    parser.add_argument('--model-dir', required=True, help='Local Transformers DINOv3 model directory with weights.')
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train_dinov3(args.ir_dir, args.vis_dir, args.model_dir, args.save_dir,
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)

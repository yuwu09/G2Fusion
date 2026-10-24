import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

class DINOv3(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # if hasattr(self.model, 'layer'):
        #     layers = self.model.layer
        #     total_layers = len(layers)
        #     for i in range(total_layers - train_last_blocks, total_layers):
        #         for param in layers[i].parameters():
        #             param.requires_grad = True
        # else:
        #     raise AttributeError(" self.model.layer not found")

    def forward(self, x):
        out = self.model(x) # x: [B, 3, H, W]
        last_hidden_state = out.last_hidden_state  
        patch_tokens = last_hidden_state[:, 5:, :]  # [B, 196, 768]
        B, N, D = patch_tokens.shape
        H_feat = x.shape[2] // 16
        W_feat = x.shape[3] // 16
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)
        return patch_tokens
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys


class gradientloss(nn.Module):
    def __init__(self):
        super(gradientloss,self).__init__()
        self.AP5 = nn.AvgPool2d(5,stride=1,padding=2).cuda()
        self.MP5 = nn.MaxPool2d(5,stride=1,padding=2).cuda()
    def forward(self,img1,img2,mask=1,eps=1e-2):
        #img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
        mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
        mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
        mean_ = mean_.detach()/2
        std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
        std_ = std_.detach()/2 
        img1 = (img1-mean_)/std_
        img2 = (img2-mean_)/std_
        grad1 = KF.spatial_gradient(img1,order=2)
        grad2 = KF.spatial_gradient(img2,order=2)
        mask = mask.unsqueeze(1)
        # grad1 = self.AP5(self.MP5(grad1))
        # grad2 = self.AP5(self.MP5(grad2))
        # print((grad1-grad2).abs().mean())
        l = (((grad1-grad2)+(grad1-grad2).pow(2)*10)*mask).abs().clamp(min=eps).mean()
        #l = l[...,5:-5,10:-10].mean()
        return l

class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()


class Pur_loss(nn.Module):
    def __init__(self,loss_type='l1'):
        super(Pur_loss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, fea_pur, fea_GT, image_recon, image_GT):
        loss_fea=self.loss_fn(fea_pur, fea_GT)
        loss_image=self.loss_fn(image_recon, image_GT)

        total_loss = loss_fea + loss_image
        
        return total_loss, loss_fea, loss_image
        
class Fusion_loss(nn.Module):
    def __init__(self,loss_type='l1'):
        super(Fusion_loss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, X_GT_Img, Fuse_Img, P_Fea=None, P_Img=None):
    
        X_GT_Img_y, X_GT_Img_cbcr = self.rgb_to_y(X_GT_Img)
        Fuse_Img_y, Fuse_Img_cbcr = self.rgb_to_y(Fuse_Img)
        loss_color=self.loss_fn(Fuse_Img_cbcr, X_GT_Img_cbcr)
        total_loss = 10 * loss_color

        Fusion_adv_loss_fea = torch.tensor(0.0, device=X_GT_Img.device)
        Fusion_adv_loss_img = torch.tensor(0.0, device=X_GT_Img.device)

        if P_Fea is not None:
            with torch.no_grad():
                target_fea_0 = torch.empty_like(P_Fea[:, 0]).uniform_(0.8, 1.2)
                target_fea_1 = torch.empty_like(P_Fea[:, 1]).uniform_(0.8, 1.2)      
            Fusion_adv_loss_fea=self.loss_fn(P_Fea[:, 0], target_fea_0) + self.loss_fn(P_Fea[:, 1], target_fea_1)
            total_loss += Fusion_adv_loss_fea

        if P_Img is not None:
            with torch.no_grad():
                target_img_0 = torch.empty_like(P_Img[:, 0]).uniform_(0.8, 1.2)
                target_img_1 = torch.empty_like(P_Img[:, 1]).uniform_(0.8, 1.2)
            Fusion_adv_loss_img=self.loss_fn(P_Img[:, 0], target_img_0) + self.loss_fn(P_Img[:, 1], target_img_1)
            total_loss += Fusion_adv_loss_img

        return total_loss, loss_color, Fusion_adv_loss_fea, Fusion_adv_loss_img
        
    def rgb_to_y(self, image):
        r = image[:, 0:1, :, :]
        g = image[:, 1:2, :, :]
        b = image[:, 2:3, :, :]
    
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - y)
        cr = 0.713 * (r - y)
        cbcr = torch.cat((cb, cr), dim=1)
        
        return y, cbcr       


        
        
        
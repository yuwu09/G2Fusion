import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import lpips
import numpy as np
import sys
from math import exp
from torch.autograd import Variable
import kornia.utils as KU
import kornia.filters as KF



# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=1):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    #print(mask.shape,ssim_map.shape)
    ssim_map = ssim_map*mask

    ssim_map = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def sigma_transformation(x):
    return torch.exp(2 * x) -1

def Contrast(img1, img2, window_size=11, channel=1, eps=1e-6):
    ## img1 is the infrared image, img2 is the visible image
    window = create_window(window_size, channel)    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq 
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    
    sigma1 = sigma1_sq / (2 * (sigma1_sq + sigma2_sq) + eps)
    sigma2 = 1 - sigma1
    return sigma1, sigma2

    
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=1):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel
        mask = torch.logical_and(img1>0,img2>0).float()
        for i in range(self.window_size//2):
            mask = (F.conv2d(mask, window, padding=self.window_size//2, groups=channel)>0.8).float()
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average, mask=mask)







def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    return sobelx, sobely


class GradLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target):


        predict_1 = torch.unsqueeze(predict[:,0,:,:],1)
        predict_2 = torch.unsqueeze(predict[:,1,:,:],1)
        predict_3 = torch.unsqueeze(predict[:,2,:,:],1)

        target_1 = torch.unsqueeze(target[:,0,:,:],1)
        target_2 = torch.unsqueeze(target[:,1,:,:],1)
        target_3 = torch.unsqueeze(target[:,2,:,:],1)
        
        predict_1_grad_x, predict_1_grad_y =   Sobelxy(predict_1)
        predict_2_grad_x, predict_2_grad_y =   Sobelxy(predict_2)
        predict_3_grad_x, predict_3_grad_y =   Sobelxy(predict_3)
        predict_grad_x = torch.cat([predict_1_grad_x, predict_2_grad_x, predict_3_grad_x], 1)
        predict_grad_y = torch.cat([predict_1_grad_y, predict_2_grad_y, predict_3_grad_y], 1)        


        target_1_grad_x, target_1_grad_y =   Sobelxy(target_1)
        target_2_grad_x, target_2_grad_y =   Sobelxy(target_2)
        target_3_grad_x, target_3_grad_y =   Sobelxy(target_3)
        target_grad_x = torch.cat([target_1_grad_x, target_2_grad_x, target_3_grad_x], 1)
        target_grad_y = torch.cat([target_1_grad_y, target_2_grad_y, target_3_grad_y], 1)               

        loss = self.loss_fn(predict_grad_x, target_grad_x) + self.loss_fn(predict_grad_y, target_grad_y)

        return loss.mean()



class NCCLoss(nn.Module):
    def __init__(self):
        super(NCCLoss, self).__init__()

    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J
        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)
        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        return I_var, J_var, cross

    def forward(self, I, J, win=[15]):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims
        sum_filt = torch.ones([1, I.shape[1], *win]).cuda()/I.shape[1]
        pad_no = math.floor(win[0] / 2)
        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)
        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)
        cc = cross * cross / ((I_var * J_var).clamp(min=1e-3) + 1e-3)
        return -1 * torch.mean(cc)


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





class PerceptualMatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted
        self.lpips_fn = lpips.LPIPS(net='vgg') #.to(device)

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        ploss = self.lpips_fn(predict.contiguous(), target.contiguous(), normalize=True)

        if self.is_weighted and weights is not None:
            loss = weights * loss

        loss = loss + ploss * 0.01
        return loss.mean()

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
    
def l1loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    B = img1.shape[0]
    sigma = torch.ones(B, 2, device=img1.device)
    img1 = KF.gaussian_blur2d(img1,[3,3],sigma)*mask_
    img2 = KF.gaussian_blur2d(img2,[3,3],sigma)*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).mean()

def l2loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    B = img1.shape[0]
    sigma = torch.ones(B, 2, device=img1.device)
    img1 = KF.gaussian_blur2d(img1,[3,3],sigma)*mask_
    img2 = KF.gaussian_blur2d(img2,[3,3],sigma)*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).pow(2).mean()

def norm(tensor):
    return tensor * 2 - 1

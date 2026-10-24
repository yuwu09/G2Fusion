import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools
import kornia.utils as KU
import kornia.filters as KF
from .module_util import (
 Conv2d, ResConv2d
)

class Feature_extractor_unshare(nn.Module):
    def __init__(self,depth,base_ic,base_oc,base_dilation,norm):
        super(Feature_extractor_unshare,self).__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i%2==1:
                dilation *= 2
            if ic == oc:
                feature_extractor.append(ResConv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            else:
                feature_extractor.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            if i%2==1 and i<depth-1:
                oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x
    
    
class DispRefiner(nn.Module):
    def __init__(self,channel,dilation=1,depth=4):
        super(DispRefiner,self).__init__()
        self.preprocessor = nn.Sequential(Conv2d(channel,channel,3,dilation=dilation,padding=dilation,norm=None,act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,norm=None,act=None))
        oc = channel
        ic = channel+2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator)
    def forward(self,feat1,feat2,disp):
        
        b=feat1.shape[0]
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b],feat[b:]],dim=1))
        corr = torch.cat([feat,disp],dim=1)
        delta_disp = self.estimator(corr)
        disp = disp+delta_disp
        return disp 
    
class DispEstimator(nn.Module):
    def __init__(self,channel,depth=4,norm=nn.BatchNorm2d,dilation=1):
        super(DispEstimator,self).__init__()
        estimator = nn.ModuleList([])
        self.corrks = 7
        self.preprocessor = Conv2d(channel,channel,3,act=None,norm=None,dilation=dilation,padding=dilation)
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,act=None))
        #self.localcorrpropcessor = nn.Sequential(Conv2d(self.corrks**2,32,3,padding=1,bias=True,norm=None),
        #                                         Conv2d(32,2,3,padding=1,bias=True,norm=None),)
        oc = channel
        ic = channel+self.corrks**2
        dilation = 1
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.layers = estimator
        self.scale = torch.FloatTensor([256,256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
        #self.corrpropcessor = Conv2d(9+channel,channel,3,padding=1,bias=True,norm=nn.InstanceNorm2d)
        #self.AP3=nn.AvgPool2d(3,stride=1,padding=1)

    # def localcorr(self,feat1,feat2):
    #     feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
    #     feat1 = F.normalize(feat1,dim=1)
    #     feat2 = F.normalize(feat2,dim=1)
    #     b,c,h,w = feat2.shape
    #     feat2_smooth = KF.gaussian_blur2d(feat2,[9,9],[3,3])
    #     feat2_loc_blk = F.unfold(feat2_smooth,kernel_size=self.corrks,dilation=4,padding=4*(self.corrks-1)//2,stride=1).reshape(b,c,-1,h,w)
    #     localcorr = (feat1.unsqueeze(2)*feat2_loc_blk).sum(dim=1)
    #     localcorr = self.localcorrpropcessor(localcorr)
    #     corr = torch.cat([feat,localcorr],dim=1)
    #     return corr
    def localcorr(self,feat1,feat2):
        feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
        b,c,h,w = feat2.shape
        feat1_smooth = KF.gaussian_blur2d(feat1,(13,13),(3,3),border_type='constant')
        feat1_loc_blk = F.unfold(feat1_smooth,kernel_size=self.corrks,dilation=4,padding=2*(self.corrks-1),stride=1).reshape(b,c,-1,h,w)
        localcorr = (feat2.unsqueeze(2)-feat1_loc_blk).pow(2).mean(dim=1)
        corr = torch.cat([feat,localcorr],dim=1)
        return corr

    def forward(self,feat1,feat2):
        b,c,h,w = feat1.shape
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat1 = feat[:b]
        feat2 = feat[b:]
        if self.scale[0,1,0,0] != w-1 or self.scale[0,0,0,0] != h-1:
            self.scale = torch.FloatTensor([w,h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
            self.scale = self.scale.to(feat1.device)
        corr = self.localcorr(feat1,feat2)
        for i,layer in enumerate(self.layers):
            corr = layer(corr)
        corr = KF.gaussian_blur2d(corr,(13,13),(3,3),border_type='replicate')
        disp = corr.clamp(min=-300,max=300)
        # print(disp.shape)
        # print(feat1.shape)
        return disp/self.scale
    
class DenseMatcher(nn.Module):
    def __init__(self, unshare_depth=4, matcher_depth=4, in_channels=3):
        super(DenseMatcher, self).__init__()

        # -------------------------
        # 1. Feature extraction layers
        # -------------------------
        self.dino_reduce = nn.Conv2d(768, 64, 1, bias=False)

        self.feature_extractor_unshare1 = Feature_extractor_unshare(
            depth=unshare_depth, base_ic=in_channels, base_oc=8, base_dilation=1, norm=nn.InstanceNorm2d
        )
        self.feature_extractor_unshare2 = Feature_extractor_unshare(
            depth=unshare_depth, base_ic=in_channels, base_oc=8, base_dilation=1, norm=nn.InstanceNorm2d
        )
        base_oc = self.feature_extractor_unshare1.oc

        self.feature_extractor_share1 = nn.Sequential(
            Conv2d(base_oc, base_oc*2, 3, 1, 1, 1, norm=nn.InstanceNorm2d),
            Conv2d(base_oc*2, base_oc*2, 3, 2, 1, 1, norm=nn.InstanceNorm2d)
        )
        self.feature_extractor_share2 = nn.Sequential(
            Conv2d(base_oc*2, base_oc*4, 3, 1, 2, 2, norm=nn.InstanceNorm2d),
            Conv2d(base_oc*4, base_oc*4, 3, 2, 2, 2, norm=nn.InstanceNorm2d)
        )
        self.feature_extractor_share3 = nn.Sequential(
            Conv2d(base_oc*4, base_oc*8, 3, 1, 4, 4, norm=nn.InstanceNorm2d),
            Conv2d(base_oc*8, base_oc*8, 3, 2, 4, 4, norm=nn.InstanceNorm2d)
        )
        self.feature_extractor_share4 = nn.Sequential(
            Conv2d(base_oc*8, base_oc*16, 3, 1, 8, 8, norm=nn.InstanceNorm2d),
            Conv2d(base_oc*16, base_oc*16, 3, 2, 8, 8, norm=nn.InstanceNorm2d)
        )

        # -------------------------
        # 2. Matching layers
        # -------------------------
        self.matcher4 = DispEstimator(64, matcher_depth, dilation=1)          # 1/16 (DINO)
        self.matcher3 = DispEstimator(base_oc*8, matcher_depth, dilation=2)  # 1/8
        self.matcher2 = DispEstimator(base_oc*4, matcher_depth, dilation=4)   # 1/4
        self.matcher1 = DispEstimator(base_oc*2, matcher_depth, dilation=4)   # 1/2
        
        self.refiner = DispRefiner(base_oc*2, 1)

    def warp(self, x, flow):
        """ Resample feature maps using the displacement field. """
        B, C, H, W = x.shape
        grid = KU.create_meshgrid(H, W).to(x.device) # range [-1, 1]
        # grid_sample expects normalized grid coordinates in [-1, 1].
        # The flow must use the same normalized coordinate space.
        v_grid = grid + flow.permute(0, 2, 3, 1)
        return F.grid_sample(x, v_grid, align_corners=True, mode='bilinear', padding_mode='zeros')

    def match(self, feat11, feat12, feat21, feat22, feat31, feat32, feat41, feat42, dino_src=None, dino_tgt=None):
        B = feat11.shape[0]

        # -------------------------
        # Level 4: 1/16 (DINO Coarse)
        # -------------------------
        if dino_src is not None:
            disp4 = self.matcher4(dino_src, dino_tgt)
        else:
            disp4 = torch.zeros(B, 2, feat41.shape[2], feat41.shape[3], device=feat41.device)

        # Check the level-4 displacement range.
        # print(f"[Debug] disp4 range: min={disp4.min().item():.4f}, max={disp4.max().item():.4f}")

        # -------------------------
        # Level 3: 1/8
        # -------------------------
        # Upsample the previous displacement to the current resolution.
        disp4_up = F.interpolate(disp4, size=feat31.shape[-2:], mode='bilinear', align_corners=True)
        # Warp current features using the previous displacement estimate.
        feat31_warped = self.warp(feat31, disp4_up)
        # Predict the residual displacement.
        res3 = self.matcher3(feat31_warped, feat32)
        disp3 = disp4_up + res3

        # -------------------------
        # Level 2: 1/4
        # -------------------------
        disp3_up = F.interpolate(disp3, size=feat21.shape[-2:], mode='bilinear', align_corners=True)
        feat21_warped = self.warp(feat21, disp3_up)
        res2 = self.matcher2(feat21_warped, feat22)
        disp2 = disp3_up + res2

        # -------------------------
        # Level 1: 1/2
        # -------------------------
        disp2_up = F.interpolate(disp2, size=feat11.shape[-2:], mode='bilinear', align_corners=True)
        feat11_warped = self.warp(feat11, disp2_up)
        res1 = self.matcher1(feat11_warped, feat12)
        disp1 = disp2_up + res1

        # Check the final displacement range.
        # print(f"[Debug] Final disp1 shape: {disp1.shape}, range: [{disp1.min().item():.2f}, {disp1.max().item():.2f}]")

        # -------------------------
        # Refinement & Output
        # -------------------------
        # Convert the displacement to pixel coordinates for the refiner.
        scale = torch.tensor([feat11.shape[3], feat11.shape[2]], device=feat11.device).view(1, 2, 1, 1) - 1
        disp_pixel = disp1 * scale

        # Refine the warped feat11 features against feat12.
        refined_disp_pixel = self.refiner(feat11_warped, feat12, disp_pixel)

        # Return intermediate levels during training for loss computation.
        if self.training:
            return refined_disp_pixel / scale, disp1, disp2

        # Apply Gaussian smoothing during inference.
        final_disp = KF.gaussian_blur2d(refined_disp_pixel, (17, 17), (5, 5), border_type='replicate') / scale
        return final_disp, None, None

    def forward(self, src, tgt, dino_src=None, dino_tgt=None):
        b, c, h, w = src.shape

        # 1. Extract features.
        feat01 = self.feature_extractor_unshare1(src)
        feat02 = self.feature_extractor_unshare2(tgt)
        
        # 2. Build feature pyramids using a combined batch.
        feat_cat = torch.cat([feat01, feat02], dim=0)
        feat1 = self.feature_extractor_share1(feat_cat)
        feat2 = self.feature_extractor_share2(feat1)
        feat3 = self.feature_extractor_share3(feat2)
        feat4 = self.feature_extractor_share4(feat3)

        feat11, feat12 = feat1[:b], feat1[b:]
        feat21, feat22 = feat2[:b], feat2[b:]
        feat31, feat32 = feat3[:b], feat3[b:]
        feat41, feat42 = feat4[:b], feat4[b:]

        # 3. Reduce DINO feature channels and resize spatial dimensions.
        if dino_src is not None:
            d_src = self.dino_reduce(dino_src)
            d_tgt = self.dino_reduce(dino_tgt)
            dino_src = F.interpolate(d_src, size=feat41.shape[-2:], mode='bilinear', align_corners=True)
            dino_tgt = F.interpolate(d_tgt, size=feat41.shape[-2:], mode='bilinear', align_corners=True)

        # 4. Match features.
        disp, disp_coarse, _ = self.match(
            feat11, feat12, feat21, feat22, feat31, feat32, feat41, feat42, dino_src, dino_tgt
        )

        # 5. Upsample to the original image size.
        return F.interpolate(disp, size=(h, w), mode='bilinear', align_corners=True)
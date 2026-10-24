import logging
from collections import OrderedDict
import os
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
import kornia.utils as KU
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion
from models.modules.loss import MatchingLoss, NCCLoss,gradientloss,l1loss,l2loss
import torch.nn.functional as F
from .base_model import BaseModel
import kornia.filters as KF

logger = logging.getLogger("base")


class match_model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.AP = nn.AvgPool2d(5, stride=1, padding=2)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_Match(opt).to(self.device)
        self.dino_model = networks.define_dino(opt).to(self.device)

        for param in self.dino_model.parameters():
            param.requires_grad = False
            
        if opt["dist"]:
            self.model = DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )
        # else:
        #     self.model = DataParallel(self.model)
        # print network
        # self.print_network()
        self.load()

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device) #l1
            self.ncc_loss_fn = NCCLoss().to(self.device)
            self.gradientloss = gradientloss().to(self.device)
            # self.grad_loss_fn = GradLoss(loss_type).to(self.device)
            # self.ssim_loss_fn = SSIMLoss().to(self.device)
            # self.loss_fn = PerceptualMatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']
            # optimizers
            wd_AE = train_opt["weight_decay_AE"] if train_opt["weight_decay_AE"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_AE"],
                    weight_decay=wd_AE,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_AE"],
                    weight_decay=wd_AE,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_AE"],
                    weight_decay=wd_AE,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_AE"],
                    weight_decay=wd_AE,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            # self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()
            


    def generate_mask(self):
        B, C, H, W = self.X.shape
        self.border_mask = torch.zeros([1, 1, H, W])
        self.border_mask[:, :, 10:-10, 10:-10] = 1
        self.grid = KU.create_meshgrid(H, W).to(self.device)
        flow = self.grid + self.flow_gt
        goodmask = torch.logical_and(flow >= -1, flow <= 1)
        if self.border_mask.device != goodmask.device:
            self.border_mask = self.border_mask.to(goodmask.device)
        self.goodmask = torch.logical_and(goodmask[..., 0], goodmask[..., 1]).unsqueeze(1) * 1.0
        for _ in range(2):
            self.goodmask = (self.AP(self.goodmask) > 0.3).float()

        flow = self.grid - self.flow_gt
        goodmask = F.grid_sample(self.goodmask, flow)
        self.goodmask_inverse = goodmask

    def feed_data(self,X,Y,fea_X, fea_Y, flow_gt, X_GT):
        self.X = X .to(self.device)
        self.Y = Y. to(self.device)
        self.fea_X = fea_X.to(self.device)  
        self.fea_Y = fea_Y.to(self.device)
        self.flow_gt = flow_gt.to(self.device) #[B,H,W,2]
        self.X_GT = X_GT.to(self.device)
        
    def optimize_parameters(self, step):
        self.optimizer.zero_grad()
        self.generate_mask()
        if self.opt["dist"]:
            flow = self.model.module
        else:
            flow = self.model
        
        flow_pred = flow(
            src=self.X,      #vi_wrap    
            tgt=self.Y,      #ir_clean  
            dino_src= self.fea_X , 
            dino_tgt = self.fea_Y ,
        ).permute(0, 2, 3, 1)#[B,2,H,W] -> [B,H,W,2]
        X_rec = self.warp_image_with_flow(self.X, flow_pred)
        loss_img = self.imgloss(X_rec, self.X_GT, self.goodmask * self.goodmask_inverse)
        loss_ncc = self.ncc_loss_fn(X_rec, self.Y)
        loss_flow = self.weightfiledloss(self.X, self.Y, flow_pred,self.flow_gt)
        loss = loss_flow + loss_img
        loss.backward()
        self.optimizer.step()
        # set log
        self.log_dict["loss_img"] = loss_img.item()
        self.log_dict["loss_flow"] = loss_flow.item()
        self.log_dict["loss_ncc"] = loss_ncc.item()

    def test(self,current_step,idx=0):
        self.model.eval()

        if self.opt["dist"]:
            flow = self.model.module
        else:
            flow = self.model

        with torch.no_grad():
            flow_pred = flow(
                src=self.X,      #vi_wrap    
                tgt=self.Y,      #ir_clean  
                dino_src= self.fea_X , 
                dino_tgt = self.fea_Y ,
            ).permute(0, 2, 3, 1)#[B,2,H,W] -> [B,H,W,2]
            self.X_rec = self.warp_image_with_flow(self.X, flow_pred)

        self.model.train()
        test_folder = './image/'
        if not os.path.exists(test_folder): 
            os.makedirs(test_folder)
        save_dir = os.path.join('image', f'step_{current_step}')
        os.makedirs(save_dir, exist_ok=True)
        batch_size = self.X.shape[0]
        for i in range(batch_size):
            index = idx + i
            tvutils.save_image((self.X_rec[i]* 0.5 + 0.5).data, os.path.join(save_dir, f'{index}_X_rec.png'), normalize=False)
            tvutils.save_image((self.X_GT[i]* 0.5 + 0.5).data, os.path.join(save_dir, f'{index}_X_GT.png'), normalize=False)
            tvutils.save_image((self.X[i]* 0.5 + 0.5).data, os.path.join(save_dir, f'{index}_X.png'), normalize=False)
            

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        # out_dict["feat_X"] = self.feat_X.detach()[0].float().cpu()
        # out_dict["feat_Y"] = self.feat_Y.detach()[0].float().cpu()
        out_dict["X_rec"] = self.X_rec.detach()[0].float().cpu()
        out_dict["X_GT"] = self.X_GT.detach()[0].float().cpu()

        # if need_GT:
        #     out_dict["X_GT"] = self.feat_X.detach()[0].float().cpu()
        #     out_dict["Y_GT"] = self.feat_Y.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network Matcher structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_Match = self.opt["path"]["pretrain_model_Matcher"]
        if load_path_Match is not None:
            logger.info("Loading model for Matcher [{:s}] ...".format(load_path_Match))
            self.load_network(load_path_Match, self.model, self.opt["path"]["strict_load"])

        load_path_Dino = self.opt["path"]["pretrain_model_Dino"]
        if load_path_Dino is not None:
            logger.info("Loading model for Dino [{:s}] ...".format(load_path_Dino))
            self.load_network(load_path_Dino, self.dino_model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "Matcher", iter_label)

    def warp_image_with_flow(self, img, flow):
        B, C, H, W = img.shape
        # flow = flow.permute(0, 2, 3, 1)
        # print("flow min:", flow.min(),"flow max:", flow.max())
        grid = KU.create_meshgrid(H, W).to(img.device) # [1, H, W, 2]
        grid = grid.expand(B, -1, -1, -1)  
        img_warped = F.grid_sample(
            img,
            grid + flow,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        mask = torch.ones_like(img[:, :1, :, :])

        mask_warp = F.grid_sample(
            mask,
            grid + flow,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        valid_mask = (mask_warp > 0.999).float()
        warped = img_warped * valid_mask + (1 - valid_mask) * (-1)
        return warped

    def weightfiledloss(self, ref, tgt, disp, disp_gt):
        disp = disp.permute(0, 3, 1, 2).contiguous()     # [B, 2, H, W]
        disp_gt = disp_gt.permute(0, 3, 1, 2).contiguous()
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        w = (((g_ref + g_tgt)) * 2 + 1) * self.border_mask
        return (w * (1000 * (disp - disp_gt).abs().clamp(min=1e-2).pow(2))).mean()
    
    def imgloss(self, src, tgt, mask=1, weights=[0.1, 0.9]):
        return weights[0] * (l1loss(src, tgt, mask) + l2loss(src, tgt, mask)) + weights[1] * self.gradientloss(src, tgt,mask)
    

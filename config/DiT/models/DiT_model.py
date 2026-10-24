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
from ema_pytorch import EMA
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import Fusion_loss

from .base_model import BaseModel
import torch.nn.functional as F
from copy import deepcopy
from models.modules.diffusion import create_diffusion

logger = logging.getLogger("base")


class DiT_Model(BaseModel):
    def __init__(self, opt):
        super(DiT_Model, self).__init__(opt)
        os.makedirs("image", exist_ok=True)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.ae_model = networks.define_AE(opt).to(self.device)
        self.match_model = networks.define_Match(opt).to(self.device)
        self.dino_model = networks.define_dino(opt).to(self.device)

        # self.diffuion_model=networks.define_Diffusion(opt).to(self.device)
        self.fusion_model = networks.define_Fusion(opt).to(self.device)
        self.ema = None
        # Respace the original 1,000-step schedule for evaluation and validation.
        self.sampling_diffusion = create_diffusion(
            timestep_respacing=str(opt.get("sampling_steps", 20))
        )

        for param in self.ae_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False
        for param in self.match_model.parameters():
            param.requires_grad = False

        if opt["dist"]:
            self.fusion_model = DistributedDataParallel(
                self.fusion_model, device_ids=[torch.cuda.current_device()]
            )

        self.load()

        self.ae_model.eval()
        self.dino_model.eval()
        self.match_model.eval()

        self.encode = self.ae_model.encode
        self.decode = self.ae_model.decode

        if self.is_train:
            self.fusion_model.train()
            # self.loss_fusion_fn = Fusion_loss().to(self.device)

            self.weight = opt["train"]["weight"]

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0

            optim_params_fusion = []
            for (
                k,
                v,
            ) in (
                self.fusion_model.named_parameters()
            ):  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params_fusion.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt["optimizer"] == "Adam":
                self.optimizer_fusion = torch.optim.Adam(
                    optim_params_fusion,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            elif train_opt["optimizer"] == "AdamW":
                self.optimizer_fusion = torch.optim.AdamW(
                    optim_params_fusion,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            elif train_opt["optimizer"] == "Lion":
                self.optimizer_fusion = Lion(
                    optim_params_fusion,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            else:
                print("Not implemented optimizer, default using Adam!")
                self.optimizer_fusion = torch.optim.Adam(
                    optim_params_fusion,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            self.optimizers.append(self.optimizer_fusion)

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
                            eta_min=train_opt["eta_min"],
                        )
                    )
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.log_dict = OrderedDict()

    def feed_data(self, X_fea, Y_fea, Z_fea, t=None, diffusion=None):
        # LQ_fea & GT_fea is a list
        self.latent_X = X_fea.to(self.device)
        self.latent_Y = Y_fea.to(self.device)  # latent
        self.latent_Z = Z_fea.to(self.device)
        self.t = t.to(self.device)
        self.diffusion = diffusion

    def optimize_parameters(self, step):

        if self.opt["dist"]:
            Fuse_fn = self.fusion_model.module
        else:
            Fuse_fn = self.fusion_model

        # X_img = self.decode(self.latent_X)
        # Y_img = self.decode(self.latent_Y)

        ####Optimize loss####
        self.optimizer_fusion.zero_grad()
        loss_dict = self.diffusion.training_losses(
            Fuse_fn,
            self.latent_Z,
            self.t,
            dict(cond_img1=self.latent_X, cond_img2=self.latent_Y),
        )
        total_loss = loss_dict["loss"].mean()
        total_loss.backward()
        self.optimizer_fusion.step()
        self.log_dict["total_loss"] = total_loss.item()

    def test(self, current_step, idx=0):
        self.ae_model.eval()
        self.dino_model.eval()
        self.match_model.eval()
        if self.ema is None:
            raise RuntimeError(
                "EMA model not initialized. Please assign model.ema before calling test()"
            )
        self.ema.eval()

        with torch.no_grad():
            rec_X_img = self.decode(self.latent_X / 0.13605)
            rec_Y_img = self.decode(self.latent_Y / 0.13605)
            rec_Z_img = self.decode(self.latent_Z / 0.13605)

            # if self.opt["dist"]:
            #     fusion_fn = self.ema.module
            # else:
            fusion_fn = self.ema

        n, c, h, w = self.latent_X.shape
        z = torch.randn(n, c, h, w, device=self.device)
        samples = self.sampling_diffusion.p_sample_loop(
            fusion_fn.forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs={
                "cond_img1": self.latent_X,
                "cond_img2": self.latent_Y,
            },
            progress=True,
            device=self.device,
        )
        # print(f"==> Sample latent shape: {samples.shape}, min: {samples.min().item():.4f}, max: {samples.max().item():.4f},mean: {samples.mean().item():.4f} std: {samples.std().item():.4f}")
        with torch.no_grad():
            fuse_recon = self.decode(samples / 0.13605)

        self.output = fuse_recon
        self.Rec_X_img = rec_X_img
        self.Rec_Y_img = rec_Y_img
        self.Rec_Z_img = rec_Z_img

        self.fusion_model.train()

        save_dir = os.path.join("image", f"step_{current_step}")
        os.makedirs(save_dir, exist_ok=True)

        batch_size = self.output.shape[0]
        for i in range(batch_size):
            index = idx + i
            tvutils.save_image(
                (self.output[i] * 0.5 + 0.5).data,
                os.path.join(save_dir, f"{index}_Fused.png"),
                normalize=False,
            )
            tvutils.save_image(
                (self.Rec_X_img[i] * 0.5 + 0.5).data,
                os.path.join(save_dir, f"{index}_Rec_X.png"),
                normalize=False,
            )
            tvutils.save_image(
                (self.Rec_Y_img[i] * 0.5 + 0.5).data,
                os.path.join(save_dir, f"{index}_Rec_Y.png"),
                normalize=False,
            )
            tvutils.save_image(
                (self.Rec_Z_img[i] * 0.5 + 0.5).data,
                os.path.join(save_dir, f"{index}_Rec_Z.png"),
                normalize=False,
            )

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["Fused"] = self.output.detach()[0].float().cpu()
        out_dict["Rec_X_img"] = self.Rec_X_img.detach()[0].float().cpu()
        out_dict["Rec_Y_img"] = self.Rec_Y_img.detach()[0].float().cpu()
        out_dict["Rec_Z_img"] = self.Rec_Z_img.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.fusion_model)
        if isinstance(self.fusion_model, nn.DataParallel) or isinstance(
            self.fusion_model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.fusion_model.__class__.__name__,
                self.fusion_model.module.__class__.__name__,
            )
        else:
            net_struc_str = "{}".format(self.fusion_model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        print("start to load pretrained model...")

        load_path_AE = self.opt["path"]["pretrain_model_AE"]
        if load_path_AE is not None:
            logger.info("Loading model for AutoEncoder [{:s}] ...".format(load_path_AE))
            self.load_network(
                load_path_AE, self.ae_model, self.opt["path"]["strict_load"]
            )

        load_path_Fusion = self.opt["path"]["pretrain_model_Fusion"]
        if load_path_Fusion is not None:
            logger.info(
                "Loading model for FusionModel [{:s}] ...".format(load_path_Fusion)
            )
            self.load_network(
                load_path_Fusion, self.fusion_model, self.opt["path"]["strict_load"]
            )

        load_path_Match = self.opt["path"]["pretrain_model_Matcher"]
        if load_path_Match is not None:
            logger.info("Loading model for Matcher [{:s}] ...".format(load_path_Match))
            self.load_network(
                load_path_Match, self.match_model, self.opt["path"]["strict_load"]
            )

        load_path_Dino = self.opt["path"]["pretrain_model_Dino"]
        if load_path_Dino is not None:
            logger.info("Loading model for Dino [{:s}] ...".format(load_path_Dino))
            self.load_network(
                load_path_Dino, self.dino_model, self.opt["path"]["strict_load"]
            )
        # load_path_Fusion_EMA = self.opt["path"]["pretrain_model_Fusion_EMA"]
        # if load_path_Fusion_EMA is not None:
        #     logger.info("Loading model for FusionModel [{:s}] ...".format(load_path_Fusion_EMA))
        #     ema_state = torch.load(load_path_Fusion_EMA, map_location=self.device)
        #     self.ema.load_state_dict(ema_state)
        # else:
        #     logger.warning("EMA model not found. Using current FusionModel weights as EMA." )

    def save(self, iter_label):
        self.save_network(self.fusion_model, "Fusion", iter_label)
        if self.ema is not None:
            self.save_network(self.ema, "Fusion_EMA", iter_label)

    def warp_image_with_flow(self, img, flow):
        B, C, H, W = img.shape
        flow = flow.permute(0, 2, 3, 1)
        # print("flow min:", flow.min(),"flow max:", flow.max())
        grid = KU.create_meshgrid(H, W).to(img.device)  # [1, H, W, 2]
        grid = grid.expand(B, -1, -1, -1)
        img_warped = F.grid_sample(
            img, grid + flow, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return img_warped

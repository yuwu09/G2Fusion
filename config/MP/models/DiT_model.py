import logging
import os
from collections import OrderedDict
import copy
import kornia.filters as KF
import kornia.utils as KU
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvutils
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.modules.diffusion import create_diffusion
from models.optimizer import Lion
from .base_model import BaseModel

logger = logging.getLogger("base")


class DiT_Model(BaseModel):
    def __init__(self, opt):
        super(DiT_Model, self).__init__(opt)
        os.makedirs("image", exist_ok=True)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1

        train_opt = opt["train"]

        self.ae_model = networks.define_AE(opt).to(self.device)
        self.match_model = networks.define_Match(opt).to(self.device)
        self.dino_model = networks.define_dino(opt).to(self.device)
        self.fusion_model = networks.define_Fusion(opt).to(self.device)
        self.diffusion = create_diffusion(timestep_respacing="")
        # Keep training on the full schedule; evaluate with 20 respaced steps.
        self.sampling_diffusion = create_diffusion(
            timestep_respacing=str(opt.get("sampling_steps", 20))
        )
        self.fusion_model_ema = copy.deepcopy(self.fusion_model).to(self.device)
        for param in self.fusion_model_ema.parameters():
            param.requires_grad = False
        self.fusion_model_ema.eval()
        self.ema_decay = 0.999
        for param in self.ae_model.parameters():
            param.requires_grad = False
        for param in self.dino_model.parameters():
            param.requires_grad = False

        if opt["dist"]:
            self.match_model = DistributedDataParallel(
                self.match_model,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=True,
            )
            self.fusion_model = DistributedDataParallel(
                self.fusion_model,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=True,
            )

        self.load()
        self.fusion_model_ema.load_state_dict(
            self.fusion_net().state_dict(), strict=True
        )
        self.fusion_model_ema.eval()
        self.ae_model.eval()
        self.dino_model.eval()

        self.encode = self.ae_model.encode
        self.decode = self.ae_model.decode

        self.log_dict = OrderedDict()
        self.output = None
        self.Rec_X_img = None

        if self.is_train:
            self._build_optimizers_and_schedulers(train_opt)
            self.set_train_phase("dit")

    def _build_optimizer(self, params, lr, wd, beta1, beta2, optimizer_name):
        if optimizer_name == "Adam":
            return torch.optim.Adam(
                params, lr=lr, weight_decay=wd, betas=(beta1, beta2)
            )
        if optimizer_name == "AdamW":
            return torch.optim.AdamW(
                params, lr=lr, weight_decay=wd, betas=(beta1, beta2)
            )
        if optimizer_name == "Lion":
            return Lion(params, lr=lr, weight_decay=wd, betas=(beta1, beta2))
        logger.warning("Unknown optimizer %s, fallback to Adam.", optimizer_name)
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=(beta1, beta2))

    def _append_scheduler(self, optimizer, train_opt):
        if train_opt["lr_scheme"] == "MultiStepLR":
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
            self.schedulers.append(
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=train_opt["niter"],
                    eta_min=train_opt["eta_min"],
                )
            )
        else:
            raise NotImplementedError(
                "Unsupported learning rate scheme: {}".format(train_opt["lr_scheme"])
            )

    def _build_optimizers_and_schedulers(self, train_opt):
        wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
        optimizer_name = train_opt.get("optimizer", "Adam")
        beta1 = train_opt["beta1"]
        beta2 = train_opt["beta2"]

        lr_match = train_opt["lr_match"]
        lr_dit = train_opt["lr_dit"]

        optim_params_match = [
            v for _, v in self.match_model.named_parameters() if v.requires_grad
        ]
        optim_params_fusion = [
            v for _, v in self.fusion_model.named_parameters() if v.requires_grad
        ]

        self.optimizer_match = self._build_optimizer(
            optim_params_match, lr_match, wd_G, beta1, beta2, optimizer_name
        )
        self.optimizer_fusion = self._build_optimizer(
            optim_params_fusion, lr_dit, wd_G, beta1, beta2, optimizer_name
        )

        self.optimizers.append(self.optimizer_match)
        self.optimizers.append(self.optimizer_fusion)

        self._append_scheduler(self.optimizer_match, train_opt)
        self._append_scheduler(self.optimizer_fusion, train_opt)

    def match_net(self):
        return self.match_model.module if self.opt["dist"] else self.match_model

    def fusion_net(self):
        return self.fusion_model.module if self.opt["dist"] else self.fusion_model

    def fusion_net_ema(self):
        return (
            self.fusion_model_ema.module
            if self.opt["dist"]
            and isinstance(self.fusion_model_ema, DistributedDataParallel)
            else self.fusion_model_ema
        )

    def set_train_phase(self, phase):
        if phase == "match":
            for param in self.match_model.parameters():
                param.requires_grad = True
            for param in self.fusion_model.parameters():
                param.requires_grad = False
            self.match_model.train()
            self.fusion_model.eval()
        elif phase == "dit":
            for param in self.match_model.parameters():
                param.requires_grad = False
            for param in self.fusion_model.parameters():
                param.requires_grad = True
            self.match_model.eval()
            self.fusion_model.train()
        else:
            raise ValueError("Unknown phase: {}".format(phase))
        self.current_phase = phase

    def feed_data(self, X, Y, Z, flow_gt=None):
        self.X = X.to(self.device)
        self.Y = Y.to(self.device)
        self.Z = Z.to(self.device)
        if self.is_train and flow_gt is None:
            raise ValueError("Training requires a ground-truth deformation field.")
        self.flow_gt = flow_gt.to(self.device) if flow_gt is not None else None

    def fused_proxy_from_pred_xstart(self, latent_X, latent_Y, latent_Z):
        t = torch.randint(
            low=0,
            high=25,
            size=(latent_X.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        loss_dict = self.diffusion.training_losses(
            self.fusion_net(),
            latent_Z,
            t,
            model_kwargs={"cond_img1": latent_X, "cond_img2": latent_Y},
        )
        pred_xstart = loss_dict["pred_xstart"]
        fuse_img = self.decode(pred_xstart / 0.18215)
        return fuse_img

    def train_match_step(self):
        self.optimizer_match.zero_grad()
        with torch.no_grad():
            fea_X = self.dino_model(self.X)
            fea_Y = self.dino_model(self.Y)

        flow_pred_xy = self.match_net()(
            src=self.X,
            tgt=self.Y,
            dino_src=fea_X,
            dino_tgt=fea_Y,
        ).permute(0, 2, 3, 1)

        X_wrap = self.warp_image_with_flow(self.X, flow_pred_xy)
        latent_X = self.encode(X_wrap) * 0.18215
        latent_Y = self.encode(self.Y) * 0.18215
        latent_Z = self.encode(self.Z) * 0.18215

        fuse_img = self.fused_proxy_from_pred_xstart(latent_X, latent_Y, latent_Z)

        with torch.no_grad():
            fea_fuse = self.dino_model(fuse_img)
        flow_pred_fuse = self.match_net()(
            src=fuse_img,
            tgt=self.Y,
            dino_src=fea_fuse,
            dino_tgt=fea_Y,
        ).permute(0, 2, 3, 1)

        flow_zero = torch.zeros_like(flow_pred_fuse)
        loss_fuse = self.weightfiledloss(fuse_img, self.Y, flow_pred_fuse, flow_zero)
        loss_real = self.weightfiledloss(self.X, self.Y, flow_pred_xy, self.flow_gt)

        total_loss = loss_real + 0.1 * loss_fuse

        total_loss.backward()
        self.optimizer_match.step()

        self.output = fuse_img.detach()
        self.Rec_X_img = X_wrap.detach()

        self.log_dict["phase"] = "match"
        self.log_dict["total_loss"] = total_loss.item()
        self.log_dict["loss_match_real"] = loss_real.item()
        self.log_dict["loss_match_fuse"] = loss_fuse.item()

    def train_dit_step(self):
        self.optimizer_fusion.zero_grad()

        with torch.no_grad():
            fea_X = self.dino_model(self.X)
            fea_Y = self.dino_model(self.Y)
            flow_pred_xy = self.match_net()(
                src=self.X,
                tgt=self.Y,
                dino_src=fea_X,
                dino_tgt=fea_Y,
            ).permute(0, 2, 3, 1)
            X_wrap = self.warp_image_with_flow(self.X, flow_pred_xy)
            latent_X = self.encode(X_wrap) * 0.18215
            latent_Y = self.encode(self.Y) * 0.18215
            latent_Z = self.encode(self.Z) * 0.18215

        t_main = torch.randint(
            0,
            self.diffusion.num_timesteps,
            (latent_X.shape[0],),
            device=self.device,
            dtype=torch.long,
        )

        loss_dict_main = self.diffusion.training_losses(
            self.fusion_net(),
            latent_Z,
            t_main,
            model_kwargs={"cond_img1": latent_X, "cond_img2": latent_Y},
        )
        diff_loss = loss_dict_main["loss"].mean()
        fuse_img = self.fused_proxy_from_pred_xstart(latent_X, latent_Y, latent_Z)
        fea_fuse = self.dino_model(fuse_img)
        flow_pred_fuse = self.match_net()(
            src=fuse_img,
            tgt=self.Y,
            dino_src=fea_fuse,
            dino_tgt=fea_Y,
        ).permute(0, 2, 3, 1)
        flow_zero = torch.zeros_like(flow_pred_fuse)
        loss_flow = self.weightfiledloss(fuse_img, self.Y, flow_pred_fuse, flow_zero)

        total_loss = diff_loss + 0.01 * loss_flow
        total_loss.backward()
        self.optimizer_fusion.step()
        self.update_fusion_ema()

        self.output = fuse_img.detach()
        self.Rec_X_img = X_wrap.detach()

        self.log_dict["phase"] = "dit"
        self.log_dict["total_loss"] = total_loss.item()
        self.log_dict["loss_dit_diff"] = diff_loss.item()
        self.log_dict["loss_dit_flow"] = loss_flow.item()

    def optimize_parameters(self, step):
        cycle = 600
        step_in_cycle = (step - 1) % cycle + 1
        if step_in_cycle <= 500:
            phase = "dit"
        else:
            phase = "match"
        self.set_train_phase(phase)
        self.ae_model.eval()
        self.dino_model.eval()

        if phase == "match":
            self.train_match_step()
        else:
            self.train_dit_step()

    @torch.no_grad()
    def test(self, current_step, idx=0):
        self.ae_model.eval()
        self.dino_model.eval()
        self.match_model.eval()
        self.fusion_model.eval()
        self.fusion_model_ema.eval()

        fea_X = self.dino_model(self.X)
        fea_Y = self.dino_model(self.Y)
        flow_pred_xy = self.match_net()(
            src=self.X,
            tgt=self.Y,
            dino_src=fea_X,
            dino_tgt=fea_Y,
        ).permute(0, 2, 3, 1)

        X_wrap = self.warp_image_with_flow(self.X, flow_pred_xy)
        latent_X = self.encode(X_wrap) * 0.18215
        latent_Y = self.encode(self.Y) * 0.18215

        n, c, h, w = latent_X.shape
        z = torch.randn(n, c, h, w, device=self.device)
        samples = self.sampling_diffusion.p_sample_loop(
            self.fusion_net_ema().forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs={"cond_img1": latent_X, "cond_img2": latent_Y},
            progress=True,
            device=self.device,
        )
        fuse_img = self.decode(samples / 0.18215)

        self.output = fuse_img
        self.Rec_X_img = X_wrap

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
                (self.X[i] * 0.5 + 0.5).data,
                os.path.join(save_dir, f"{index}_X.png"),
                normalize=False,
            )
            tvutils.save_image(
                (self.Y[i] * 0.5 + 0.5).data,
                os.path.join(save_dir, f"{index}_Y.png"),
                normalize=False,
            )
            tvutils.save_image(
                (self.Z[i] * 0.5 + 0.5).data,
                os.path.join(save_dir, f"{index}_Z.png"),
                normalize=False,
            )

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["Fused"] = self.output.detach()[0].float().cpu()
        out_dict["Rec_X_img"] = self.Rec_X_img.detach()[0].float().cpu()
        out_dict["X_img"] = self.X.detach()[0].float().cpu()
        out_dict["Y_img"] = self.Y.detach()[0].float().cpu()
        out_dict["Z_img"] = self.Z.detach()[0].float().cpu()
        return out_dict

    @torch.no_grad()
    def get_current_train_visuals(self):
        """
        Return train-stage visuals for debugging/visualization.
        Uses fast proxy fuse (pred_xstart -> decode), not full sampling.
        """
        out_dict = OrderedDict()

        # Original images
        out_dict["X"] = self.X.detach()[0].float().cpu()
        out_dict["Y"] = self.Y.detach()[0].float().cpu()
        out_dict["Z"] = self.Z.detach()[0].float().cpu()

        # Currently registered X
        fea_X = self.dino_model(self.X)
        fea_Y = self.dino_model(self.Y)

        flow_pred_xy = self.match_net()(
            src=self.X,
            tgt=self.Y,
            dino_src=fea_X,
            dino_tgt=fea_Y,
        ).permute(0, 2, 3, 1)

        X_wrap = self.warp_image_with_flow(self.X, flow_pred_xy)
        out_dict["X_wrap"] = X_wrap.detach()[0].float().cpu()

        # Encode the latent representations.
        latent_X = self.encode(X_wrap) * 0.18215
        latent_Y = self.encode(self.Y) * 0.18215
        latent_Z = self.encode(self.Z) * 0.18215

        # Use the same proxy fusion as during training.
        fuse_img = self.fused_proxy_from_pred_xstart(latent_X, latent_Y, latent_Z)

        out_dict["Fuse_train"] = fuse_img.detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        for net_name, net in [
            ("Match", self.match_model),
            ("Fusion", self.fusion_model),
        ]:
            s, n = self.get_network_description(net)
            if isinstance(net, (DataParallel, DistributedDataParallel)):
                net_struc_str = "{} - {}".format(
                    net.__class__.__name__, net.module.__class__.__name__
                )
            else:
                net_struc_str = "{}".format(net.__class__.__name__)
            if self.rank <= 0:
                logger.info(
                    "Network {} structure: {}, with parameters: {:,d}".format(
                        net_name, net_struc_str, n
                    )
                )
                logger.info(s)

    def load(self):
        print("start to load pretrained model...")

        load_path_AE = self.opt["path"].get("pretrain_model_AE", None)
        if load_path_AE is not None:
            logger.info("Loading model for AutoEncoder [{:s}] ...".format(load_path_AE))
            self.load_network(
                load_path_AE, self.ae_model, self.opt["path"]["strict_load"]
            )

        load_path_Fusion = self.opt["path"].get("pretrain_model_Fusion", None)
        if load_path_Fusion is not None:
            logger.info(
                "Loading model for FusionModel [{:s}] ...".format(load_path_Fusion)
            )
            self.load_network(
                load_path_Fusion, self.fusion_model, self.opt["path"]["strict_load"]
            )

        load_path_Match = self.opt["path"].get("pretrain_model_Matcher", None)
        if load_path_Match is not None:
            logger.info("Loading model for Matcher [{:s}] ...".format(load_path_Match))
            self.load_network(
                load_path_Match, self.match_model, self.opt["path"]["strict_load"]
            )

        load_path_Dino = self.opt["path"].get("pretrain_model_Dino", None)
        if load_path_Dino is not None:
            logger.info("Loading model for Dino [{:s}] ...".format(load_path_Dino))
            self.load_network(
                load_path_Dino, self.dino_model, self.opt["path"]["strict_load"]
            )

    def save(self, iter_label):
        self.save_network(self.fusion_model, "Fusion", iter_label)
        self.save_network(self.fusion_model_ema, "Fusion_EMA", iter_label)
        self.save_network(self.match_model, "Matcher", iter_label)

    @torch.no_grad()
    def update_fusion_ema(self, decay=None):
        if decay is None:
            decay = self.ema_decay

        net = self.fusion_net()
        net_ema = self.fusion_net_ema()

        param_dict_src = dict(net.named_parameters())
        param_dict_ema = dict(net_ema.named_parameters())

        for k in param_dict_src.keys():
            param_dict_ema[k].data.mul_(decay).add_(
                param_dict_src[k].data, alpha=1 - decay
            )

        buffer_dict_src = dict(net.named_buffers())
        buffer_dict_ema = dict(net_ema.named_buffers())

        for k in buffer_dict_src.keys():
            buffer_dict_ema[k].data.copy_(buffer_dict_src[k].data)

    def weightfiledloss(self, ref, tgt, disp, disp_gt):
        _, _, H, W = ref.shape
        border_mask = torch.zeros([1, 1, H, W]).to(self.device)
        border_mask[:, :, 10:-10, 10:-10] = 1

        disp = disp.permute(0, 3, 1, 2).contiguous()
        disp_gt = disp_gt.permute(0, 3, 1, 2).contiguous()

        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (
            ref.std(dim=[-1, -2], keepdim=True) + 1e-5
        )
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (
            tgt.std(dim=[-1, -2], keepdim=True) + 1e-5
        )

        g_ref = (
            KF.spatial_gradient(ref, order=2)
            .mean(dim=1)
            .abs()
            .sum(dim=1)
            .detach()
            .unsqueeze(1)
        )
        g_tgt = (
            KF.spatial_gradient(tgt, order=2)
            .mean(dim=1)
            .abs()
            .sum(dim=1)
            .detach()
            .unsqueeze(1)
        )
        w = ((g_ref + g_tgt) * 2 + 1) * border_mask
        return (w * (1000 * (disp - disp_gt).abs().clamp(min=1e-2).pow(2))).mean()

    def warp_image_with_flow(self, img, flow):
        B, C, H, W = img.shape
        grid = KU.create_meshgrid(H, W).to(img.device)
        grid = grid.expand(B, -1, -1, -1)
        img_warped = F.grid_sample(
            img, grid + flow, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        mask = torch.ones_like(img[:, :1, :, :])
        mask_warp = F.grid_sample(
            mask,
            grid + flow,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        valid_mask = (mask_warp > 0.999).float()
        warped = img_warped * valid_mask + (1 - valid_mask) * (-1)
        return warped

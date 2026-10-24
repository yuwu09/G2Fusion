##TODO:add ema

import argparse
import logging
import math
import os
import random
import sys
import copy
import gc
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# from IPython import embed
import kornia.utils as KU
import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr
import torch.nn.functional as F
from copy import deepcopy
from models.modules.diffusion import create_diffusion
from utils.ema_utils import update_ema, requires_grad

# torch.autograd.set_detect_anomaly(True)


def init_dist(backend="nccl", **kwargs):
    """initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def warp_image_with_flow(img, flow):
    B, C, H, W = img.shape
    grid = KU.create_meshgrid(H, W).to(img.device)  # [1, H, W, 2]
    grid = grid.expand(B, -1, -1, -1)

    img_warped = F.grid_sample(
        img, grid + flow, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    mask = torch.ones_like(img[:, :1, :, :])
    mask_warp = F.grid_sample(
        mask, grid + flow, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    warped = img_warped * mask_warp + (1 - mask_warp) * (-1)
    return warped


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                    and isinstance(path, (str, os.PathLike))
                )
            )

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir=os.path.join(opt["path"]["experiments_root"], "tb_logger"))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    #### create train and val dataloader
    dataset_ratio = 1000  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None
    print("create dataset done!!")

    #### create model
    model = create_model(opt)
    device = model.device
    if opt["dist"]:
        ema = deepcopy(model.fusion_model.module).to(device)
    else:
        ema = deepcopy(model.fusion_model).to(device)
    requires_grad(ema, False)
    model.ema = ema
    update_ema(
        model.ema,
        model.fusion_model.module if opt["dist"] else model.fusion_model,
        decay=0,
    )
    model.ema.eval()
    diffusion = create_diffusion(timestep_respacing="")
    print("create model done!!")

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value("b", False)

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            # print(current_step)
            X, Y, Z = train_data["X"], train_data["Y"], train_data["Z"]
            with torch.no_grad():
                fea_X = model.dino_model(X.to(device))
                fea_Y = model.dino_model(Y.to(device))
                flow_pred = model.match_model(
                    src=X.to(device),  # vi_wrap
                    tgt=Y.to(device),  # ir_clean
                    dino_src=fea_X,
                    dino_tgt=fea_Y,
                ).permute(0, 2, 3, 1)

                X_wrap = warp_image_with_flow(X.to(device), flow_pred)
                latent_X = model.encode(X_wrap).mul_(0.13605)
                latent_Y = model.encode(Y.to(device)).mul_(0.13605)
                latent_Z = model.encode(Z.to(device)).mul_(0.13605)

            t = torch.randint(
                0, diffusion.num_timesteps, (latent_Z.shape[0],), device=device
            )
            model.feed_data(latent_X, latent_Y, latent_Z, t, diffusion)  # xt, mu, x0
            model.optimize_parameters(current_step)
            update_ema(
                model.ema,
                model.fusion_model.module if opt["dist"] else model.fusion_model,
                decay=0.999,
            )
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr = 0.0
                avg_lncc = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):
                    X, Y, Z = val_data["X"], val_data["Y"], val_data["Z"]
                    X_GT = val_data["X_GT"]
                    with torch.no_grad():
                        fea_X = model.dino_model(X.to(device))
                        fea_Y = model.dino_model(Y.to(device))
                        flow_pred = model.match_model(
                            src=X.to(device),  # vi_wrap
                            tgt=Y.to(device),  # ir_clean
                            dino_src=fea_X,
                            dino_tgt=fea_Y,
                        ).permute(0, 2, 3, 1)

                        X_wrap = warp_image_with_flow(X.to(device), flow_pred)
                        t = torch.randint(
                            0,
                            diffusion.num_timesteps,
                            (latent_Z.shape[0],),
                            device=device,
                        )
                        latent_X = model.encode(X_wrap).mul_(0.13605)
                        latent_Y = model.encode(Y.to(device)).mul_(0.13605)
                        latent_Z = model.encode(Z.to(device)).mul_(0.13605)

                    model.feed_data(
                        latent_X, latent_Y, latent_Z, t, diffusion
                    )  # xt, mu, x0
                    # valid Predictor
                    model.test(current_step, idx)
                    visuals = model.get_current_visuals()

                    fuse = util.tensor2img(
                        (visuals["Fused"] * 0.5 + 0.5).squeeze()
                    )  # uint8
                    rec_z = util.tensor2img((Z * 0.5 + 0.5).squeeze())
                    x_gt_img = util.tensor2img((X_GT * 0.5 + 0.5).squeeze())
                    y_img = util.tensor2img((Y * 0.5 + 0.5).squeeze())
                    # calculate PSNR
                    avg_psnr += util.calculate_psnr(fuse, rec_z)
                    avg_lncc += util.calculate_lncc(fuse, x_gt_img, y_img)
                    # avg_en += util.calculate_en(fuse)
                    idx += 1

                    # break
                    gc.collect()
                    torch.cuda.empty_cache()

                avg_psnr = avg_psnr / idx
                avg_lncc = avg_lncc / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                # log
                logger.info(
                    "# Validation # PSNR: {:.6f}, #LNCC: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(
                        avg_psnr, avg_lncc, best_psnr, best_iter
                    )
                )
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, PSNR: {:.6f}, LNCC: {:.6f}".format(
                        epoch, current_step, avg_psnr, avg_lncc
                    )
                )
                print(
                    "<epoch:{:3d}, iter:{:8,d}, PSNR: {:.6f}, LNCC: {:.6f}".format(
                        epoch, current_step, avg_psnr, avg_lncc
                    )
                )
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)
                    tb_logger.add_scalar("lncc", avg_lncc, current_step)

            if error.value:
                sys.exit(0)
            #### save models and training states
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()

# --master-port=29502
# torchrun --nnodes=1 --nproc_per_node=2  train_DiT.py -opt ./options/train/train_DiT.yml

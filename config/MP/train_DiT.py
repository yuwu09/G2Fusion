import argparse
import gc
import logging
import math
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler


def init_dist(backend="nccl", **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)

    tb_logger = None

    if args.launcher == "none":
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    torch.backends.cudnn.benchmark = True

    if opt["path"].get("resume_state", None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])
    else:
        resume_state = None

    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(opt["path"]["experiments_root"])
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if key != "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                    and isinstance(path, (str, os.PathLike))
                )
            )

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

        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:
                from torch.utils.tensorboard import SummaryWriter
            else:
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir=os.path.join(opt["path"]["experiments_root"], "tb_logger"))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    dataset_ratio = 1000
    train_loader = None
    val_loader = None
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

    model = create_model(opt)
    print("create model done!!")

    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)
    else:
        current_step = 0
        start_epoch = 0

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

            X = train_data["X"]
            Y = train_data["Y"]
            Z = train_data["Z"]
            flow_gt = train_data["flow_gt"]

            model.feed_data(X, Y, Z, flow_gt)
            model.optimize_parameters(current_step)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            if current_step % 500 == 0:
                train_visuals = model.get_current_train_visuals()

                save_dir = os.path.join(opt["path"]["experiments_root"], "train_vis")
                os.makedirs(save_dir, exist_ok=True)

                x_img = util.tensor2img(train_visuals["X"], min_max=(-1, 1))
                y_img = util.tensor2img(train_visuals["Y"], min_max=(-1, 1))
                z_img = util.tensor2img(train_visuals["Z"], min_max=(-1, 1))
                xw_img = util.tensor2img(train_visuals["X_wrap"], min_max=(-1, 1))
                fuse_img = util.tensor2img(train_visuals["Fuse_train"], min_max=(-1, 1))
                panel = np.concatenate([x_img, xw_img, y_img, z_img, fuse_img], axis=1)
                util.save_img(panel, os.path.join(save_dir, f"{current_step:06d}_panel.png"))
                
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    if isinstance(v, str):
                        message += "{:s}: {:s} ".format(k, v)
                    else:
                        message += "{:s}: {:.4e} ".format(k, v)
                        if (
                            opt["use_tb_logger"]
                            and "debug" not in opt["name"]
                            and rank <= 0
                        ):
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr = 0.0
                avg_lncc = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):
                    X = val_data["X"]
                    Y = val_data["Y"]
                    Z = val_data["Z"]
                    X_GT = val_data["X_GT"]
                    flow_gt = val_data["flow_gt"]

                    model.feed_data(X, Y, Z, flow_gt)
                    model.test(current_step, idx)
                    visuals = model.get_current_visuals()

                    fuse = util.tensor2img((visuals["Fused"] * 0.5 + 0.5).squeeze())
                    y_img = util.tensor2img((visuals["Y_img"] * 0.5 + 0.5).squeeze())
                    z_img = util.tensor2img((visuals["Z_img"] * 0.5 + 0.5).squeeze())
                    x_gt_img = util.tensor2img((X_GT * 0.5 + 0.5).squeeze())

                    avg_psnr += util.calculate_psnr(fuse, z_img)
                    avg_lncc += util.calculate_lncc(fuse, x_gt_img, y_img)
                    idx += 1

                    gc.collect()
                    torch.cuda.empty_cache()

                avg_psnr = avg_psnr / idx
                avg_lncc = avg_lncc / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step

                logger.info(
                    "# Validation # PSNR: {:.6f}, #LNCC: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(
                        avg_psnr, avg_lncc, best_psnr, best_iter
                    )
                )
                logger_val = logging.getLogger("val")
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, PSNR: {:.6f}, LNCC: {:.6f} ".format(
                        epoch, current_step, avg_psnr, avg_lncc
                    )
                )
                print(
                    "<epoch:{:3d}, iter:{:8,d}, PSNR: {:.6f}, LNCC: {:.6f}".format(
                        epoch, current_step, avg_psnr, avg_lncc
                    )
                )
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)
                    tb_logger.add_scalar("lncc", avg_lncc, current_step)

            if error.value:
                logger.error("Training aborted due to worker/data error flag.")
                break

            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                if rank <= 0:
                    logger.info("Saving models and training states.")
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

        if error.value:
            break

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        model.save_training_state(epoch, current_step)
        logger.info("End of training.")

    if tb_logger is not None:
        tb_logger.close()


if __name__ == "__main__":
    main()

"""Evaluate the registration and fusion pipeline after mutual promotion."""

import argparse
import logging
import os
import time

import numpy as np
import torch

import options as option
import utils as util
from data import create_dataloader, create_dataset
from models import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-opt", required=True, help="Path to the test YAML file.")
    args = parser.parse_args()
    opt = option.dict_to_nonedict(option.parse(args.opt, is_train=False))
    util.mkdirs([opt["path"]["results_root"], opt["path"]["log"]])
    util.setup_logger(
        "base", opt["path"]["log"], "test_" + opt["name"],
        level=logging.INFO, screen=True, tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))
    model = create_model(opt)
    device = model.device

    for dataset_opt in opt["datasets"].values():
        dataset = create_dataset(dataset_opt)
        loader = create_dataloader(dataset, dataset_opt, opt)
        dataset_dir = os.path.join(opt["path"]["results_root"], dataset_opt["name"])
        fused_dir = os.path.join(dataset_dir, "fused")
        registered_dir = os.path.join(dataset_dir, "registered")
        util.mkdirs([fused_dir, registered_dir])
        durations = []

        for index, batch in enumerate(loader):
            name = os.path.splitext(os.path.basename(batch["X_path"][0]))[0]
            with torch.no_grad():
                # MP takes images and performs registration and encoding internally.
                model.feed_data(batch["X"], batch["Y"], batch["Z"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                model.test(current_step=0, idx=index)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                durations.append(time.perf_counter() - start)

            visuals = model.get_current_visuals()
            filename = name + (opt["suffix"] or "") + ".png"
            for key, folder in [("Fused", fused_dir), ("Rec_X_img", registered_dir)]:
                img = util.tensor2img(visuals[key], min_max=(-1, 1))
                util.save_img(img, os.path.join(folder, filename))
            logger.info("[%d/%d] %s", index + 1, len(loader), name)

        if durations:
            logger.info("Average test time: %.4fs per image", np.mean(durations))


if __name__ == "__main__":
    main()

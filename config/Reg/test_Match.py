import argparse
import logging
import os
import sys
import time
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import torch

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to options YMAL file."
    )
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    #### mkdir and logger
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


    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"],
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    #### create test dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt, opt, None)
        logger.info(
            "Number of test images in [{:s}]: {:d}".format(
                dataset_opt["name"], len(test_set)
            )
        )
        test_loaders.append(test_loader)

    #### load model
    model = create_model(opt)
    device = model.device

    model.dino_model.eval()
    # model.eval()

    with torch.no_grad():
        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt["name"]
            logger.info("\nTesting [{:s}]...".format(test_set_name))

            dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
            util.mkdir(dataset_dir)

            test_results = OrderedDict()
            test_results["psnr"] = []
            test_results["ssim"] = []
            test_times = []

            for i, test_data in enumerate(test_loader):
                X = test_data["X"].to(device)
                Y = test_data["Y"].to(device)
                flow_gt = test_data["flow_gt"].to(device)
                X_GT = test_data["X_GT"].to(device)
                X_path = test_data["X_path"][0]
                img_name = os.path.splitext(os.path.basename(X_path))[0]
                fea_X = model.dino_model(X.to(device))
                fea_Y = model.dino_model(Y.to(device))
                #### inference
                model.feed_data(X, Y, fea_X, fea_Y, flow_gt, X_GT)
                tic = time.time()
                model.test(current_step=0, idx=i)
                toc = time.time()
                test_times.append(toc - tic)

                visuals = model.get_current_visuals()

                # [-1,1] -> [0,255]
                X_gt = util.tensor2img((visuals["X_GT"] * 0.5 + 0.5).squeeze())
                X_rec = util.tensor2img((visuals["X_rec"] * 0.5 + 0.5).squeeze())

                #### save images
                util.save_img(X_rec, os.path.join(dataset_dir, img_name + "_X_rec.png"))
                util.save_img(X_gt, os.path.join(dataset_dir, img_name + "_X_GT.png"))

                #### metrics
                psnr = util.calculate_psnr(X_rec, X_gt)
                ssim = util.calculate_ssim(X_rec, X_gt)

                test_results["psnr"].append(psnr)
                test_results["ssim"].append(ssim)

                logger.info(
                    "img{:3d}:{:15s} | PSNR: {:.4f} | "
                    "SSIM{:.4f}".format(i, img_name, psnr, ssim)
                )

            #### average results
            ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
            ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])

            logger.info(
                "----Average results for {}----\n"
                "\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
                    test_set_name, ave_psnr, ave_ssim
                )
            )

            print(f"Average test time: {np.mean(test_times):.4f}s")


if __name__ == "__main__":
    main()

import argparse
import logging
import os.path
import sys
import time
import numpy as np
import torch

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from copy import deepcopy

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)
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

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model
model = create_model(opt)
device = model.device
if opt["dist"]:
    ema = deepcopy(model.fusion_model.module).to(device)
else:
    ema = deepcopy(model.fusion_model).to(device)
model.ema = ema
diffusion = model.sampling_diffusion

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_times = []
    # Create three output subdirectories under dataset_dir.
    fused_dir = os.path.join(dataset_dir, "fused")
    recz_dir = os.path.join(dataset_dir, "recZ")
    util.mkdir(fused_dir)
    util.mkdir(recz_dir)

    for i, test_data in enumerate(test_loader):
        img_path = test_data["X_path"][0]  # Use the X image filename for the saved result.
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{i+1}/{len(test_loader)}] {img_name}")

        X, Y, Z = test_data["X"], test_data["Y"], test_data["Z"]
        with torch.no_grad():
            latent_X = model.encode(X.to(device)).mul_(0.13605)
            latent_Y = model.encode(Y.to(device)).mul_(0.13605)
            latent_Z = model.encode(Z.to(device)).mul_(0.13605)
            t = torch.randint(
                0, diffusion.num_timesteps, (latent_Z.shape[0],), device=device
            )
            tic = time.time()
            model.feed_data(latent_X, latent_Y, latent_Z, t, diffusion)
            model.test(current_step=0)
            toc = time.time()
            test_times.append(toc - tic)

        visuals = model.get_current_visuals()
        fused = util.tensor2img((visuals["Fused"] * 0.5 + 0.5).squeeze())
        rec_z = util.tensor2img((Z * 0.5 + 0.5).squeeze())
        # fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        # rec_x = (rec_x - rec_x.min()) / (rec_x.max() - rec_x.min() + 1e-8)
        suffix = opt["suffix"] if opt["suffix"] else ""
        util.save_img(fused, os.path.join(fused_dir, img_name + suffix + ".png"))
        util.save_img(rec_z, os.path.join(recz_dir, img_name + suffix + ".png"))

    print(f"Average test time: {np.mean(test_times):.4f}s per image")

# python test_DiT.py -opt ./options/test/test_DiT.yml

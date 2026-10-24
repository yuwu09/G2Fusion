import os
import random
import sys
from data.util import randflow, randrot, randfilp
import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data
import math
import torch.nn.functional as F

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


class XYZflowplusDataset(data.Dataset):
    """
    Read X / Y / Z image pairs.

    Train:
        - fixed-size crop using X_size / Y_size / Z_size
        - then generate warped X online by randflow

    Test/Val:
        - if opt['dataroot_X_warp'] is provided, directly read warped X from that path
        - otherwise keep old behavior (generate warped X by randflow)
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.X_paths, self.Y_paths, self.Z_paths = None, None, None
        self.X_warp_paths = None

        self.X_env, self.Y_env, self.Z_env = None, None, None
        self.X_warp_env = None

        self.X_size, self.Y_size, self.Z_size = (
            opt["X_size"],
            opt["Y_size"],
            opt["Z_size"],
        )

        self.use_external_warp = (
            opt.get("dataroot_X_warp", None) is not None
            and opt.get("dataroot_X_warp", "") != ""
        )

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.X_paths, self.X_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_X"]
            )
            self.Y_paths, self.Y_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_Y"]
            )
            self.Z_paths, self.Z_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_Z"]
            )

            if self.use_external_warp:
                self.X_warp_paths, self.X_warp_sizes = util.get_image_paths(
                    opt["data_type"], opt["dataroot_X_warp"]
                )

        elif opt["data_type"] == "img":
            self.X_paths = util.get_image_paths(opt["data_type"], opt["dataroot_X"])
            self.Y_paths = util.get_image_paths(opt["data_type"], opt["dataroot_Y"])
            self.Z_paths = util.get_image_paths(opt["data_type"], opt["dataroot_Z"])

            if self.use_external_warp:
                self.X_warp_paths = util.get_image_paths(
                    opt["data_type"], opt["dataroot_X_warp"]
                )
        else:
            raise ValueError("Error: data_type is not matched in Dataset")

        assert self.Y_paths, "Error: Y paths are empty."

        if self.X_paths and self.Y_paths and self.Z_paths:
            assert len(self.X_paths) == len(
                self.Y_paths
            ), "Y and X datasets have different number of images - {}, {}.".format(
                len(self.X_paths), len(self.Y_paths)
            )
            assert len(self.X_paths) == len(
                self.Z_paths
            ), "Z and X datasets have different number of images - {}, {}.".format(
                len(self.X_paths), len(self.Z_paths)
            )

        if self.use_external_warp:
            assert (
                self.X_warp_paths is not None and len(self.X_warp_paths) > 0
            ), "dataroot_X_warp is provided, but no images were found."
            assert len(self.X_warp_paths) == len(
                self.X_paths
            ), "X_warp and X datasets have different number of images - {}, {}.".format(
                len(self.X_warp_paths), len(self.X_paths)
            )
            assert len(self.X_warp_paths) == len(
                self.Y_paths
            ), "X_warp and Y datasets have different number of images - {}, {}.".format(
                len(self.X_warp_paths), len(self.Y_paths)
            )

        self.random_scale_list = [1]

    def _init_lmdb(self):
        self.Y_env = lmdb.open(
            self.opt["dataroot_Y"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.X_env = lmdb.open(
            self.opt["dataroot_X"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.Z_env = lmdb.open(
            self.opt["dataroot_Z"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if self.use_external_warp:
            self.X_warp_env = lmdb.open(
                self.opt["dataroot_X_warp"],
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def _ensure_three_channels(self, img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2] == 1:
            img = np.dstack((img, img, img))
        return img

    def _read_one_img(self, env, path, size_info=None):
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in size_info.split("_")]
        else:
            resolution = None
        img = util.read_img(env, path, resolution)  # HWC, BGR, [0,1]
        return img

    def _to_tensor(self, img):
        return torch.from_numpy(
            np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
        ).float()

    def _bgr_to_rgb_if_needed(self, img):
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        return img

    def _resize_min_size(self, img, min_h, min_w):
        h, w = img.shape[:2]
        if h >= min_h and w >= min_w:
            return img

        scale = max(min_h / h, min_w / w)
        new_h = math.ceil(h * scale)
        new_w = math.ceil(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if (self.Y_env is None) or (self.X_env is None) or (self.Z_env is None):
                self._init_lmdb()

        Y_path, X_path, Z_path, X_warp_path = None, None, None, None
        scale = self.opt["scale"] if self.opt["scale"] else 1
        Y_size = self.opt["Y_size"]
        X_size = self.opt["X_size"]
        Z_size = self.opt["Z_size"]

        # ======================
        # read Y
        # ======================
        Y_path = self.Y_paths[index]
        if self.opt["data_type"] == "lmdb":
            y_resolution = self.Y_sizes[index]
        else:
            y_resolution = None
        img_Y = self._read_one_img(self.Y_env, Y_path, y_resolution)

        if self.opt["phase"] != "train":
            img_Y = util.modcrop(img_Y, scale)

        # ======================
        # read Z
        # ======================
        Z_path = self.Z_paths[index]
        if self.opt["data_type"] == "lmdb":
            z_resolution = self.Z_sizes[index]
        else:
            z_resolution = None
        img_Z = self._read_one_img(self.Z_env, Z_path, z_resolution)

        if self.opt["phase"] != "train":
            img_Z = util.modcrop(img_Z, scale)

        # ======================
        # read X
        # ======================
        if self.X_paths:
            X_path = self.X_paths[index]
            if self.opt["data_type"] == "lmdb":
                x_resolution = self.X_sizes[index]
            else:
                x_resolution = None
            img_X = self._read_one_img(self.X_env, X_path, x_resolution)
        else:
            if self.opt["phase"] == "train":
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_Y.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, Y_size)
                W_s = _mod(W_s, random_scale, scale, Y_size)
                img_Y = cv2.resize(
                    np.copy(img_Y), (W_s, H_s), interpolation=cv2.INTER_LINEAR
                )
                if img_Y.ndim == 2:
                    img_Y = cv2.cvtColor(img_Y, cv2.COLOR_GRAY2BGR)

            img_X = util.imresize(img_Y, 1 / scale, True)
            if img_X.ndim == 2:
                img_X = np.expand_dims(img_X, axis=2)

        # ======================
        # read external warped X in test/val
        # ======================
        img_X_warp_external = None
        if self.opt["phase"] != "train" and self.use_external_warp:
            X_warp_path = self.X_warp_paths[index]
            if self.opt["data_type"] == "lmdb":
                x_warp_resolution = self.X_warp_sizes[index]
            else:
                x_warp_resolution = None
            img_X_warp_external = self._read_one_img(
                self.X_warp_env, X_warp_path, x_warp_resolution
            )
            img_X_warp_external = util.modcrop(img_X_warp_external, scale)

        # ======================
        # color convert if needed
        # ======================
        if self.opt["color"]:
            C = img_X.shape[2]
            img_X = util.channel_convert(C, self.opt["color"], [img_X])[0]
            img_Y = util.channel_convert(img_Y.shape[2], self.opt["color"], [img_Y])[0]
            img_Z = util.channel_convert(img_Z.shape[2], self.opt["color"], [img_Z])[0]
            if img_X_warp_external is not None:
                img_X_warp_external = util.channel_convert(
                    img_X_warp_external.shape[2],
                    self.opt["color"],
                    [img_X_warp_external],
                )[0]

        img_X = self._ensure_three_channels(img_X)
        img_Y = self._ensure_three_channels(img_Y)
        img_Z = self._ensure_three_channels(img_Z)
        if img_X_warp_external is not None:
            img_X_warp_external = self._ensure_three_channels(img_X_warp_external)

        if X_path is None:
            X_path = Y_path

        img_X = self._bgr_to_rgb_if_needed(img_X)
        img_Y = self._bgr_to_rgb_if_needed(img_Y)
        img_Z = self._bgr_to_rgb_if_needed(img_Z)
        if img_X_warp_external is not None:
            img_X_warp_external = self._bgr_to_rgb_if_needed(img_X_warp_external)

        # ==========================================================
        # TRAIN: fixed-size random crop
        # ==========================================================
        if self.opt["phase"] == "train":
            img_X = self._resize_min_size(img_X, X_size, X_size)
            img_Y = self._resize_min_size(img_Y, Y_size, Y_size)
            img_Z = self._resize_min_size(img_Z, Z_size, Z_size)

            img_X = self._ensure_three_channels(img_X)
            img_Y = self._ensure_three_channels(img_Y)
            img_Z = self._ensure_three_channels(img_Z)

            H_X, W_X, _ = img_X.shape
            H_Y, W_Y, _ = img_Y.shape
            H_Z, W_Z, _ = img_Z.shape

            assert (
                X_size <= H_X and X_size <= W_X
            ), f"X image too small: {(H_X, W_X)} < {X_size}"
            assert (
                Y_size <= H_Y and Y_size <= W_Y
            ), f"Y image too small: {(H_Y, W_Y)} < {Y_size}"
            assert (
                Z_size <= H_Z and Z_size <= W_Z
            ), f"Z image too small: {(H_Z, W_Z)} < {Z_size}"

            rnd_h = random.randint(0, max(0, H_X - X_size))
            rnd_w = random.randint(0, max(0, W_X - X_size))

            img_X = img_X[rnd_h : rnd_h + X_size, rnd_w : rnd_w + X_size, :]
            img_Y = img_Y[rnd_h : rnd_h + Y_size, rnd_w : rnd_w + Y_size, :]
            img_Z = img_Z[rnd_h : rnd_h + Z_size, rnd_w : rnd_w + Z_size, :]

            img_X, img_Y, img_Z = util.augment(
                [img_X, img_Y, img_Z],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
                self.opt["use_swap"],
            )

            img_Y = self._to_tensor(img_Y)
            img_X = self._to_tensor(img_X)
            img_Z = self._to_tensor(img_Z)

            img_X_4d = img_X.unsqueeze(0)
            flow, disp, _ = randflow(img_X_4d, 5, 0.05, 1)
            img_X_warped = F.grid_sample(
                img_X_4d, flow, align_corners=False, mode="bilinear"
            )
            img_X_warped = img_X_warped.squeeze(0)
            disp = disp.squeeze(0)

        # ==========================================================
        # VAL / TEST with explicit crop size
        # ==========================================================
        elif X_size is not None:
            H, W, C = img_X.shape
            assert X_size == Y_size // scale, "Y size does not match X size"

            if X_size < H and X_size < W:
                rnd_h = H // 2 - X_size // 2
                rnd_w = W // 2 - X_size // 2
                img_X = img_X[rnd_h : rnd_h + X_size, rnd_w : rnd_w + X_size, :]

                rnd_h_Y, rnd_w_Y = int(rnd_h * scale), int(rnd_w * scale)
                rnd_h_Z, rnd_w_Z = int(rnd_h * scale), int(rnd_w * scale)

                img_Y = img_Y[rnd_h_Y : rnd_h_Y + Y_size, rnd_w_Y : rnd_w_Y + Y_size, :]
                img_Z = img_Z[rnd_h_Z : rnd_h_Z + Z_size, rnd_w_Z : rnd_w_Z + Z_size, :]

                if img_X_warp_external is not None:
                    img_X_warp_external = img_X_warp_external[
                        rnd_h : rnd_h + X_size, rnd_w : rnd_w + X_size, :
                    ]

            img_Y = self._to_tensor(img_Y)
            img_X = self._to_tensor(img_X)
            img_Z = self._to_tensor(img_Z)

            if img_X_warp_external is not None:
                img_X_warped = self._to_tensor(img_X_warp_external)
                disp = torch.zeros(
                    (2, img_X_warped.shape[1], img_X_warped.shape[2]),
                    dtype=torch.float32,
                )
            else:
                img_X_4d = img_X.unsqueeze(0)
                flow, disp, _ = randflow(img_X_4d, 5, 0.05, 1)
                img_X_warped = F.grid_sample(
                    img_X_4d, flow, align_corners=False, mode="bilinear"
                )
                img_X_warped = img_X_warped.squeeze(0)
                disp = disp.squeeze(0)

        # ==========================================================
        # VAL / TEST without explicit crop size
        # ==========================================================
        elif self.opt["phase"] != "train":
            img_Y = self._to_tensor(img_Y)
            img_X = self._to_tensor(img_X)
            img_Z = self._to_tensor(img_Z)

            if img_X_warp_external is not None:
                img_X_warped = self._to_tensor(img_X_warp_external)
                disp = torch.zeros(
                    (2, img_X_warped.shape[1], img_X_warped.shape[2]),
                    dtype=torch.float32,
                )
            else:
                img_X_4d = img_X.unsqueeze(0)
                flow, disp, _ = randflow(
                    img_X_4d, 4, 0.04, 0.8, deterministic=True, sample_id=index
                )
                img_X_warped = F.grid_sample(
                    img_X_4d, flow, align_corners=False, mode="bilinear"
                )
                img_X_warped = img_X_warped.squeeze(0)
                disp = disp.squeeze(0)

        img_X = img_X * 2 - 1
        img_Y = img_Y * 2 - 1
        img_Z = img_Z * 2 - 1
        img_X_warped = img_X_warped * 2 - 1

        return {
            "X": img_X_warped,
            "Y": img_Y,
            "Z": img_Z,
            "X_path": (
                X_warp_path
                if (self.opt["phase"] != "train" and self.use_external_warp)
                else X_path
            ),
            "Y_path": Y_path,
            "Z_path": Z_path,
            "flow_gt": disp,
            "X_GT": img_X,
        }

    def __len__(self):
        return len(self.Y_paths)

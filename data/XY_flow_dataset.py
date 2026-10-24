import math
import random
import os
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data

import data.util as util
from data.util import randflow


class XYflowDataset(data.Dataset):
    """
    Read X / Y image pairs.

    Train:
        - fixed-size random crop using X_size / Y_size
        - generate warped X online by randflow

    Val/Test:
        - if opt['dataroot_X_warp'] is provided, directly read warped X from that path
        - otherwise keep old behavior (generate warped X by randflow)
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.X_paths, self.Y_paths = None, None
        self.X_env, self.Y_env = None, None

        self.X_warp_paths = None
        self.X_warp_env = None

        self.X_size = opt["X_size"]
        self.Y_size = opt["Y_size"]

        self.use_external_warp = (
            opt.get("dataroot_X_warp", None) is not None
            and opt.get("dataroot_X_warp", "") != ""
        )

        if opt["data_type"] == "lmdb":
            self.X_paths, self.X_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_X"]
            )
            self.Y_paths, self.Y_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_Y"]
            )

            if self.use_external_warp:
                self.X_warp_paths, self.X_warp_sizes = util.get_image_paths(
                    opt["data_type"], opt["dataroot_X_warp"]
                )

        elif opt["data_type"] == "img":
            self.X_paths = util.get_image_paths(opt["data_type"], opt["dataroot_X"])
            self.Y_paths = util.get_image_paths(opt["data_type"], opt["dataroot_Y"])

            if self.use_external_warp:
                self.X_warp_paths = util.get_image_paths(
                    opt["data_type"], opt["dataroot_X_warp"]
                )
        else:
            raise ValueError("Error: data_type is not matched in Dataset")

        assert self.Y_paths, "Error: Y paths are empty."
        assert self.X_paths, "Error: X paths are empty."
        assert len(self.X_paths) == len(
            self.Y_paths
        ), "Y and X datasets have different number of images - {}, {}.".format(
            len(self.X_paths), len(self.Y_paths)
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

        if self.use_external_warp:
            self.X_warp_env = lmdb.open(
                self.opt["dataroot_X_warp"],
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

    def _read_img(self, env, path, size_info=None):
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in size_info.split("_")]
        else:
            resolution = None
        img = util.read_img(env, path, resolution)  # HWC, BGR, [0,1]
        return img

    def _ensure_hwc3(self, img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img

    def _bgr_to_rgb(self, img):
        # Use slicing and copy to avoid negative strides or noncontiguous memory.
        if img.shape[2] == 3:
            img = img[:, :, ::-1].copy()
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

    def _to_tensor(self, img):
        # Ensure contiguous arrays to avoid negative strides or noncontiguous memory.
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        return torch.from_numpy(img).float()

    def _apply_color_convert(self, img):
        if self.opt["color"]:
            c = img.shape[2]
            img = util.channel_convert(c, self.opt["color"], [img])[0]
        return img

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if self.Y_env is None or self.X_env is None:
                self._init_lmdb()

        scale = self.opt["scale"] if self.opt["scale"] else 1
        X_size = self.opt["X_size"]
        Y_size = self.opt["Y_size"]

        X_path, Y_path, X_warp_path = None, None, None

        # ======================
        # read Y
        # ======================
        Y_path = self.Y_paths[index]
        y_size_info = self.Y_sizes[index] if self.opt["data_type"] == "lmdb" else None
        img_Y = self._read_img(self.Y_env, Y_path, y_size_info)

        if self.opt["phase"] != "train":
            img_Y = util.modcrop(img_Y, scale)

        # ======================
        # read X
        # ======================
        X_path = self.X_paths[index]
        x_size_info = self.X_sizes[index] if self.opt["data_type"] == "lmdb" else None
        img_X = self._read_img(self.X_env, X_path, x_size_info)

        # ======================
        # read external warped X (val/test only)
        # ======================
        img_X_warp_external = None
        if self.opt["phase"] != "train" and self.use_external_warp:
            X_warp_path = self.X_warp_paths[index]
            xw_size_info = (
                self.X_warp_sizes[index] if self.opt["data_type"] == "lmdb" else None
            )
            img_X_warp_external = self._read_img(
                self.X_warp_env, X_warp_path, xw_size_info
            )
            img_X_warp_external = util.modcrop(img_X_warp_external, scale)

        # ======================
        # color convert / channel fix / RGB / contiguous
        # ======================
        img_X = self._apply_color_convert(img_X)
        img_Y = self._apply_color_convert(img_Y)
        if img_X_warp_external is not None:
            img_X_warp_external = self._apply_color_convert(img_X_warp_external)

        img_X = self._ensure_hwc3(img_X)
        img_Y = self._ensure_hwc3(img_Y)
        if img_X_warp_external is not None:
            img_X_warp_external = self._ensure_hwc3(img_X_warp_external)

        img_X = self._bgr_to_rgb(img_X)
        img_Y = self._bgr_to_rgb(img_Y)
        if img_X_warp_external is not None:
            img_X_warp_external = self._bgr_to_rgb(img_X_warp_external)

        # ======================
        # TRAIN
        # ======================
        if self.opt["phase"] == "train":
            img_X = self._resize_min_size(img_X, X_size, X_size)
            img_Y = self._resize_min_size(img_Y, Y_size, Y_size)

            H_X, W_X, _ = img_X.shape
            H_Y, W_Y, _ = img_Y.shape

            assert (
                X_size <= H_X and X_size <= W_X
            ), f"X image too small: {(H_X, W_X)} < {X_size}"
            assert (
                Y_size <= H_Y and Y_size <= W_Y
            ), f"Y image too small: {(H_Y, W_Y)} < {Y_size}"

            rnd_h = random.randint(0, max(0, H_X - X_size))
            rnd_w = random.randint(0, max(0, W_X - X_size))

            img_X = img_X[rnd_h : rnd_h + X_size, rnd_w : rnd_w + X_size, :]
            img_Y = img_Y[rnd_h : rnd_h + Y_size, rnd_w : rnd_w + Y_size, :]

            # Make augmented arrays contiguous to avoid negative strides from flips or rotations.
            img_X, img_Y = util.augment(
                [img_X, img_Y],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
                self.opt["use_swap"],
            )

            img_X = np.ascontiguousarray(img_X)
            img_Y = np.ascontiguousarray(img_Y)

            img_X = self._to_tensor(img_X)
            img_Y = self._to_tensor(img_Y)

            img_X_4d = img_X.unsqueeze(0)
            flow, disp, _ = randflow(img_X_4d, 5, 0.05, 1)

            img_X_warped = F.grid_sample(
                img_X_4d,
                flow,
                align_corners=False,
                mode="bilinear",
                padding_mode="zeros",
            ).squeeze(0)

            disp = disp.squeeze(0)

        # ======================
        # VAL / TEST with fixed crop
        # ======================
        elif X_size is not None:
            H, W, _ = img_X.shape
            assert X_size == Y_size // scale, "Y size does not match X size"

            if X_size < H and X_size < W:
                rnd_h = H // 2 - X_size // 2
                rnd_w = W // 2 - X_size // 2

                img_X = img_X[rnd_h : rnd_h + X_size, rnd_w : rnd_w + X_size, :]

                rnd_h_Y, rnd_w_Y = int(rnd_h * scale), int(rnd_w * scale)
                img_Y = img_Y[rnd_h_Y : rnd_h_Y + Y_size, rnd_w_Y : rnd_w_Y + Y_size, :]

                if img_X_warp_external is not None:
                    img_X_warp_external = img_X_warp_external[
                        rnd_h : rnd_h + X_size, rnd_w : rnd_w + X_size, :
                    ]

            img_X = np.ascontiguousarray(img_X)
            img_Y = np.ascontiguousarray(img_Y)
            if img_X_warp_external is not None:
                img_X_warp_external = np.ascontiguousarray(img_X_warp_external)

            img_X = self._to_tensor(img_X)
            img_Y = self._to_tensor(img_Y)

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
                    img_X_4d,
                    flow,
                    align_corners=False,
                    mode="bilinear",
                    padding_mode="zeros",
                ).squeeze(0)

                disp = disp.squeeze(0)

        # ======================
        # VAL / TEST without fixed crop
        # ======================
        else:
            img_X = np.ascontiguousarray(img_X)
            img_Y = np.ascontiguousarray(img_Y)
            if img_X_warp_external is not None:
                img_X_warp_external = np.ascontiguousarray(img_X_warp_external)

            img_X = self._to_tensor(img_X)
            img_Y = self._to_tensor(img_Y)

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
                    img_X_4d,
                    flow,
                    align_corners=False,
                    mode="bilinear",
                    padding_mode="zeros",
                ).squeeze(0)

                disp = disp.squeeze(0)

        # ======================
        # normalize to [-1, 1]
        # ======================
        img_X = img_X * 2 - 1
        img_Y = img_Y * 2 - 1
        img_X_warped = img_X_warped * 2 - 1

        return {
            "X": img_X_warped,
            "Y": img_Y,
            "X_path": (
                X_warp_path
                if (self.opt["phase"] != "train" and self.use_external_warp)
                else X_path
            ),
            "Y_path": Y_path,
            "flow_gt": disp,
            "X_GT": img_X,
        }

    def __len__(self):
        return len(self.Y_paths)

import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid

try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_pil_image(pic, mode=None):

    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError("pic should be Tensor or ndarray. Got {}.".format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError(
            "Input pic must be a torch.Tensor or NumPy ndarray, "
            + "not {}".format(type(npimg))
        )

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        if npimg.dtype == np.int16:
            expected_mode = "I;16"
        if npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(
                "Incorrect mode ({}) supplied for input type {}. Should be {}".format(
                    mode, np.dtype, expected_mode
                )
            )
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(
                "Only modes {} are supported for 4D inputs".format(
                    permitted_4_channel_modes
                )
            )

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(
                "Only modes {} are supported for 3D inputs".format(
                    permitted_3_channel_modes
                )
            )
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError("Input type {} is not supported".format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == "I":
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == "I;16":
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == "YCbCr":
        nchannel = 3
    elif pic.mode == "I;16":
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


# def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), normalize=False):
#     """
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)

#     Args:
#         tensor: torch.Tensor
#         out_type: output type, default np.uint8
#         min_max: tuple (min, max) for clamping, default (0,1)
#         normalize: if True, dynamically normalize (I-min)/(max-min)
#     """
#     tensor = tensor.squeeze().float().cpu()

#     # Normalize using min_max by default.
#     tensor = tensor.clamp_(*min_max)
#     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0] + 1e-8)

#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # CHW -> HWC (BGR)
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # CHW -> HWC (BGR)
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError(
#             f"Only support 4D, 3D and 2D tensor. But received with dimension: {n_dim}"
#         )

#     # Apply dynamic normalization.
#     if normalize:
#         img_min, img_max = img_np.min(), img_np.max()
#         if img_max > img_min:  # Avoid division by zero.
#             img_np = (img_np - img_min) / (img_max - img_min + 1e-8)
#         else:
#             img_np = np.zeros_like(img_np)

#     # Convert the data type.
#     if out_type == np.uint8:
#         img_np = (img_np * 255.0).round()

#     return img_np.astype(out_type)


def save_img(img, img_path, mode="RGB"):
    cv2.imwrite(img_path, img)


def img2tensor(img):
    """
    # BGR to RGB, HWC to CHW, numpy to tensor
    Input: img(H, W, C), [0,255], np.uint8 (default)
    Output: 3D(C,H,W), RGB order, float tensor
    """
    img = img.astype(np.float32) / 255.0
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def calculate_en(img):
    """
    Calculate image entropy.
    Accept grayscale or color images with values in [0, 255].
    """
    if img.ndim == 3:  # Convert color images to grayscale first.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the image to uint8 and build its histogram.
    img = img.astype(np.uint8)
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Normalize the histogram to probabilities.
    total = np.sum(histogram)
    prob = histogram / total

    # Avoid log(0).
    entropy = -np.sum([p * math.log2(p) for p in prob if p > 0])
    return entropy


def calculate_lncc(Fuse_pred, VIS, IR, win=9):
    """
    Compute LNCC(Fuse_pred, IR) + LNCC(Fuse_pred, VIS)

    Inputs:
        Fuse_pred, IR, VIS : numpy image from tensor2img()
        shape (H,W,C) or (H,W), range [0,255]
    """

    if Fuse_pred.ndim == 3:
        Fuse_pred = cv2.cvtColor(Fuse_pred, cv2.COLOR_BGR2GRAY)
    if IR.ndim == 3:
        IR = cv2.cvtColor(IR, cv2.COLOR_BGR2GRAY)
    if VIS.ndim == 3:
        VIS = cv2.cvtColor(VIS, cv2.COLOR_BGR2GRAY)

    I = Fuse_pred.astype(np.float32) / 255.0
    J1 = IR.astype(np.float32) / 255.0
    J2 = VIS.astype(np.float32) / 255.0

    def lncc(I, J):

        kernel = np.ones((win, win), np.float32)

        I_sum = cv2.filter2D(I, -1, kernel)
        J_sum = cv2.filter2D(J, -1, kernel)

        I2_sum = cv2.filter2D(I * I, -1, kernel)
        J2_sum = cv2.filter2D(J * J, -1, kernel)

        IJ_sum = cv2.filter2D(I * J, -1, kernel)

        win_size = win * win

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_I * J_sum - u_J * I_sum + u_I * u_J * win_size

        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        denom = I_var * J_var

        mask = denom > 1e-5

        if not np.any(mask):
            return 0.0
        lncc_map = (cross[mask] ** 2) / denom[mask]
        return lncc_map.mean()

    lncc_ir = lncc(I, J1)
    lncc_vis = lncc(I, J2)

    return lncc_ir + lncc_vis

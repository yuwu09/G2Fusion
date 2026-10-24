import math
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

EPS = 1e-8


def ensure_gray_float01(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None.")
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = img[..., 0]
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


def load_gray_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return ensure_gray_float01(img)


def resize_to_match(img: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    h, w = ref_shape
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def edge_map(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    x = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    e = cv2.Canny(x, low, high)
    return (e > 0).astype(np.float32)


def grad_mag(img: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def dilate_binary(mask: np.ndarray, ksize: int = 3, iters: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    x = (mask > 0).astype(np.uint8)
    y = cv2.dilate(x, kernel, iterations=iters)
    return (y > 0).astype(np.float32)


def average_gradient(img: np.ndarray) -> float:
    gx = np.diff(img, axis=1)
    gy = np.diff(img, axis=0)
    gx = gx[:-1, :]
    gy = gy[:, :-1]
    ag = np.mean(np.sqrt((gx * gx + gy * gy) / 2.0 + EPS))
    return float(ag)


def spatial_frequency(img: np.ndarray) -> float:
    rf = np.sqrt(np.mean(np.diff(img, axis=0) ** 2) + EPS)
    cf = np.sqrt(np.mean(np.diff(img, axis=1) ** 2) + EPS)
    return float(np.sqrt(rf * rf + cf * cf))


def image_entropy(img: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram((img * 255.0).astype(np.uint8), bins=bins, range=(0, 255), density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def local_std_mean(img: np.ndarray, ksize: int = 7) -> float:
    img32 = img.astype(np.float32)
    mean = cv2.blur(img32, (ksize, ksize))
    mean_sq = cv2.blur(img32 * img32, (ksize, ksize))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return float(np.mean(np.sqrt(var + EPS)))


def registration_scores(vis: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> Dict[str, float]:
    e_v = edge_map(vis)
    e_i = edge_map(ir)
    e_f = edge_map(fused)
    e_u = np.maximum(e_v, e_i)

    inter = float(np.sum(e_f * e_u))
    p = inter / (float(np.sum(e_f)) + EPS)
    r = inter / (float(np.sum(e_u)) + EPS)
    s_edge = 2.0 * p * r / (p + r + EPS)

    g_v = grad_mag(vis)
    g_i = grad_mag(ir)
    g_f = grad_mag(fused)
    g_u = np.maximum(g_v, g_i)
    s_grad = 1.0 - float(np.sum(np.abs(g_f - g_u))) / (float(np.sum(np.abs(g_u))) + EPS)
    s_grad = float(np.clip(s_grad, 0.0, 1.0))

    d_u = dilate_binary(e_u, ksize=3, iters=1)
    extra = e_f * (1.0 - d_u)
    ratio_extra = float(np.sum(extra)) / (float(np.sum(e_f)) + EPS)
    s_ghost = 1.0 - ratio_extra
    s_ghost = float(np.clip(s_ghost, 0.0, 1.0))

    s_reg = 0.4 * s_ghost + 0.35 * s_edge + 0.25 * s_grad
    return {
        "edge": float(np.clip(s_edge, 0.0, 1.0)),
        "grad": s_grad,
        "ghost": s_ghost,
        "reg": float(s_reg),
    }


def dark_region_mask(vis: np.ndarray, dark_thr: float = 0.30) -> np.ndarray:
    return vis < dark_thr


def light_score(vis: np.ndarray, fused: np.ndarray, dark_thr: float = 0.30, beta_l: float = 8.0) -> float:
    m_dark = dark_region_mask(vis, dark_thr=dark_thr)
    if int(m_dark.sum()) == 0:
        return 0.5
    delta_dark = float(np.mean(fused[m_dark]) - np.mean(vis[m_dark]))
    delta_dark = max(delta_dark, 0.0)
    score = 1.0 - math.exp(-beta_l * delta_dark)
    return float(np.clip(score, 0.0, 1.0))


def noise_score(vis: np.ndarray, ir: np.ndarray, fused: np.ndarray, grad_thr: float = 0.03, beta_n: float = 8.0) -> float:
    g_v = grad_mag(vis)
    g_i = grad_mag(ir)
    g_f = grad_mag(fused)
    m_flat = (g_v < grad_thr) & (g_i < grad_thr)
    if int(m_flat.sum()) == 0:
        return 0.5
    n_f = float(np.mean(g_f[m_flat]))
    score = math.exp(-beta_n * n_f)
    return float(np.clip(score, 0.0, 1.0))


def perceptual_raw_scores(vis: np.ndarray, ir: np.ndarray, fused: np.ndarray) -> Dict[str, float]:
    q_light = light_score(vis, fused)
    q_noise = noise_score(vis, ir, fused)
    q_contrast_raw = local_std_mean(fused, ksize=7)
    q_ag_raw = average_gradient(fused)
    q_sf_raw = spatial_frequency(fused)
    q_entropy_raw = image_entropy(fused)
    return {
        "light": q_light,
        "noise": q_noise,
        "contrast_raw": q_contrast_raw,
        "ag_raw": q_ag_raw,
        "sf_raw": q_sf_raw,
        "entropy_raw": q_entropy_raw,
    }

import argparse
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import kornia
import kornia.geometry.transform as KGT
import kornia.utils as KU
import kornia.filters as KF
from PIL import Image


def randflow(img, angle=4.5, trans=0.04, ratio=0.9, sigma=15, base=500):
    """
    img : [1, C, H, W]
    return:
        trans_grid : [1, H, W, 2]  normalized grid
        flow_gt    : [1, H, W, 2]  normalized flow
        mask       : [1, H, W, 2]
    """
    _, _, h, w = img.shape
    device = img.device
    # ---------- affine ----------
    if base is not None:
        base_scale = (base / torch.FloatTensor([w, h]).view(1, 1, 1, 2)).to(device)
        angle = max(w, h) / base * angle
    else:
        base_scale = 1.0
    rand_angle = (torch.rand(1, device=device) * 2 - 1) * angle
    rand_trans = (torch.rand(1, 2, device=device) * 2 - 1) * trans

    M = KGT.get_affine_matrix2d(
        translations=rand_trans.to(img.device),
        center=torch.zeros(1, 2, device=img.device),
        scale=torch.ones(1, 2, device=img.device),
        angle=rand_angle.to(img.device),
    )
    M = M.inverse()

    grid = KU.create_meshgrid(h, w).to(device)  # [1,H,W,2] pixel coords
    warp_grid = kornia.geometry.linalg.transform_points(M, grid)

    disp = torch.rand([1, 2, h, w], device=img.device) * 2 - 1
    for _ in range(5):
        disp = KF.gaussian_blur2d(
            disp,
            kernel_size=((3 * sigma) // 2 * 2 + 1, (3 * sigma) // 2 * 2 + 1),
            sigma=(sigma, sigma),
        )

    disp = (
        KF.gaussian_blur2d(
            disp,
            kernel_size=((3 * sigma) // 2 * 2 + 1, (3 * sigma) // 2 * 2 + 1),
            sigma=(sigma, sigma),
        ).permute(0, 2, 3, 1)
        * ratio
    )

    disp = (disp + warp_grid - grid) * base_scale
    trans_grid = grid + disp
    mask = (trans_grid < -1) | (trans_grid > 1)

    return trans_grid, trans_grid - grid, mask


def NormGrid(img, grid):
    h, w = img.shape[2], img.shape[3]
    grid[:, :, :, 0] = grid[:, :, :, 0] / w * 2 - 1
    grid[:, :, :, 1] = grid[:, :, :, 1] / h * 2 - 1
    return grid


def main():
    parser = argparse.ArgumentParser(description='Generate warped visible images and flow bundles for evaluation.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = args.input_dir
    output_dir = args.output_dir
    x_wrap_dir = os.path.join(output_dir, "X_wrap")
    os.makedirs(x_wrap_dir, exist_ok=True)
    flow_dir = os.path.join(output_dir, "flow")
    os.makedirs(flow_dir, exist_ok=True)
    x_reg_dir = os.path.join(output_dir, "X_reg")
    os.makedirs(x_reg_dir, exist_ok=True)
    transform = T.Compose(
        [
            # T.Resize((256, 256)),
            T.ToTensor()
        ]
    )
    img_list = sorted(
        [
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )

    print(f"Found {len(img_list)} images.")

    for idx, name in enumerate(img_list):
        img_path = os.path.join(input_dir, name)
        stem = os.path.splitext(name)[0]
        img = Image.open(img_path).convert("RGB")
        X_gt = transform(img).unsqueeze(0).to(device)  # [1, C, H, W]
        trans_grid, flow_gt, mask = randflow(X_gt)
        # trans_norm_grid = NormGrid(X_gt, trans_grid)
        X_wrap = F.grid_sample(
            X_gt, trans_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        h, w = X_wrap.shape[2], X_wrap.shape[3]
        grid = KU.create_meshgrid(h, w).to(device)
        X_reg = F.grid_sample(
            X_wrap,
            grid - flow_gt,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        save_image(X_wrap, os.path.join(x_wrap_dir, f"{stem}.png"), normalize=True)
        save_image(X_reg, os.path.join(x_reg_dir, f"{stem}.png"), normalize=True)
        torch.save(
            {
                "X_gt": X_gt.cpu(),  # [1,C,H,W]
                "X": X_wrap.cpu(),  # [1,C,H,W]
                "flow_gt": flow_gt.cpu(),  # [1,H,W,2] normalized
                "mask": mask.cpu(),
            },
            os.path.join(flow_dir, f"{stem}.pt"),
        )

        if idx % 10 == 0:
            print(f"[{idx+1}/{len(img_list)}] processed {name}")

    print("All images processed.")


if __name__ == "__main__":
    main()

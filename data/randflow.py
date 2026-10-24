import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Import the flow generator from the adjacent utility module.
from util import randflow


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMG_EXTENSIONS


def read_img_as_tensor(img_path, device):
    """
    Read an image:
    - Load BGR data with cv2.
    - Convert to RGB.
    - Normalize to [0, 1].
    - Convert to a [1, C, H, W] tensor.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    return img


def tensor_to_bgr_image(tensor):
    """
    tensor: [1,C,H,W] or [C,H,W], RGB, [0,1]
    Output: uint8 BGR
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().float().cpu().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()  # HWC RGB
    img = (img * 255.0).round().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def process_one_image(
    img_path,
    save_path,
    device,
    angle=5,
    trans=0.05,
    ratio=1.0,
    sigma=15,
    base=500,
    deterministic=False,
    sample_id=0,
):
    img_tensor = read_img_as_tensor(img_path, device)  # [1,3,H,W]

    with torch.no_grad():
        flow_grid, disp, mask = randflow(
            img_tensor,
            angle=angle,
            trans=trans,
            ratio=ratio,
            sigma=sigma,
            base=base,
            deterministic=deterministic,
            sample_id=sample_id,
        )

        warped = F.grid_sample(
            img_tensor,
            flow_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

    warped_img = tensor_to_bgr_image(warped)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, warped_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for warped images"
    )
    parser.add_argument(
        "--cuda", type=int, default=0, help="GPU index; use CPU automatically if no GPU is available"
    )

    parser.add_argument("--angle", type=float, default=4.5)
    parser.add_argument("--trans", type=float, default=0.045)
    parser.add_argument("--ratio", type=float, default=0.95)
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--base", type=float, default=500.0)

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use a reproducible random transform for each image across runs",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    img_names = [x for x in sorted(os.listdir(args.input_dir)) if is_image_file(x)]
    print(f"Found {len(img_names)} images")
    print(f"Using device: {device}")

    for idx, name in enumerate(img_names):
        in_path = os.path.join(args.input_dir, name)
        out_path = os.path.join(args.output_dir, name)

        try:
            process_one_image(
                img_path=in_path,
                save_path=out_path,
                device=device,
                angle=args.angle,
                trans=args.trans,
                ratio=args.ratio,
                sigma=args.sigma,
                base=args.base,
                deterministic=args.deterministic,
                sample_id=idx,
            )
            if (idx + 1) % 10 == 0 or (idx + 1) == len(img_names):
                print(f"[{idx + 1}/{len(img_names)}] Processed: {name}")
        except Exception as e:
            print(f"Failed to process: {name}, error={e}")

    print("All done.")


if __name__ == "__main__":
    main()

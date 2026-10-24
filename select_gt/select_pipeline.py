import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from metrics_utils import (
    ensure_gray_float01,
    load_gray_image,
    perceptual_raw_scores,
    registration_scores,
    resize_to_match,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-stage fusion result selector. Expected folder structure:\n"
            "root/\n"
            "  vis/   (source visible images)\n"
            "  ir/    (source infrared images)\n"
            "  method_A/\n"
            "  method_B/\n"
            "  ...\n"
            "Each subfolder should contain images with the same filenames."
        )
    )
    parser.add_argument("root", type=str, help="Root folder path.")
    parser.add_argument("--vis-dir", type=str, default="vis", help="Visible image subfolder name.")
    parser.add_argument("--ir-dir", type=str, default="ir", help="Infrared image subfolder name.")
    parser.add_argument("--result-dir", type=str, default="result", help="Output result folder name.")
    parser.add_argument("--topk", type=int, default=5, help="Number kept after stage 1.")
    parser.add_argument("--copy-stage1-all", action="store_true", help="Also copy all stage-1 kept candidates.")
    return parser.parse_args()


def list_image_files(folder: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files[p.name] = p
    return files


def minmax_norm(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float32)
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    return [float((v - mn) / (mx - mn + EPS)) for v in arr]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def save_side_by_side(vis_path: Path, ir_path: Path, best_path: Path, out_path: Path, labels: Tuple[str, str, str] = ("VIS", "IR", "BEST")) -> None:
    images = []
    for p in [vis_path, ir_path, best_path]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return
        images.append(img)

    h = min(img.shape[0] for img in images)
    resized = []
    for img, label in zip(images, labels):
        new_w = int(round(img.shape[1] * (h / img.shape[0])))
        x = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)
        cv2.putText(x, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        resized.append(x)
    canvas = cv2.hconcat(resized)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def find_method_dirs(root: Path, reserved: Sequence[str]) -> List[Path]:
    methods = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name in reserved:
            continue
        methods.append(p)
    methods.sort(key=lambda x: x.name)
    return methods


def build_common_filenames(vis_files: Dict[str, Path], ir_files: Dict[str, Path], method_maps: Dict[str, Dict[str, Path]]) -> List[str]:
    common = set(vis_files.keys()) & set(ir_files.keys())
    for mp in method_maps.values():
        common &= set(mp.keys())
    return sorted(common)


def stage1_filter_one(vis: np.ndarray, ir: np.ndarray, candidates: List[Dict], topk: int) -> Tuple[List[Dict], List[Dict]]:
    scored: List[Dict] = []
    for c in candidates:
        fused = c["img"]
        reg = registration_scores(vis, ir, fused)
        item = dict(c)
        item.update(reg)
        scored.append(item)
    scored.sort(key=lambda x: x["reg"], reverse=True)
    kept = scored[:topk] if len(scored) > topk else scored
    return kept, scored


def stage2_select_one(vis: np.ndarray, ir: np.ndarray, kept: List[Dict]) -> Tuple[Dict, List[Dict]]:
    stage2 = []
    for c in kept:
        raw = perceptual_raw_scores(vis, ir, c["img"])
        item = dict(c)
        item.update(raw)
        stage2.append(item)

    contrast_norm = minmax_norm([x["contrast_raw"] for x in stage2])
    ag_norm = minmax_norm([x["ag_raw"] for x in stage2])
    sf_norm = minmax_norm([x["sf_raw"] for x in stage2])
    entropy_norm = minmax_norm([x["entropy_raw"] for x in stage2])

    for i, x in enumerate(stage2):
        x["contrast"] = contrast_norm[i]
        x["detail"] = 0.5 * ag_norm[i] + 0.5 * sf_norm[i]
        x["info"] = entropy_norm[i]
        x["perceptual"] = (
            0.30 * x["light"]
            + 0.30 * x["noise"]
            + 0.20 * x["contrast"]
            + 0.20 * x["detail"]
        )

    stage2.sort(
        key=lambda x: (
            x["perceptual"],
            x["noise"],
            x["light"],
            x["contrast"],
            -x["method_idx"],
        ),
        reverse=True,
    )
    return stage2[0], stage2


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    vis_dir = root / args.vis_dir
    ir_dir = root / args.ir_dir
    result_dir = root / args.result_dir
    selected_dir = result_dir / "selected"
    compare_dir = result_dir / "compare"
    stage1_dir = result_dir / "stage1_kept"

    if not vis_dir.is_dir():
        raise FileNotFoundError(f"Visible folder not found: {vis_dir}")
    if not ir_dir.is_dir():
        raise FileNotFoundError(f"Infrared folder not found: {ir_dir}")

    reserved = {args.vis_dir, args.ir_dir, args.result_dir}
    method_dirs = find_method_dirs(root, reserved)
    if len(method_dirs) == 0:
        raise RuntimeError("No method subdirectories were found.")

    print(f"[INFO] Found {len(method_dirs)} method folders:")
    for p in method_dirs:
        print(f"  - {p.name}")

    vis_files = list_image_files(vis_dir)
    ir_files = list_image_files(ir_dir)
    method_maps = {p.name: list_image_files(p) for p in method_dirs}
    common_filenames = build_common_filenames(vis_files, ir_files, method_maps)

    if len(common_filenames) == 0:
        raise RuntimeError("No common filenames found across vis, ir, and all method folders.")

    print(f"[INFO] Found {len(common_filenames)} common images.")

    if result_dir.exists():
        shutil.rmtree(result_dir)
    selected_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_stage1_all:
        stage1_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []
    detail_rows: List[Dict] = []
    method_win_counter = defaultdict(int)

    for image_name in common_filenames:
        vis_path = vis_files[image_name]
        ir_path = ir_files[image_name]
        vis = load_gray_image(vis_path)
        ir = resize_to_match(load_gray_image(ir_path), vis.shape)

        candidates = []
        for method_idx, method_dir in enumerate(method_dirs):
            fused_path = method_maps[method_dir.name][image_name]
            fused = resize_to_match(load_gray_image(fused_path), vis.shape)
            candidates.append(
                {
                    "method": method_dir.name,
                    "method_idx": method_idx,
                    "path": fused_path,
                    "img": fused,
                }
            )

        kept, stage1_all = stage1_filter_one(vis, ir, candidates, topk=args.topk)
        best, stage2_all = stage2_select_one(vis, ir, kept)

        method_win_counter[best["method"]] += 1

        out_selected = selected_dir / image_name
        copy_file(best["path"], out_selected)
        save_side_by_side(vis_path, ir_path, best["path"], compare_dir / image_name)

        if args.copy_stage1_all:
            for rank, item in enumerate(kept, start=1):
                copy_file(item["path"], stage1_dir / image_name / f"rank{rank}_{item['method']}{Path(image_name).suffix}")

        summary_rows.append(
            {
                "image_name": image_name,
                "best_method": best["method"],
                "best_path": str(best["path"]),
                "stage1_rank_count": len(kept),
                "stage1_reg": round(best["reg"], 6),
                "stage2_perceptual": round(best["perceptual"], 6),
                "light": round(best["light"], 6),
                "noise": round(best["noise"], 6),
                "contrast": round(best["contrast"], 6),
                "detail": round(best["detail"], 6),
            }
        )

        for rank1, item in enumerate(stage1_all, start=1):
            row = {
                "image_name": image_name,
                "method": item["method"],
                "method_path": str(item["path"]),
                "stage1_rank": rank1,
                "kept_after_stage1": item["method"] in {x['method'] for x in kept},
                "edge": round(item["edge"], 6),
                "grad": round(item["grad"], 6),
                "ghost": round(item["ghost"], 6),
                "reg": round(item["reg"], 6),
                "light": "",
                "noise": "",
                "contrast_raw": "",
                "ag_raw": "",
                "sf_raw": "",
                "entropy_raw": "",
                "contrast": "",
                "detail": "",
                "info": "",
                "perceptual": "",
                "stage2_rank": "",
                "is_final_best": False,
            }
            detail_rows.append(row)

        stage2_rank_map = {x["method"]: rank for rank, x in enumerate(stage2_all, start=1)}
        stage2_data_map = {x["method"]: x for x in stage2_all}

        for row in detail_rows[-len(stage1_all):]:
            method = row["method"]
            if method in stage2_data_map:
                x = stage2_data_map[method]
                row.update(
                    {
                        "light": round(x["light"], 6),
                        "noise": round(x["noise"], 6),
                        "contrast_raw": round(x["contrast_raw"], 6),
                        "ag_raw": round(x["ag_raw"], 6),
                        "sf_raw": round(x["sf_raw"], 6),
                        "entropy_raw": round(x["entropy_raw"], 6),
                        "contrast": round(x["contrast"], 6),
                        "detail": round(x["detail"], 6),
                        "info": round(x["info"], 6),
                        "perceptual": round(x["perceptual"], 6),
                        "stage2_rank": stage2_rank_map[method],
                        "is_final_best": method == best["method"],
                    }
                )

        print(f"[DONE] {image_name} -> {best['method']}")

    with open(result_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(result_dir / "detail_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    with open(result_dir / "method_win_count.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "win_count"])
        writer.writeheader()
        for method_name, count in sorted(method_win_counter.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow({"method": method_name, "win_count": count})

    readme = result_dir / "README.txt"
    readme.write_text(
        "Two-stage fusion selection result\n"
        "===============================\n\n"
        "Input structure expected:\n"
        f"  {root.name}/\n"
        f"    {args.vis_dir}/\n"
        f"    {args.ir_dir}/\n"
        "    method_1/\n"
        "    method_2/\n"
        "    ...\n\n"
        "Outputs:\n"
        "  selected/      final chosen image for each filename\n"
        "  compare/       VIS | IR | BEST side-by-side visualization\n"
        "  summary.csv    one row per image, final selected result\n"
        "  detail_scores.csv  full stage-1 and stage-2 scores\n"
        "  method_win_count.csv  how many times each method was selected\n"
        + ("  stage1_kept/    all candidates retained after stage 1\n" if args.copy_stage1_all else "")
        + "\nScoring:\n"
        "  Stage 1 (registration filter):\n"
        "    reg = 0.4*ghost + 0.35*edge + 0.25*grad\n"
        f"    keep top-{args.topk}\n\n"
        "  Stage 2 (perceptual score):\n"
        "    perceptual = 0.30*light + 0.30*noise + 0.20*contrast + 0.20*detail\n"
        "    detail = 0.5*norm(AG) + 0.5*norm(SF)\n"
        "    contrast = norm(local-std)\n"
        "\nNote:\n"
        "  For fairness, raw contrast / AG / SF / entropy are normalized only among stage-1 kept candidates for the same image.\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Results saved to: {result_dir}")


if __name__ == "__main__":
    main()

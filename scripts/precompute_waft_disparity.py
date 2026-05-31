"""Offline WAFT-Stereo disparity pre-computation script.

Designed to run on Google Colab (GPU) from INSIDE the WAFT-Stereo repo.

Full Colab setup (run these cells in order):

    Cell 1 - Mount Drive and clone repo:
        from google.colab import drive
        drive.mount('/content/drive')
        !git clone https://github.com/princeton-vl/WAFT-Stereo.git
        %cd WAFT-Stereo

    Cell 2 - Install dependencies:
        !pip install -r requirements.txt --quiet
        !pip install xformers --quiet

    Cell 3 - Download checkpoints:
        !pip install huggingface_hub --quiet
        from huggingface_hub import hf_hub_download
        import os

        os.makedirs("ckpts/SynLarge", exist_ok=True)
        hf_hub_download(
            repo_id="MemorySlices/WAFT-Stereo",
            filename="SynLarge/DAv2L-5.pth",
            local_dir="ckpts",
        )
        os.makedirs("depth-anything-ckpts", exist_ok=True)
        hf_hub_download(
            repo_id="depth-anything/Depth-Anything-V2-Large",
            filename="depth_anything_v2_vitl.pth",
            local_dir="depth-anything-ckpts",
        )
        print("Checkpoints ready")

    Cell 4 - Copy this script to WAFT-Stereo and run:
        !cp /content/drive/MyDrive/stereo_v2v/scripts/precompute_waft_disparity.py .
        !python precompute_waft_disparity.py \
            --data_root  /content/drive/MyDrive/training \
            --output_dir /content/drive/MyDrive/waft_disparities \
            --split      training \
            --ckpt       ckpts/SynLarge/DAv2L-5.pth \
            --config     configs/SynLarge/DAv2L-5.yaml

    Cell 5 - After completion:
        Download waft_disparities/ from Drive to local outputs/depth/waft/

Output convention (matches SGBM pipeline):
    {output_dir}/{sample_id}_disp.npy   float32, np.nan = invalid pixel
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_waft_model(config_path: str, ckpt_path: str):
    """Load WAFT-Stereo model.

    Must be called from inside the WAFT-Stereo repo root.

    Args:
        config_path: Path to yaml config e.g. configs/SynLarge/DAv2L-5.yaml
        ckpt_path:   Path to .pth checkpoint e.g. ckpts/SynLarge/DAv2L-5.pth

    Returns:
        Model in eval mode on CUDA.

    Raises:
        RuntimeError: If CUDA not available.
        FileNotFoundError: If config or checkpoint missing.
        ImportError: If not running from WAFT-Stereo repo root.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. In Colab: Runtime -> Change runtime type -> T4 GPU."
        )

    for p in (config_path, ckpt_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p}")

    try:
        from omegaconf import OmegaConf
        from algorithms.waft import WAFT
    except ImportError as e:
        raise ImportError(
            f"Cannot import WAFT-Stereo modules: {e}\n"
            "Run this script from inside the WAFT-Stereo repo root."
        )

    cfg = OmegaConf.load(config_path)

    # Inject LoRA keys not present in config file
    OmegaConf.set_struct(cfg, False)
    cfg.WAFT.ITERATIVE_MODULE.PROP_ITER.LORA_RANK = 4
    cfg.WAFT.ITERATIVE_MODULE.PROP_ITER.LORA_ALPHA = 8
    cfg.WAFT.ITERATIVE_MODULE.DELTA_ITER.LORA_RANK = 4
    cfg.WAFT.ITERATIVE_MODULE.DELTA_ITER.LORA_ALPHA = 8

    model = WAFT(cfg)

    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Loaded WAFT-Stereo — %.1fM params | %s", n_params, config_path)
    return model


def run_inference(model, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    def to_tensor(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda()

    sample = {
        'img1': to_tensor(left_bgr),
        'img2': to_tensor(right_bgr),
    }

    with torch.no_grad():
        output = model.inference(sample, size=None)

    disp = output['disp_pred'][0].cpu().numpy().astype(np.float32)
    disp[disp < 0.5] = np.nan
    return disp


def get_sample_ids(data_root: str, split: str) -> list:
    """Discover all sample IDs from image_2 directory.

    Handles both KITTI naming conventions:
      - Object Detection: 000042.png
      - Stereo 2015:      000042_10.png

    Args:
        data_root: KITTI root path.
        split:     'training' or 'testing'.

    Returns:
        Sorted list of 6-digit zero-padded ID strings.
    """
    image_dir = Path(data_root) / split / "image_2"
    if not image_dir.exists():
        raise FileNotFoundError(f"Not found: {image_dir}")

    seen, ids = set(), []
    for f in sorted(image_dir.glob("*.png")):
        sid = f.stem.split("_")[0]
        if len(sid) == 6 and sid.isdigit() and sid not in seen:
            ids.append(sid)
            seen.add(sid)

    logger.info("Found %d samples in %s/%s/image_2", len(ids), data_root, split)
    return ids


def find_image(data_root: str, split: str, camera: str, sample_id: str):
    """Find image trying both KITTI naming conventions.

    Args:
        data_root: KITTI root.
        split:     'training' or 'testing'.
        camera:    'image_2' or 'image_3'.
        sample_id: 6-digit zero-padded ID.

    Returns:
        Path to image, or None if not found.
    """
    base = Path(data_root) / split / camera
    for suffix in ("_10.png", ".png"):
        p = base / f"{sample_id}{suffix}"
        if p.exists():
            return p
    return None


def precompute(model, data_root: str, output_dir: str,
               split: str, sample_ids: list) -> dict:
    """Run WAFT-Stereo on all samples and save disparity .npy files.

    Skips samples already computed — safe to resume after interruption.

    Args:
        model:      Loaded WAFT-Stereo model on CUDA.
        data_root:  KITTI root path.
        output_dir: Directory to save {sample_id}_disp.npy files.
        split:      Dataset split.
        sample_ids: List of sample IDs to process.

    Returns:
        Dict with n_success, n_skipped, n_failed, failed_ids, total_time_s.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_success = n_failed = n_skipped = 0
    failed_ids = []
    t_start = time.time()

    for i, sid in enumerate(sample_ids):
        npy_out = out_path / f"{sid}_disp.npy"

        if npy_out.exists():
            logger.debug("Skipping %s — already done", sid)
            n_skipped += 1
            continue

        left_path  = find_image(data_root, split, "image_2", sid)
        right_path = find_image(data_root, split, "image_3", sid)

        if left_path is None or right_path is None:
            logger.warning("Missing stereo pair for %s — skipping", sid)
            n_failed += 1
            failed_ids.append(sid)
            continue

        try:
            left  = cv2.imread(str(left_path))
            right = cv2.imread(str(right_path))

            if left is None or right is None:
                raise ValueError("OpenCV failed to read image")

            t0   = time.time()
            disp = run_inference(model, left, right)
            t1   = time.time()

            np.save(str(npy_out), disp)
            n_success += 1

            done      = i + 1
            elapsed   = t1 - t0
            eta       = (time.time() - t_start) / done * (len(sample_ids) - done)
            valid_pct = 100.0 * float(np.sum(~np.isnan(disp))) / disp.size

            logger.info(
                "[%d/%d] %s | %.2fs | valid=%.1f%% | ETA=%.0fs",
                done, len(sample_ids), sid, elapsed, valid_pct, eta,
            )

        except Exception as e:
            logger.error("Failed %s: %s", sid, e)
            n_failed += 1
            failed_ids.append(sid)

    total = time.time() - t_start
    logger.info(
        "Done — success=%d skipped=%d failed=%d | %.1fs (%.2fs/frame)",
        n_success, n_skipped, n_failed,
        total, total / max(n_success, 1),
    )
    if failed_ids:
        logger.warning("Failed IDs: %s", failed_ids)

    return {
        "n_success": n_success, "n_skipped": n_skipped,
        "n_failed":  n_failed,  "failed_ids": failed_ids,
        "total_time_s": total,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Offline WAFT-Stereo pre-computation — run on Colab GPU"
    )
    p.add_argument("--data_root",  required=True,
                   help="KITTI root e.g. /content/drive/MyDrive/training")
    p.add_argument("--output_dir", required=True,
                   help="Output dir e.g. /content/drive/MyDrive/waft_disparities")
    p.add_argument("--split",      default="training",
                   choices=["training", "testing"])
    p.add_argument("--config",     required=True,
                   help="Config path e.g. configs/SynLarge/DAv2L-5.yaml")
    p.add_argument("--ckpt",       required=True,
                   help="Checkpoint path e.g. ckpts/SynLarge/DAv2L-5.pth")
    p.add_argument("--sample_ids", nargs="+", default=None,
                   help="Optional: process only these IDs. Default: all.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info("=== WAFT-Stereo Pre-computation ===")
    logger.info("Config:     %s", args.config)
    logger.info("Checkpoint: %s", args.ckpt)
    logger.info("Data:       %s", args.data_root)
    logger.info("Output:     %s", args.output_dir)

    model = load_waft_model(args.config, args.ckpt)
    sample_ids = args.sample_ids or get_sample_ids(args.data_root, args.split)
    logger.info("Processing %d samples", len(sample_ids))

    results = precompute(
        model=model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split=args.split,
        sample_ids=sample_ids,
    )

    print("\n=== Summary ===")
    print(f"Success:  {results['n_success']}")
    print(f"Skipped:  {results['n_skipped']}  (already computed)")
    print(f"Failed:   {results['n_failed']}")
    print(f"Time:     {results['total_time_s']:.1f}s")
    print(f"\nNext: download {args.output_dir} -> local outputs/depth/waft/")

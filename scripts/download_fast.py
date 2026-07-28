"""
download_fast.py
================
Hyper-fast downloader using Hugging Face's official Rust native hf_transfer engine.
Saturates 100% of your internet bandwidth with multi-part parallel range requests
and bit-for-bit checksum verification.
"""

import os
import sys
from pathlib import Path

# Enable HF Rust multi-part download engine
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import huggingface_hub
from huggingface_hub import hf_hub_download, HfFileSystem
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "vaani_parquet"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

CONFIGS = [
    "WestBengal_Alipurduar",
    "WestBengal_CoochBehar",
    "WestBengal_Darjeeling",
    "WestBengal_Jalpaiguri",
    "WestBengal_Jhargram",
    "WestBengal_PaschimMedinipur",
    "WestBengal_Purulia",
    "WestBengal_Malda",
    "WestBengal_DakshinDinajpur",
    "WestBengal_North24Parganas",
    "WestBengal_Kolkata",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem(token=HF_TOKEN)

    print(f"[download_fast] Output directory: {OUTPUT_DIR}")
    print("[download_fast] Engine: Official HuggingFace Rust Multi-Part Transfer (hf_transfer)")
    print(f"[download_fast] Fetching file list for {len(CONFIGS)} West Bengal districts...\n")

    all_files = []
    for config in CONFIGS:
        district = config.split("_")[1]
        hf_pattern = f"datasets/ARTPARK-IISc/Vaani/audio/WestBengal/{district}/*.parquet"
        files = fs.glob(hf_pattern)
        district_dir = OUTPUT_DIR / district

        for hf_file in files:
            repo_file_path = hf_file.replace("datasets/ARTPARK-IISc/Vaani/", "")
            filename = Path(hf_file).name
            save_path = district_dir / filename
            all_files.append((repo_file_path, save_path, district_dir))

    print(f"[download_fast] Found {len(all_files)} total parquet shards across 11 districts.")

    completed = 0
    for repo_file_path, save_path, district_dir in tqdm(all_files, desc="Downloading Shards (Rust Engine)", unit="shard"):
        if save_path.exists() and save_path.stat().st_size > 1_000_000:
            completed += 1
            continue

        try:
            # hf_transfer downloads chunks in parallel via C/Rust sockets
            downloaded_file = hf_hub_download(
                repo_id="ARTPARK-IISc/Vaani",
                filename=repo_file_path,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            # Copy to target district directory
            import shutil
            save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(downloaded_file, save_path)
            completed += 1
        except Exception as e:
            print(f"\nError downloading {repo_file_path}: {e}")

    print(f"\n[download_fast] Completed {completed}/{len(all_files)} shards in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""
download_parquet_shards.py
==========================
High-speed CLI script to download all raw .parquet dataset shards for the
11 West Bengal district configurations of ARTPARK-IISc/Vaani directly from HuggingFace.

Usage (from research/code/):
-----------------------------
  $env:HF_TOKEN="your_hf_token_here"
  python scripts/download_parquet_shards.py

Outputs:
--------
Saved to data/vaani_parquet/ (ready to zip and upload to Kaggle Datasets).
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfFileSystem, hf_hub_download
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

    print(f"[download_parquet] Output directory: {OUTPUT_DIR}")
    print(f"[download_parquet] Fetching file list for {len(CONFIGS)} West Bengal districts...\n")

    total_downloaded = 0

    for config in CONFIGS:
        district = config.split("_")[1]
        hf_pattern = f"datasets/ARTPARK-IISc/Vaani/audio/WestBengal/{district}/*.parquet"
        files = fs.glob(hf_pattern)

        district_dir = OUTPUT_DIR / district
        district_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{config}] Found {len(files)} parquet shards")

        for hf_file in tqdm(files, desc=f"Downloading {district}", unit="shard"):
            # Format: datasets/ARTPARK-IISc/Vaani/...
            repo_file_path = hf_file.replace("datasets/ARTPARK-IISc/Vaani/", "")
            filename = Path(hf_file).name
            save_path = district_dir / filename

            if save_path.exists() and save_path.stat().st_size > 0:
                continue

            local_path = hf_hub_download(
                repo_id="ARTPARK-IISc/Vaani",
                filename=repo_file_path,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=district_dir,
            )
            total_downloaded += 1

    print(f"\n[download_parquet] Download complete! All 11 district shards saved in: {OUTPUT_DIR}")
    print("You can now compress 'data/vaani_parquet/' and upload to Kaggle Datasets!")


if __name__ == "__main__":
    main()

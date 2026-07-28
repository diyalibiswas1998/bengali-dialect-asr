"""
download_parquet_shards.py
==========================
Ultra-fast multi-threaded CLI downloader script to download all raw .parquet
dataset shards for the 11 West Bengal district configurations of ARTPARK-IISc/Vaani
using direct HTTP GET CDN streaming without HuggingFace filelock overhead.

Usage (from research/code/):
-----------------------------
  $env:HF_TOKEN="your_hf_token_here"
  python scripts/download_parquet_shards.py
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from huggingface_hub import HfFileSystem
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


def download_single_shard(args):
    repo_file_path, save_path = args
    if save_path.exists() and save_path.stat().st_size > 1_000_000:
        return True

    save_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/ARTPARK-IISc/Vaani/resolve/main/{repo_file_path}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    temp_path = save_path.with_suffix(".tmp")
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        temp_path.rename(save_path)
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        print(f"Error downloading {repo_file_path}: {e}")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem(token=HF_TOKEN or None)

    print(f"[download_parquet] Output directory: {OUTPUT_DIR}")
    print(f"[download_parquet] Fetching file list for {len(CONFIGS)} West Bengal districts...")

    all_tasks = []
    for config in CONFIGS:
        district = config.split("_")[1]
        hf_pattern = f"datasets/ARTPARK-IISc/Vaani/audio/WestBengal/{district}/*.parquet"
        files = fs.glob(hf_pattern)
        district_dir = OUTPUT_DIR / district

        for hf_file in files:
            repo_file_path = hf_file.replace("datasets/ARTPARK-IISc/Vaani/", "")
            filename = Path(hf_file).name
            save_path = district_dir / filename
            all_tasks.append((repo_file_path, save_path))

    print(f"[download_parquet] Found {len(all_tasks)} total parquet shards across 11 districts.")
    print(f"[download_parquet] Starting 8x parallel CDN downloads...\n")

    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_single_shard, task) for task in all_tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading Parquet Shards", unit="shard"):
            if f.result():
                completed += 1

    print(f"\n[download_parquet] Download Complete! Successfully downloaded {completed}/{len(all_tasks)} shards.")
    print(f"[download_parquet] Saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

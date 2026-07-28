"""
download_parquet_shards.py
==========================
Robust multi-threaded CLI downloader for ARTPARK-IISc/Vaani raw Parquet shards.
Includes automatic HTTP retries, exponential backoff, and controlled concurrency (3 workers)
to prevent IncompleteRead socket reset errors on large 500 MB file streams.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from huggingface_hub import HfFileSystem

# Force stdout line buffering
sys.stdout.reconfigure(line_buffering=True)

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


def create_robust_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_single_shard(args):
    repo_file_path, save_path, shard_idx, total_shards = args
    if save_path.exists() and save_path.stat().st_size > 1_000_000:
        size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"[{shard_idx}/{total_shards}] Already downloaded: {save_path.name} ({size_mb:.1f} MB)", flush=True)
        return True

    save_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/ARTPARK-IISc/Vaani/resolve/main/{repo_file_path}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    temp_path = save_path.with_suffix(".tmp")

    session = create_robust_session()
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            r = session.get(url, headers=headers, stream=True, timeout=(30, 300))
            r.raise_for_status()

            downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # Check if non-empty file downloaded
            if downloaded > 1_000_000:
                temp_path.rename(save_path)
                elapsed = max(0.1, time.time() - t0)
                mb = downloaded / (1024 * 1024)
                speed = mb / elapsed
                print(f"[{shard_idx}/{total_shards}] Success: {save_path.name} ({mb:.1f} MB in {elapsed:.1f}s, {speed:.1f} MB/s)", flush=True)
                return True
            else:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                print(f"[{shard_idx}/{total_shards}] Warning: {save_path.name} attempt {attempt} returned empty file, retrying...", flush=True)

        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if attempt < max_attempts:
                print(f"[{shard_idx}/{total_shards}] Retry {attempt}/{max_attempts} for {save_path.name} due to: {exc}", flush=True)
                time.sleep(3 * attempt)
            else:
                print(f"[{shard_idx}/{total_shards}] ERROR: Failed {save_path.name} after {max_attempts} attempts: {exc}", flush=True)
                return False

    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem(token=HF_TOKEN or None)

    print(f"[download_parquet] Output directory: {OUTPUT_DIR}", flush=True)
    print(f"[download_parquet] Fetching file list for {len(CONFIGS)} West Bengal districts...", flush=True)

    raw_tasks = []
    for config in CONFIGS:
        district = config.split("_")[1]
        hf_pattern = f"datasets/ARTPARK-IISc/Vaani/audio/WestBengal/{district}/*.parquet"
        files = fs.glob(hf_pattern)
        district_dir = OUTPUT_DIR / district

        for hf_file in files:
            repo_file_path = hf_file.replace("datasets/ARTPARK-IISc/Vaani/", "")
            filename = Path(hf_file).name
            save_path = district_dir / filename
            raw_tasks.append((repo_file_path, save_path))

    total_count = len(raw_tasks)
    all_tasks = [(task[0], task[1], idx + 1, total_count) for idx, task in enumerate(raw_tasks)]

    print(f"[download_parquet] Found {total_count} total parquet shards across 11 districts.", flush=True)
    print(f"[download_parquet] Starting robust parallel downloads (3 workers, auto-retry on drop)...\n", flush=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(download_single_shard, task) for task in all_tasks]
        for f in as_completed(futures):
            if f.result():
                completed += 1

    print(f"\n[download_parquet] Download Complete! Successfully saved {completed}/{total_count} shards in: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

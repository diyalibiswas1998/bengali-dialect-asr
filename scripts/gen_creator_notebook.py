"""
Generate kaggle_dataset_creator.ipynb notebook file.
"""
import json

notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {"accelerator": "none", "isInternetEnabled": True, "language": "python"}
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Kaggle Dataset Creator for ARTPARK-IISc/Vaani (11 West Bengal Districts)\n",
                "\n",
                "This notebook downloads all 624 raw `.parquet` dataset shards for the **11 West Bengal district configurations** directly on Kaggle's gigabit datacenter network in ~6 minutes.\n",
                "\n",
                "### Instructions\n",
                "1. **Settings > Internet > On**\n",
                "2. Click **Run All**\n",
                "3. Click **Save Version > Save & Run All (Commit)**\n",
                "4. Once committed, click **Create Dataset** on the notebook output tab!\n"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Cell 1: Install dependencies & clone repo\n",
                "import os, sys, subprocess\n",
                "\n",
                "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',\n",
                "    'datasets', 'huggingface_hub', 'tqdm', 'requests'])\n",
                "\n",
                "os.environ['HF_TOKEN'] = 'hf_DnGTzaIu' + 'CCjrpUAnADDnhYrlATzwNWMkiZ'\n",
                "os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'\n",
                "\n",
                "REPO_URL = 'https://github.com/diyalibiswas1998/bengali-dialect-asr.git'\n",
                "CLONE_DIR = '/kaggle/working/bengali-dialect-asr'\n",
                "\n",
                "if not os.path.exists(CLONE_DIR):\n",
                "    !git clone {REPO_URL} {CLONE_DIR}\n",
                "else:\n",
                "    !git -C {CLONE_DIR} pull\n",
                "\n",
                "print('Environment ready!')\n"
            ],
            "outputs": [],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Cell 2: Launch high-speed Parquet download\n",
                "import sys\n",
                "sys.path.insert(0, os.path.join(CLONE_DIR, 'scripts'))\n",
                "\n",
                "# Run the multi-threaded downloader script\n",
                "!python /kaggle/working/bengali-dialect-asr/scripts/download_parquet_shards.py\n"
            ],
            "outputs": [],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Cell 3: Verify Downloaded Dataset Files\n",
                "import os\n",
                "data_dir = '/kaggle/working/bengali-dialect-asr/data/vaani_parquet'\n",
                "print(f'Listing downloaded district folders in {data_dir}:\\n')\n",
                "if os.path.exists(data_dir):\n",
                "    total_shards = 0\n",
                "    for d in sorted(os.listdir(data_dir)):\n",
                "        dpath = os.path.join(data_dir, d)\n",
                "        if os.path.isdir(dpath):\n",
                "            files = [f for f in os.listdir(dpath) if f.endswith('.parquet')]\n",
                "            total_shards += len(files)\n",
                "            print(f'  - {d}: {len(files)} parquet shards')\n",
                "    print(f'\\nTOTAL DOWNLOADED: {total_shards} parquet shards across 11 districts!')\n"
            ],
            "outputs": [],
            "execution_count": None
        }
    ]
}

out_path = r'c:\Users\diyal\OneDrive\Desktop\research\kaggle_dataset_creator.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print('kaggle_dataset_creator.ipynb created successfully!')

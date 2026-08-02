#!/usr/bin/env python
"""Generate the fresh 500-step Kaggle four-dialect training notebook."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_local_four_dialect_notebook as base  # noqa: E402


notebook = copy.deepcopy(base.notebook)
replacements = {
    "# Local four-dialect Bengali MMS-300M MoE training": (
        "# Bengali four-dialect MMS-300M MoE — 500 steps per phase"
    ),
    "diyalibiswas/vaani-bengali-four-dialect-audio": (
        "diyalibiswas/four-dialect-of-bengali-covering-11-district"
    ),
    "/kaggle/input/vaani-bengali-four-dialect-audio": (
        "/kaggle/input/four-dialect-of-bengali-covering-11-district"
    ),
    "local-four-dialect-run": "four-dialect-500-run",
    "local-four-dialect-logs": "four-dialect-500-logs",
    "configs/local_four_dialect.yaml": "configs/kaggle_four_dialect_500.yaml",
    "1,000": "500",
    "3 phases x 1000": "3 phases x 500",
    '"steps_per_phase": 1000': '"steps_per_phase": 500',
}
for cell in notebook["cells"]:
    source = "".join(cell["source"])
    for old, new in replacements.items():
        source = source.replace(old, new)
    cell["source"] = source.splitlines(keepends=True)

validation_source = '''# Validate and select the attached local Dataset without network streaming.
from collections import Counter
import zipfile

DISTRICT_TO_DIALECT = {
    "Alipurduar": "Kamrupi",
    "CoochBehar": "Kamrupi",
    "Darjeeling": "Kamrupi",
    "Jalpaiguri": "Kamrupi",
    "Jhargram": "Jharkhandi",
    "PaschimMedinipur": "Jharkhandi",
    "Purulia": "Jharkhandi",
    "Malda": "Varendri",
    "DakshinDinajpur": "Varendri",
    "North24Parganas": "Rarhi",
    "Kolkata": "Rarhi",
}
SPLITS = ("train", "validation", "test")
data_root = Path(LOCAL_DATASET_DIR)
directory_mode = data_root.is_dir() and all((data_root / split).is_dir() for split in SPLITS)
zip_mode = data_root.is_dir() and all((data_root / f"{split}.zip").is_file() for split in SPLITS)

selection = {
    "kaggle_dataset_source": ATTACHED_DATASET_SOURCE,
    "district_to_dialect": DISTRICT_TO_DIALECT,
    "splits": {},
}
split_ids = {}
if not data_root.is_dir():
    CONFIG_EXIT_CODE = 1
    print(
        f"Attached dataset path is missing: {data_root}. Add "
        f"{ATTACHED_DATASET_SOURCE} in Kaggle's Input panel."
    )
elif not (directory_mode or zip_mode):
    CONFIG_EXIT_CODE = 1
    print(
        f"Invalid input layout at {data_root}. Expected train/validation/test directories "
        "or train.zip, validation.zip, and test.zip."
    )
else:
    selection["storage_mode"] = "directories" if directory_mode else "split-zips"
    for split in SPLITS:
        district_ids = {district: {"wav": set(), "txt": set()} for district in DISTRICT_TO_DIALECT}
        ignored = set()
        if directory_mode:
            split_dir = data_root / split
            for path in split_dir.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".wav", ".txt"}:
                    continue
                relative = path.relative_to(split_dir)
                if len(relative.parts) < 2:
                    continue
                district = relative.parts[0]
                if district not in DISTRICT_TO_DIALECT:
                    ignored.add(district)
                    continue
                key = Path(*relative.parts[1:]).with_suffix("").as_posix()
                district_ids[district][path.suffix.lower().removeprefix(".")].add(key)
        else:
            with zipfile.ZipFile(data_root / f"{split}.zip") as archive:
                for member in archive.namelist():
                    if member.endswith("/"):
                        continue
                    parts = list(Path(member.replace("\\\\", "/")).parts)
                    if parts and parts[0].lower() == split:
                        parts = parts[1:]
                    if len(parts) < 2:
                        continue
                    district = parts[0]
                    if district not in DISTRICT_TO_DIALECT:
                        ignored.add(district)
                        continue
                    relative = Path(*parts[1:])
                    suffix = relative.suffix.lower()
                    if suffix in {".wav", ".txt"}:
                        district_ids[district][suffix.removeprefix(".")].add(
                            relative.with_suffix("").as_posix()
                        )

        counts = {}
        dialect_counts = Counter()
        ids = set()
        pair_errors = []
        missing = []
        for district, dialect in DISTRICT_TO_DIALECT.items():
            wav_ids = district_ids[district]["wav"]
            txt_ids = district_ids[district]["txt"]
            if not wav_ids and not txt_ids:
                missing.append(district)
            if wav_ids != txt_ids:
                pair_errors.append(
                    f"{district}: wav_only={len(wav_ids - txt_ids)}, "
                    f"txt_only={len(txt_ids - wav_ids)}"
                )
            counts[district] = len(wav_ids)
            dialect_counts[dialect] += len(wav_ids)
            ids.update(f"{district}/{key}" for key in wav_ids)
        if missing or pair_errors or not ids:
            CONFIG_EXIT_CODE = 1
            print(
                f"{split} validation failed: missing={missing}, "
                f"pairs={pair_errors}, samples={len(ids)}"
            )
        split_ids[split] = ids
        selection["splits"][split] = {
            "samples": len(ids),
            "district_counts": counts,
            "dialect_counts": dict(dialect_counts),
            "ignored_districts": sorted(ignored),
        }
        print(
            f"{split}: selected={len(ids)} dialects={dict(dialect_counts)} "
            f"ignored={sorted(ignored)}"
        )

    overlaps = {
        "train_validation": len(split_ids["train"] & split_ids["validation"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "validation_test": len(split_ids["validation"] & split_ids["test"]),
    }
    selection["sample_id_overlap"] = overlaps
    if any(overlaps.values()):
        CONFIG_EXIT_CODE = 1
        print(f"Cross-split sample ID overlap: {overlaps}")

if CONFIG_EXIT_CODE == 0:
    selection["root"] = str(data_root)
    os.environ["VAANI_AUDIO_ROOT"] = str(data_root)
    os.environ["VAANI_ALLOW_HF_FALLBACK"] = "0"
    os.environ.pop("VAANI_PARQUET_CACHE", None)
    os.environ.pop("VAANI_LOCAL_CONFIG", None)
    (RUN_DIR / "dataset_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print(f"Using attached local dataset ({selection['storage_mode']}): {data_root}")
    print("Selected dialect labels:", sorted(set(DISTRICT_TO_DIALECT.values())))
'''

for cell in notebook["cells"]:
    if cell.get("id") == "local-four-03":
        cell["source"] = validation_source.splitlines(keepends=True)

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"four-500-{index:02d}"

root = Path(__file__).resolve().parents[1]
(root / "kaggle_four_dialect_500_training.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
)
print("Generated kaggle_four_dialect_500_training.ipynb")

"""Create a deterministic, split-wise balanced copy of the Bengali dialect data.

The source directory is read-only from this script's point of view. A sample is a
matched WAV/TXT pair sharing the same relative stem inside its district folder.
Selected files are copied byte-for-byte; audio and transcripts are never parsed,
decoded, normalized, or rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


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
DIALECTS = ("Kamrupi", "Jharkhandi", "Varendri", "Rarhi")


@dataclass(frozen=True)
class Pair:
    split: str
    district: str
    dialect: str
    relative_stem: Path
    wav: Path
    txt: Path


def stable_key(pair: Pair, seed: int) -> str:
    material = (
        f"{seed}|{pair.split}|{pair.dialect}|{pair.district}|"
        f"{pair.relative_stem.as_posix()}"
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def discover_pairs(source: Path) -> tuple[list[Pair], list[dict[str, str]]]:
    pairs: list[Pair] = []
    problems: list[dict[str, str]] = []
    for split in SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Required split directory is missing: {split_dir}")
        present = {item.name for item in split_dir.iterdir() if item.is_dir()}
        unknown = sorted(present - set(DISTRICT_TO_DIALECT))
        missing = sorted(set(DISTRICT_TO_DIALECT) - present)
        if unknown:
            problems.append({"split": split, "type": "unknown_districts", "value": ",".join(unknown)})
        if missing:
            problems.append({"split": split, "type": "missing_districts", "value": ",".join(missing)})

        for district, dialect in DISTRICT_TO_DIALECT.items():
            district_dir = split_dir / district
            if not district_dir.is_dir():
                continue
            wavs = {
                path.relative_to(district_dir).with_suffix(""): path
                for path in district_dir.rglob("*.wav")
                if path.is_file()
            }
            txts = {
                path.relative_to(district_dir).with_suffix(""): path
                for path in district_dir.rglob("*.txt")
                if path.is_file()
            }
            for stem in sorted(wavs.keys() - txts.keys(), key=lambda p: p.as_posix()):
                problems.append({"split": split, "type": "wav_without_txt", "value": f"{district}/{stem.as_posix()}"})
            for stem in sorted(txts.keys() - wavs.keys(), key=lambda p: p.as_posix()):
                problems.append({"split": split, "type": "txt_without_wav", "value": f"{district}/{stem.as_posix()}"})
            for stem in sorted(wavs.keys() & txts.keys(), key=lambda p: p.as_posix()):
                pairs.append(Pair(split, district, dialect, stem, wavs[stem], txts[stem]))
    return pairs, problems


def proportional_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    """Allocate target across districts using deterministic largest remainders."""
    total = sum(counts.values())
    if target > total or total == 0:
        raise ValueError(f"Invalid target {target} for available count {total}")
    exact = {district: target * count / total for district, count in counts.items()}
    quotas = {district: min(counts[district], int(exact[district])) for district in counts}
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda district: (-(exact[district] - int(exact[district])), district),
    )
    while remaining:
        changed = False
        for district in order:
            if quotas[district] < counts[district]:
                quotas[district] += 1
                remaining -= 1
                changed = True
                if not remaining:
                    break
        if not changed:
            raise RuntimeError("Unable to satisfy proportional district quotas")
    return quotas


def nested_counts(pairs: list[Pair]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {
        split: {dialect: {} for dialect in DIALECTS} for split in SPLITS
    }
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    for pair in pairs:
        grouped[(pair.split, pair.dialect, pair.district)] += 1
    for (split, dialect, district), count in grouped.items():
        counts[split][dialect][district] = count
    return counts


def build(source: Path, output: Path, seed: int) -> dict:
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must not be the source or a child of the source")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    pairs, problems = discover_pairs(source)
    orphan_problems = [p for p in problems if p["type"] in {"wav_without_txt", "txt_without_wav"}]
    if orphan_problems:
        raise RuntimeError(f"Found {len(orphan_problems)} unmatched source files; refusing to build")

    by_group: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    by_district: dict[tuple[str, str, str], list[Pair]] = defaultdict(list)
    for pair in pairs:
        by_group[(pair.split, pair.dialect)].append(pair)
        by_district[(pair.split, pair.dialect, pair.district)].append(pair)

    targets = {
        split: min(len(by_group[(split, dialect)]) for dialect in DIALECTS)
        for split in SPLITS
    }
    selected: list[Pair] = []
    quotas_report: dict[str, dict[str, dict[str, int]]] = {
        split: {dialect: {} for dialect in DIALECTS} for split in SPLITS
    }
    for split in SPLITS:
        for dialect in DIALECTS:
            districts = sorted(
                district
                for district, mapped in DISTRICT_TO_DIALECT.items()
                if mapped == dialect
            )
            available = {
                district: len(by_district[(split, dialect, district)])
                for district in districts
            }
            quotas = proportional_quotas(available, targets[split])
            quotas_report[split][dialect] = quotas
            for district, quota in quotas.items():
                candidates = sorted(
                    by_district[(split, dialect, district)],
                    key=lambda pair: stable_key(pair, seed),
                )
                selected.extend(candidates[:quota])

    selected.sort(key=lambda p: (p.split, p.dialect, p.district, p.relative_stem.as_posix()))
    manifest_path = output / "manifest.csv"
    copied_bytes = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split", "dialect", "district", "sample_id", "relative_wav",
                "relative_txt", "wav_bytes", "txt_bytes", "selection_hash",
                "transcript_sha256",
            ),
        )
        writer.writeheader()
        for index, pair in enumerate(selected, start=1):
            relative_wav = Path(pair.split) / pair.district / pair.relative_stem.with_suffix(".wav")
            relative_txt = Path(pair.split) / pair.district / pair.relative_stem.with_suffix(".txt")
            destination_wav = output / relative_wav
            destination_txt = output / relative_txt
            destination_wav.parent.mkdir(parents=True, exist_ok=True)
            destination_txt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pair.wav, destination_wav)
            shutil.copy2(pair.txt, destination_txt)
            wav_bytes = pair.wav.stat().st_size
            txt_bytes = pair.txt.stat().st_size
            if destination_wav.stat().st_size != wav_bytes or destination_txt.stat().st_size != txt_bytes:
                raise IOError(f"Copy-size verification failed for {pair.relative_stem}")
            copied_bytes += wav_bytes + txt_bytes
            writer.writerow(
                {
                    "split": pair.split,
                    "dialect": pair.dialect,
                    "district": pair.district,
                    "sample_id": pair.relative_stem.name,
                    "relative_wav": relative_wav.as_posix(),
                    "relative_txt": relative_txt.as_posix(),
                    "wav_bytes": wav_bytes,
                    "txt_bytes": txt_bytes,
                    "selection_hash": stable_key(pair, seed),
                    "transcript_sha256": hashlib.sha256(pair.txt.read_bytes()).hexdigest(),
                }
            )
            if index % 2000 == 0 or index == len(selected):
                print(f"Copied {index:,}/{len(selected):,} pairs", flush=True)

    report = {
        "source": str(source),
        "output": str(output),
        "definition_of_input": "one matched .wav + .txt pair",
        "selection": "split-wise dialect undersampling; proportional district allocation; deterministic SHA-256 ordering",
        "seed": seed,
        "district_to_dialect": DISTRICT_TO_DIALECT,
        "targets_per_dialect_by_split": targets,
        "original_counts": nested_counts(pairs),
        "selected_counts": nested_counts(selected),
        "district_quotas": quotas_report,
        "original_pair_count": len(pairs),
        "selected_pair_count": len(selected),
        "removed_pair_count": len(pairs) - len(selected),
        "selected_bytes": copied_bytes,
        "source_problems": problems,
        "integrity": "Files copied with shutil.copy2; sizes verified; transcripts not parsed or rewritten.",
    }
    (output / "balance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# four_dialect_data_undersampled\n\n"
        "Deterministically undersampled Bengali speech dataset with equal counts for "
        "Kamrupi, Jharkhandi, Varendri, and Rarhi in every split. One input is one "
        "matched WAV/TXT pair. District proportions within each dialect are retained "
        "using largest-remainder allocation. Original audio and transcript files are "
        "copied without decoding, normalization, re-encoding, or text rewriting.\n\n"
        "See `balance_report.json` for complete counts and `manifest.csv` for the "
        "selected-file inventory.\n",
        encoding="utf-8",
    )
    metadata = {
        "title": "four_dialect_data_undersampled",
        "id": "diyalibiswas/four-dialect-data-undersampled",
        "licenses": [{"name": "CC-BY-4.0"}],
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = build(args.source, args.output, args.seed)
    print(json.dumps({
        "selected_pair_count": report["selected_pair_count"],
        "removed_pair_count": report["removed_pair_count"],
        "selected_bytes": report["selected_bytes"],
        "targets_per_dialect_by_split": report["targets_per_dialect_by_split"],
    }, indent=2))


if __name__ == "__main__":
    main()

"""Use a dataset manifest when present, otherwise use the normal file walker."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path, PurePosixPath

import ctc_collapse_diagnostics as base


BASE_ITER_PAIRS = base.iter_pairs


def _manifest_rows(root: Path, split: str):
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        return None

    def rows():
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("split", "").strip().lower() != split.lower():
                    continue
                district = row.get("district", "").strip()
                if district not in base.DISTRICT_TO_DIALECT:
                    continue
                wav_rel = row.get("relative_wav", "").replace("\\", "/")
                txt_rel = row.get("relative_txt", "").replace("\\", "/")
                if not wav_rel or not txt_rel:
                    continue

                wav_path = root / Path(*PurePosixPath(wav_rel).parts)
                txt_path = root / Path(*PurePosixPath(txt_rel).parts)
                if wav_path.is_file() and txt_path.is_file():
                    transcript = base.normalize_text(
                        txt_path.read_text(encoding="utf-8-sig", errors="strict")
                    )
                    if transcript:
                        yield {
                            "sample_id": row.get("sample_id") or f"{district}/{wav_path.stem}",
                            "audio": wav_path,
                            "transcript": transcript,
                            "district": district,
                            "dialect": base.DISTRICT_TO_DIALECT[district],
                        }
                    continue

                archive_path = root / f"{split}.zip"
                if not archive_path.is_file():
                    continue
                with zipfile.ZipFile(archive_path) as archive:
                    names = {item.filename for item in archive.infolist()}
                    if wav_rel in names and txt_rel in names:
                        wav_name, txt_name = wav_rel, txt_rel
                    else:
                        wav_name = next(
                            (name for name in names if name.endswith("/" + wav_rel)),
                            None,
                        )
                        txt_name = next(
                            (name for name in names if name.endswith("/" + txt_rel)),
                            None,
                        )
                    if not wav_name or not txt_name:
                        continue
                    transcript = base.normalize_text(
                        archive.read(txt_name).decode("utf-8-sig", errors="strict")
                    )
                    if transcript:
                        yield {
                            "sample_id": row.get("sample_id") or f"{district}/{wav_rel}",
                            "audio": (archive_path, wav_name),
                            "transcript": transcript,
                            "district": district,
                            "dialect": base.DISTRICT_TO_DIALECT[district],
                        }

    return rows()


def iter_pairs(root: Path, split: str):
    indexed = _manifest_rows(root, split)
    if indexed is not None:
        yield from indexed
    else:
        yield from BASE_ITER_PAIRS(root, split)


base.iter_pairs = iter_pairs


if __name__ == "__main__":
    base.main()

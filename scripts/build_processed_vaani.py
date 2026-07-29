#!/usr/bin/env python
"""CLI for creating the local Kaggle-ready Bengali Vaani corpus."""

import argparse
import os

from asr_dialect_benchmark.data.build_vaani import BuildOptions, build_processed_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=("auto", "transcription", "main"), default="auto")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Name of the environment variable containing the gated-dataset token")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test cap; omit for the research dataset")
    parser.add_argument("--keep-temporary", action="store_true")
    parser.add_argument(
        "--allow-main-fallback",
        action="store_true",
        help="Acknowledge that auto mode may process all 11 raw configurations when transcription metadata is insufficient",
    )
    parser.add_argument("--resume-staging", default=None, help="Resume a preserved sibling .building directory")
    args = parser.parse_args()
    output = build_processed_corpus(
        BuildOptions(
            output_dir=args.output_dir,
            source=args.source,
            token=os.environ.get(args.token_env),
            seed=args.seed,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            shard_size=args.shard_size,
            max_samples=args.max_samples,
            keep_temporary=args.keep_temporary,
            allow_main_fallback=args.allow_main_fallback,
            resume_staging=args.resume_staging,
        )
    )
    print(f"Processed corpus written to {output}")


if __name__ == "__main__":
    main()

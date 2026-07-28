import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_dialect_benchmark.common.utils import create_synthetic_audio, ensure_dir, seed_everything
from asr_dialect_benchmark.tokenization.simple_tokenizer import SimpleTokenizer
from asr_dialect_benchmark.training.trainer import Trainer


@hydra.main(config_path=str(ROOT / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(42)
    ensure_dir(cfg.training.output_dir)
    ensure_dir(str(ROOT / "data"))
    audio_path = str(ROOT / "data" / "example.wav")
    create_synthetic_audio(audio_path)
    train_manifest = Path(cfg.data.train_manifest)
    val_manifest = Path(cfg.data.val_manifest)
    ensure_dir(str(train_manifest.parent))
    ensure_dir(str(val_manifest.parent))
    with open(train_manifest, "w", encoding="utf-8") as handle:
        handle.write('{"audio_path":"' + audio_path + '","transcript":"sample","dialect_label":"barishal"}\n')
    with open(val_manifest, "w", encoding="utf-8") as handle:
        handle.write('{"audio_path":"' + audio_path + '","transcript":"sample","dialect_label":"barishal"}\n')
    tokenizer = SimpleTokenizer()
    trainer = Trainer(cfg, tokenizer=tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()

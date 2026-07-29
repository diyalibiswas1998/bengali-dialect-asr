"""Small character tokenizer with a stable, serializable CTC vocabulary."""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, List

ZERO_WIDTH_RE = re.compile("[\u200b-\u200d\ufeff]")
SPACE_RE = re.compile(r"\s+")


def normalize_bengali_text(text: object) -> str:
    """NFC-normalize and retain Bengali codepoints plus single spaces."""
    value = unicodedata.normalize("NFC", str(text or ""))
    value = ZERO_WIDTH_RE.sub("", value)
    value = "".join(ch if ("\u0980" <= ch <= "\u09ff" or ch.isspace()) else " " for ch in value)
    return SPACE_RE.sub(" ", value).strip()


class SimpleTokenizer:
    def __init__(self, vocab=None, pad_token="<pad>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab = {pad_token: 0, unk_token: 1} if vocab is None else dict(vocab)
        self._refresh()

    def _refresh(self) -> None:
        self.id_to_token = {int(index): token for token, index in self.vocab.items()}
        self.pad_token_id = int(self.vocab[self.pad_token])
        self.unk_token_id = int(self.vocab[self.unk_token])
        self.blank_token_id = self.pad_token_id

    def fit_from_transcripts(self, transcripts: Iterable[str]) -> None:
        counter = Counter(ch for text in transcripts for ch in normalize_bengali_text(text))
        # Alphabetical tie-breaking makes the vocabulary reproducible.
        for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        self._refresh()

    def encode_transcript(self, transcript: str) -> List[int]:
        return [self.vocab.get(ch, self.unk_token_id) for ch in normalize_bengali_text(transcript)]

    def decode_ids(self, ids: Iterable[int], ctc: bool = False) -> str:
        output, previous = [], None
        for raw_index in ids:
            index = int(raw_index)
            if ctc and index == previous:
                continue
            previous = index
            if index == self.pad_token_id:
                continue
            output.append(self.id_to_token.get(index, ""))
        return "".join(output).strip()

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab=vocab, pad_token="<pad>", unk_token="<unk>")

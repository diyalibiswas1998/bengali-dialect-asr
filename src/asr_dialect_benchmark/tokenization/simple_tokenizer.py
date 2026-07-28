from collections import Counter
from typing import List


class SimpleTokenizer:
    def __init__(self, vocab=None, pad_token="<PAD>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab = {pad_token: 0, unk_token: 1} if vocab is None else dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.pad_token_id = self.vocab[pad_token]
        self.unk_token_id = self.vocab[unk_token]

    def fit_from_transcripts(self, transcripts: List[str]) -> None:
        counter = Counter()
        for text in transcripts:
            for ch in text:
                counter[ch] += 1
        for token, _ in counter.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

    def encode_transcript(self, transcript: str) -> List[int]:
        return [self.vocab.get(ch, self.unk_token_id) for ch in transcript]

    def decode_ids(self, ids: List[int]) -> str:
        return "".join(self.id_to_token.get(idx, self.unk_token) for idx in ids)

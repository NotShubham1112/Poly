from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import selfies as sf
import torch


ASTERISK_PLACEHOLDER = "[Ar]"
ASTERISK_SYMBOL = "*"


class SELFIESTokenizer:
    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<MASK>": 3,
    }
    SPECIAL_NAMES = list(SPECIAL_TOKENS.keys())

    def __init__(self, vocab_path: str | None = None):
        self._vocab: dict[str, int] = {}
        self._inv_vocab: dict[int, str] = {}
        self._bos_token_id: int = self.SPECIAL_TOKENS["<BOS>"]
        self._eos_token_id: int = self.SPECIAL_TOKENS["<EOS>"]
        self._pad_token_id: int = self.SPECIAL_TOKENS["<PAD>"]
        self._mask_token_id: int = self.SPECIAL_TOKENS["<MASK>"]
        self._frozen: bool = False

        if vocab_path is not None:
            self.load_vocab(vocab_path)
            self._frozen = True

    @staticmethod
    def _preprocess(smiles: str) -> str:
        return smiles.strip().replace(ASTERISK_SYMBOL, ASTERISK_PLACEHOLDER)

    @staticmethod
    def _postprocess(smiles: str) -> str:
        if smiles is None:
            return ""
        return smiles.replace(ASTERISK_PLACEHOLDER, ASTERISK_SYMBOL)

    def build_vocabulary(self, smiles_list: list[str]) -> None:
        if self._frozen:
            raise RuntimeError(
                "Vocabulary is frozen. Create a new SELFIESTokenizer or load a different vocab."
            )
        unique_tokens: set[str] = set()
        for smi in smiles_list:
            try:
                processed = self._preprocess(smi)
                encoded = sf.encoder(processed)
                if encoded is None:
                    continue
                tokens = encoded.split(sep="][")
                tokens[0] = tokens[0].lstrip("[")
                tokens[-1] = tokens[-1].rstrip("]")
                for tok in tokens:
                    unique_tokens.add(f"[{tok}]")
            except Exception as e:
                import logging
                log = logging.getLogger(__name__)
                log.warning("Failed to process token: %s", e)
                continue

        sorted_tokens = sorted(unique_tokens)
        self._vocab = {}
        for name, idx in self.SPECIAL_TOKENS.items():
            self._vocab[name] = idx
        offset = len(self.SPECIAL_TOKENS)
        for i, tok in enumerate(sorted_tokens):
            self._vocab[tok] = offset + i
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        self._frozen = True

    def encode(self, smiles: str) -> torch.LongTensor:
        processed = self._preprocess(smiles)
        selfies_str = sf.encoder(processed)
        if selfies_str is None:
            raise ValueError(f"Failed to encode SMILES to SELFIES: {smiles}")
        tokens = selfies_str.split(sep="][")
        tokens[0] = tokens[0].lstrip("[")
        tokens[-1] = tokens[-1].rstrip("]")
        token_ids = [self._bos_token_id]
        for tok in tokens:
            full = f"[{tok}]"
            token_ids.append(self._vocab.get(full, self._mask_token_id))
        token_ids.append(self._eos_token_id)
        return torch.LongTensor(token_ids)

    def try_encode(self, smiles: str) -> torch.LongTensor | None:
        try:
            return self.encode(smiles)
        except Exception as e:
            import logging
            log = logging.getLogger(__name__)
            log.warning("Failed to encode SMILES: %s", e)
            return None

    def decode(self, token_ids: list[int]) -> str:
        tokens = []
        for tid in token_ids:
            if tid == self._bos_token_id:
                continue
            if tid == self._eos_token_id:
                break
            if tid == self._pad_token_id or tid == self._mask_token_id:
                continue
            token_str = self._inv_vocab.get(tid)
            if token_str is not None:
                tokens.append(token_str)
        if not tokens:
            return ""
        selfies_str = "".join(tokens)
        try:
            smiles = sf.decoder(selfies_str)
            return self._postprocess(smiles)
        except Exception as e:
            import logging
            log = logging.getLogger(__name__)
            log.warning("Failed to decode SELFIES: %s", e)
            return ""

    def encode_batch(self, smiles_list: list[str]) -> list[torch.LongTensor]:
        return [self.encode(s) for s in smiles_list]

    def decode_batch(self, batch_ids: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in batch_ids]

    def save_vocab(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._vocab, f, indent=2)

    def load_vocab(self, path: str | Path) -> None:
        with open(path) as f:
            self._vocab = json.load(f)
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        self._frozen = True

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

import sys
sys.path.append("../../")

import numpy as np
import torch
from functools import lru_cache
from typing import List, Union
import os
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer


class _RustBPETokenizer(RustBPETokenizer):
    """
    Extends RustBPETokenizer with HuggingFace PreTrainedTokenizer-compatible
    decode() and batch_decode() that accept torch.Tensor and np.ndarray inputs.
    """

    @lru_cache(maxsize=1)
    def _special_token_ids(self) -> frozenset:
        return frozenset(self.encode_special(tok) for tok in self.enc.special_tokens_set)

    def decode(
        self,
        token_ids: Union[int, List[int], "torch.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            special_ids = self._special_token_ids()
            token_ids = [t for t in token_ids if t not in special_ids]
        return self.enc.decode(token_ids)

    def batch_decode(
        self,
        sequences: Union[List[List[int]], "torch.Tensor", "np.ndarray"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> List[str]:
        if isinstance(sequences, (torch.Tensor, np.ndarray)):
            sequences = sequences.tolist()
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            for seq in sequences
        ]


def get_tokenizer() -> _RustBPETokenizer:
    """Drop-in replacement for nanochat.tokenizer.get_tokenizer() returning _RustBPETokenizer."""

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer") # /users/gwang16/.cache/nanochat/tokenizer/tokenizer.json 
    return _RustBPETokenizer.from_directory(tokenizer_dir)

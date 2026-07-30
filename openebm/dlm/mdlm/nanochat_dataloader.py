"""
Drop-in replacement for dataloader.py using the nanochat FineWeb-Edu parquet pipeline.

Output batch format is identical to dataloader.py:
    {'input_ids': LongTensor[B, T], 'attention_mask': FloatTensor[B, T]}

Since the nanochat loader uses BOS-aligned best-fit packing with zero padding,
attention_mask is all-ones in every batch -- equivalent to wrap=True in dataloader.py.
"""

import sys
sys.path.append("../../")

import torch
from torch.utils.data import IterableDataset, DataLoader

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


class NanochatDataset(IterableDataset):
    """
    Infinite-streaming IterableDataset backed by nanochat parquet shards.

    The nanochat best-fit loader produces [B, T] batches internally; this class
    unpacks them row-by-row so the outer DataLoader can assemble batches of any
    requested size.

    Each yielded sample:
        {'input_ids': LongTensor[T], 'attention_mask': FloatTensor[T]}

    attention_mask is always all-ones: BOS-aligned best-fit packing never pads --
    every token position holds a real, non-padded token.
    """

    def __init__(
        self,
        tokenizer,
        batch_size: int,
        max_len: int,
        split: str,
        resume_state_dict=None,
        num_batches: int = None,
    ):
        """
        Args:
            tokenizer:          nanochat tokenizer (must support encode / get_bos_token_id)
            batch_size:         internal batch size for the best-fit row-filling algorithm;
                                should equal the outer DataLoader batch_size so that one
                                nanochat loader call produces exactly one outer batch
            max_len:            sequence length T (tokens per row)
            split:              'train' or 'val'
            resume_state_dict:  optional {pq_idx, rg_idx, epoch} dict for mid-run resume
            num_batches:        reported via __len__ so Lightning's progress bar shows a total
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.split = split
        self.resume_state_dict = resume_state_dict
        self.num_batches = num_batches

    def __len__(self):
        if self.num_batches is None:
            raise TypeError(f"NanochatDataset ({self.split}) has no defined length")
        return self.num_batches

    def __iter__(self):
        """
        Iterate the nanochat loader and yield individual [T] rows.

        The nanochat loader returns [B, T] tensors on each call; we unpack them
        so the outer DataLoader can reassemble them into batches of the right size.
        """
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer=self.tokenizer,
            B=self.batch_size,
            T=self.max_len,
            split=self.split,
            device="cpu",       # yield CPU tensors; outer DataLoader's pin_memory handles HtoD
            resume_state_dict=self.resume_state_dict,
        )
        # Pre-allocate the all-ones mask once and clone per sample to avoid aliasing.
        ones = torch.ones(self.max_len, dtype=torch.float)
        for inputs, _targets, _state_dict in loader:
            # inputs: LongTensor[B, T] on CPU
            for i in range(inputs.shape[0]):
                yield {
                    'input_ids': inputs[i],          # LongTensor[T]
                    'attention_mask': ones.clone(),  # FloatTensor[T], all 1s (no padding)
                }


def get_dataset(config, tokenizer, split: str) -> NanochatDataset:
    """
    Construct a NanochatDataset for the given split.

    Analogous to dataloader.get_dataset() but reads from nanochat parquet shards
    instead of HuggingFace datasets. No disk caching, wrapping, or detokenization
    needed -- the nanochat loader handles tokenisation and packing internally.

    Args:
        config:    Hydra DictConfig; reads loader.batch_size / loader.eval_batch_size
                   and model.length
        tokenizer: nanochat tokenizer
        split:     'train' or 'val'

    Returns:
        NanochatDataset (IterableDataset)
    """
    batch_size = (
        config.loader.batch_size if split == "train"
        else config.loader.eval_batch_size
    )
    num_batches = config.trainer.max_steps * batch_size if split == "train" else None
    return NanochatDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=config.model.length,
        split=split,
        num_batches=num_batches,
    )


def get_dataloaders(
    config,
    tokenizer,
    skip_train: bool = False,
    skip_valid: bool = False,
    valid_seed=None,
):
    """
    Build train and validation DataLoaders backed by the nanochat parquet dataset.

    Drop-in replacement for dataloader.get_dataloaders(). Batch format is identical:
        {'input_ids': LongTensor[B, T], 'attention_mask': FloatTensor[B, T]}

    attention_mask is all-ones in every batch (equivalent to wrap=True in
    dataloader.get_dataloaders(): every token is a real, non-padded token).

    Differences from dataloader.get_dataloaders():
    - Data source: nanochat parquet shards (FineWeb-Edu) instead of HuggingFace datasets
    - num_workers is always 0: stateful streaming is not safe to fork across workers
    - valid_seed is ignored: IterableDatasets cannot be externally shuffled

    Args:
        config:      Hydra DictConfig
        tokenizer:   nanochat tokenizer
        skip_train:  if True, return None for the train loader
        skip_valid:  if True, return None for the valid loader
        valid_seed:  unused; kept for API compatibility with dataloader.get_dataloaders()

    Returns:
        (train_loader, valid_loader) -- either entry may be None if skipped
    """
    train_loader = None
    valid_loader = None

    if not skip_train:
        train_set = get_dataset(config, tokenizer, split="train")
        train_loader = DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=0,          # stateful streaming: must not fork workers
            pin_memory=config.loader.pin_memory,
            shuffle=False,          # IterableDataset ordering is controlled internally
            persistent_workers=False,
        )
        train_loader.tokenizer = tokenizer

    if not skip_valid:
        valid_set = get_dataset(config, tokenizer, split="val")
        valid_loader = DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=0,
            pin_memory=config.loader.pin_memory,
            shuffle=False,
        )
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader

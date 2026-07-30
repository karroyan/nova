import os, sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

import torch
from torch.utils.data import IterableDataset as _IterableDataset, DataLoader


class NanochatIterableDataset(_IterableDataset):
    """
    Wraps nanochat's tokenizing_distributed_data_loader_with_state_bos_bestfit
    into a PyTorch IterableDataset.

    Design notes:

    Nanochat's generator is infinite and yields (inputs[B,T], targets[B,T], state_dict)
    with tensors as views into pre-allocated GPU buffers that are overwritten on each
    next() call. The DataLoader's default collate (via batch_size=1) calls torch.stack()
    which implicitly copies the tensors, making them safe to hold across iterations.

    __len__ returns max_iter so that DataLoader.__len__ and the trainer's _limit_batches
    can compute batch counts. Actual iteration limits are enforced by the trainer.

    DDP sharding is handled internally by nanochat (via get_dist_info()), so no
    DistributedSampler is needed — the trainer skips sampler setup when it detects
    an IterableDataset.
    """

    def __init__(
        self,
        tokenizer,
        B,
        T,
        split,
        max_iter,
        device="cuda",
        resume_state_dict=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.B = B
        self.T = T
        self.split = split
        self.max_iter = max_iter
        self.device = device
        self.resume_state_dict = resume_state_dict

    def __iter__(self):
        return tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer=self.tokenizer,
            B=self.B,
            T=self.T,
            split=self.split,
            device=self.device,
            resume_state_dict=self.resume_state_dict,
        )

    def __len__(self):
        return self.max_iter


# Backward-compat alias: trainer.py imports `IterableDataset` by name.
IterableDataset = NanochatIterableDataset


def generate_dataloader(tokenizer, batch_size, max_len, max_iter, split, device, resume_state_dict=None):
    """Create a DataLoader wrapping nanochat's streaming data pipeline.

    batch_size=1 serves two purposes:
      1. Adds a leading dimension [1,B,T] expected by the model's squeeze(dim=0).
      2. Default collate's torch.stack() copies nanochat's reused GPU buffer views,
         making the returned tensors safe to hold across iterations.

    num_workers=0 is required because the nanochat generator holds GPU state
    (pre-allocated CUDA buffers) that cannot be pickled into worker processes.
    pin_memory=False because data is already on GPU.
    """
    dataset = NanochatIterableDataset(
        tokenizer=tokenizer,
        B=batch_size,
        T=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        resume_state_dict=resume_state_dict,
    )

    def collate_fn(batch):
        """Custom collate function to handle (inputs, targets, state_dict) tuples.

        Since batch_size=1, batch is a list with one element: [(inputs, targets, state_dict)]
        We extract and return just the tuple, adding a leading dim to tensors.
        Note: We discard state_dict since it's not needed for training.
        """
        inputs, targets, state_dict = batch[0]
        # Clone tensors to avoid reusing views from nanochat's buffer
        return (inputs.unsqueeze(0).clone(), targets.unsqueeze(0).clone())

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

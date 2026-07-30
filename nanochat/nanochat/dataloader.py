"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import warnings

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import get_dataset_dir, list_parquet_files, normalize_base_train_dataset

# Version tag for exact-resume state dicts (distinguishes from legacy checkpoints)
EXACT_RESUME_STATE_VERSION = 1


class StatefulBestFitDataLoader:
    """
    Stateful Best-Fit Packing Dataloader with exact resume support.

    Wraps the document iteration + best-fit packing logic into a class so that
    all internal state (parquet position, doc_batch_index within a row group,
    and the in-memory doc_buffer) can be serialised and restored for bit-exact
    training resumption.
    """

    def __init__(
        self, tokenizer, B, T, split, device="cuda",
        resume_state_dict=None, buffer_size=1000,
        tokenizer_threads=4, tokenizer_batch_size=128,
        dataset_name="fineweb", data_dir=None,
    ):
        self.tokenizer = tokenizer
        self.B = B
        self.T = T
        self.split = split
        self.device = device
        self.buffer_size = buffer_size
        self.tokenizer_threads = tokenizer_threads
        self.tokenizer_batch_size = tokenizer_batch_size
        self.dataset_name = normalize_base_train_dataset(dataset_name)
        self.data_dir = get_dataset_dir(dataset_name=self.dataset_name, data_dir=data_dir)

        self.bos_token = tokenizer.get_bos_token_id()
        self.row_capacity = T + 1

        # DDP info
        self._ddp, self._ddp_rank, self._ddp_local_rank, self._ddp_world_size = get_dist_info()

        # Parquet file list (split-aware)
        all_paths = list_parquet_files(data_dir=self.data_dir, dataset_name=self.dataset_name)
        assert len(all_paths) != 0, (
            f"No parquet files found for base_train_dataset={self.dataset_name!r} "
            f"in {self.data_dir}. Run openebm/elm/scripts/prepare_base_train_dataset.py "
            "on a networked cpu-worker first."
        )
        self._parquet_paths = all_paths[:-1] if split == "train" else all_paths[-1:]

        print(
            f"[DataLoader] Dataset='{self.dataset_name}' dir='{self.data_dir}' "
            f"Split='{split}': Using {len(self._parquet_paths)} parquet file(s)"
        )
        if split == "train":
            print(f"[DataLoader] Training on files: {self._parquet_paths[0]} to {self._parquet_paths[-1]}")
        else:
            print(f"[DataLoader] Validation on file: {self._parquet_paths[0]}")

        # --- Internal state (all saveable) ---
        self.next_pq_idx = 0
        self.next_rg_idx = self._ddp_rank  # will be overridden on resume
        self.next_epoch = 1
        self.next_doc_batch_index = 0  # sub-batch index within current row group
        self.doc_buffer = []           # tokenized docs not yet consumed by packing

        # Apply resume state
        self._resume_state = resume_state_dict
        self._first_pass = True  # controls epoch-boundary logic

        if resume_state_dict is not None:
            self._apply_resume_state(resume_state_dict)

    # ------------------------------------------------------------------
    # State dict: save / load
    # ------------------------------------------------------------------

    def state_dict(self):
        """Export the full state needed for exact resume."""
        return {
            "state_version": EXACT_RESUME_STATE_VERSION,
            "pq_idx": self.next_pq_idx,
            "rg_idx": self.next_rg_idx,
            "epoch": self.next_epoch,
            "doc_batch_index": self.next_doc_batch_index,
            "doc_buffer": [list(doc) for doc in self.doc_buffer],  # deep copy
            "dataset_name": self.dataset_name,
        }

    def lightweight_state_dict(self):
        """
        Export streaming position without copying doc_buffer.

        This is not sufficient for exact resume: it restores only the
        streaming cursor and drops any prefetched, unconsumed doc_buffer.
        """
        return {
            "state_version": EXACT_RESUME_STATE_VERSION,
            "pq_idx": self.next_pq_idx,
            "rg_idx": self.next_rg_idx,
            "epoch": self.next_epoch,
            "doc_batch_index": self.next_doc_batch_index,
            "dataset_name": self.dataset_name,
        }

    def _apply_resume_state(self, state):
        """Restore internal state from a checkpoint dict."""
        if state.get("state_version") == EXACT_RESUME_STATE_VERSION and "doc_buffer" in state:
            # --- Exact resume (new checkpoint) ---
            self.next_pq_idx = state["pq_idx"]
            self.next_rg_idx = state["rg_idx"]
            self.next_epoch = state["epoch"]
            self.next_doc_batch_index = state.get("doc_batch_index", 0)
            raw_buffer = state["doc_buffer"]
            self.doc_buffer = [list(doc) for doc in raw_buffer]
            self._legacy_resume = False
            print(f"[Exact Resume] rank={self._ddp_rank}: pq_idx={self.next_pq_idx}, "
                  f"rg_idx={self.next_rg_idx}, epoch={self.next_epoch}, "
                  f"doc_batch_index={self.next_doc_batch_index}, "
                  f"doc_buffer_len={len(self.doc_buffer)}")
        elif state.get("state_version") == EXACT_RESUME_STATE_VERSION:
            # --- Cursor-only resume (new lightweight checkpoint without doc_buffer) ---
            self.next_pq_idx = state["pq_idx"]
            self.next_rg_idx = state["rg_idx"]
            self.next_epoch = state["epoch"]
            self.next_doc_batch_index = state.get("doc_batch_index", 0)
            self.doc_buffer = []
            self._legacy_resume = False
            warnings.warn(
                "Dataloader resume state is missing doc_buffer; this is a "
                "cursor-only resume, not an exact resume. Any prefetched, "
                "unconsumed documents from the checkpoint cannot be restored.",
                RuntimeWarning,
                stacklevel=2,
            )
            print(f"[Cursor-only Resume] rank={self._ddp_rank}: pq_idx={self.next_pq_idx}, "
                  f"rg_idx={self.next_rg_idx}, epoch={self.next_epoch}, "
                  f"doc_batch_index={self.next_doc_batch_index}, "
                  f"doc_buffer_len=0")
        else:
            # --- Legacy resume (old checkpoint without state_version) ---
            self.next_pq_idx = state.get("pq_idx", 0)
            self.next_epoch = state.get("epoch", 1)
            self._legacy_resume = True
            self._legacy_rg_idx = state.get("rg_idx", None)
            print(f"[Legacy Resume] rank={self._ddp_rank}: pq_idx={self.next_pq_idx}, "
                  f"rg_idx(legacy)={self._legacy_rg_idx}, epoch={self.next_epoch}")

    # ------------------------------------------------------------------
    # Document iteration (replaces _document_batches generator)
    # ------------------------------------------------------------------

    def _open_row_group(self, pq_idx, rg_idx):
        """Read a single row group and return the text list."""
        filepath = self._parquet_paths[pq_idx]
        pf = pq.ParquetFile(filepath)
        if rg_idx >= pf.num_row_groups:
            return None
        rg = pf.read_row_group(rg_idx)
        return rg.column('text').to_pylist()

    def _doc_batch_iter(self):
        """
        Infinite iterator yielding (text_batch, pq_idx, rg_idx, epoch).

        On exact resume: starts at saved pq_idx/rg_idx and skips
        doc_batch_index sub-batches within the first row group.

        On legacy resume: uses the old base_idx+1 skip logic.
        """
        pq_idx = self.next_pq_idx
        epoch = self.next_epoch
        first_pass = True

        while True:  # multi-epoch loop
            pq_idx = self.next_pq_idx if first_pass else 0
            while pq_idx < len(self._parquet_paths):
                filepath = self._parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)

                # Determine starting rg_idx
                if first_pass and pq_idx == self.next_pq_idx:
                    if hasattr(self, '_legacy_resume') and self._legacy_resume:
                        # Legacy: skip forward by 1 base_idx
                        legacy_rg = self._legacy_rg_idx
                        if legacy_rg is not None:
                            base_idx = legacy_rg // self._ddp_world_size
                            base_idx += 1
                            rg_idx = base_idx * self._ddp_world_size + self._ddp_rank
                            if rg_idx >= pf.num_row_groups:
                                pq_idx += 1
                                continue
                            self._legacy_rg_idx = None
                        else:
                            rg_idx = self._ddp_rank
                    else:
                        # Exact resume: start at saved rg_idx
                        rg_idx = self.next_rg_idx
                else:
                    rg_idx = self._ddp_rank

                skip_doc_batches = self.next_doc_batch_index if (first_pass and pq_idx == self.next_pq_idx and rg_idx == self.next_rg_idx) else 0

                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist()
                    doc_batch_index = 0
                    for i in range(0, len(batch), self.tokenizer_batch_size):
                        if skip_doc_batches > 0:
                            skip_doc_batches -= 1
                            doc_batch_index += 1
                            continue
                        text_sub = batch[i:i + self.tokenizer_batch_size]
                        doc_batch_index += 1
                        # Update internal position BEFORE yielding
                        self.next_pq_idx = pq_idx
                        self.next_rg_idx = rg_idx
                        self.next_epoch = epoch
                        self.next_doc_batch_index = doc_batch_index
                        yield text_sub, pq_idx, rg_idx, epoch
                    rg_idx += self._ddp_world_size
                pq_idx += 1
            first_pass = False
            epoch += 1

    # ------------------------------------------------------------------
    # Main iteration (__iter__)
    # ------------------------------------------------------------------

    def __iter__(self):
        """Yields (inputs, targets, state_dict) indefinitely."""
        assert self.split in ["train", "val"], "split must be 'train' or 'val'"

        B, T = self.B, self.T
        row_capacity = self.row_capacity
        doc_buffer = self.doc_buffer  # alias (mutated in place)
        buffer_size = self.buffer_size

        # Start document iterator
        doc_iter = self._doc_batch_iter()

        def refill_buffer():
            text_batch, _, _, _ = next(doc_iter)
            token_lists = self.tokenizer.encode(
                text_batch, prepend=self.bos_token, num_threads=self.tokenizer_threads
            )
            for tokens in token_lists:
                doc_buffer.append(tokens)

        # Pre-allocate buffers
        use_cuda = self.device == "cuda"
        row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
        cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
        gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=self.device)
        cpu_inputs = cpu_buffer[:B * T].view(B, T)
        cpu_targets = cpu_buffer[B * T:].view(B, T)
        inputs = gpu_buffer[:B * T].view(B, T)
        targets = gpu_buffer[B * T:].view(B, T)

        while True:
            for row_idx in range(B):
                pos = 0
                while pos < row_capacity:
                    while len(doc_buffer) < buffer_size:
                        refill_buffer()

                    remaining = row_capacity - pos

                    # Find largest doc that fits entirely
                    best_idx = -1
                    best_len = 0
                    for i, doc in enumerate(doc_buffer):
                        doc_len = len(doc)
                        if doc_len <= remaining and doc_len > best_len:
                            best_idx = i
                            best_len = doc_len

                    if best_idx >= 0:
                        doc = doc_buffer.pop(best_idx)
                        doc_len = len(doc)
                        row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                        pos += doc_len
                    else:
                        shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                        doc = doc_buffer.pop(shortest_idx)
                        row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                        pos += remaining

            cpu_inputs.copy_(row_buffer[:, :-1])
            cpu_targets.copy_(row_buffer[:, 1:])

            sd = self.lightweight_state_dict()

            gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
            yield inputs, targets, sd


# ---------------------------------------------------------------------------
# Public API (backward-compatible function interfaces)
# ---------------------------------------------------------------------------

def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    dataset_name="fineweb", data_dir=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Delegates to StatefulBestFitDataLoader. Kept for backward compatibility.
    """
    loader = StatefulBestFitDataLoader(
        tokenizer=tokenizer, B=B, T=T, split=split, device=device,
        resume_state_dict=resume_state_dict, buffer_size=buffer_size,
        tokenizer_threads=tokenizer_threads, tokenizer_batch_size=tokenizer_batch_size,
        dataset_name=dataset_name, data_dir=data_dir,
    )
    yield from loader


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets

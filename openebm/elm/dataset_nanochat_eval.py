"""
Enhanced NanoChat evaluation dataset with dual-mode support:
1. PPL mode: Full sequence for perplexity calculation
2. Generation mode: Split sequence (first half as prompt, second half as ground truth)
"""
import os, sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import os

class NanoChatShardEvalDataset(Dataset):
    """
    Dataset for evaluating NanoChat on specific parquet shards.

    Supports two evaluation modes:
    1. PPL mode: Use full sequence for perplexity calculation
    2. Generation mode: Split sequence (first half prompt, second half ground truth)

    Args:
        tokenizer: Tokenizer object
        context_length: Maximum sequence length (e.g., 256)
        shard_indices: List of shard indices to evaluate (e.g., [0, 15] for first and last)
        max_samples_per_shard: Maximum number of samples to take from each shard
        data_dir: Directory containing parquet files
        enable_generation: If True, also prepare data for text generation evaluation
        generation_split_ratio: Ratio to split sequence (default 0.5 = half for prompt, half for GT)
    """

    def __init__(
        self,
        tokenizer,
        context_length=256,
        shard_indices=[0, 15],
        max_samples_per_shard=50,
        data_dir="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/base_data",
        enable_generation=True,
        generation_split_ratio=0.5,
        min_generation_length=64  # Minimum length for generation samples
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.shard_indices = shard_indices
        self.max_samples_per_shard = max_samples_per_shard
        self.data_dir = data_dir
        self.enable_generation = enable_generation
        self.generation_split_ratio = generation_split_ratio
        self.min_generation_length = min_generation_length

        # Get BOS token
        if hasattr(tokenizer, 'get_bos_token_id'):
            self.bos_token = tokenizer.get_bos_token_id()
        elif hasattr(tokenizer, 'bos_token_id'):
            self.bos_token = tokenizer.bos_token_id
        else:
            self.bos_token = 1  # Default

        # Load samples from specified shards
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """Load samples from specified shards."""
        for shard_idx in self.shard_indices:
            shard_file = os.path.join(self.data_dir, f"shard_{shard_idx:05d}.parquet")

            if not os.path.exists(shard_file):
                print(f"Warning: Shard file not found: {shard_file}")
                continue

            print(f"Loading samples from {shard_file}...")
            pf = pq.ParquetFile(shard_file)

            samples_from_shard = 0
            for rg_idx in range(pf.num_row_groups):
                if samples_from_shard >= self.max_samples_per_shard:
                    break

                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()

                for text in texts:
                    if samples_from_shard >= self.max_samples_per_shard:
                        break

                    # Tokenize the document
                    if hasattr(self.tokenizer, 'encode'):
                        # NanoChat tokenizer
                        if hasattr(self.tokenizer, 'enc'):
                            # RustBPETokenizer
                            tokens = self.tokenizer.encode([text], prepend=self.bos_token, num_threads=1)[0]
                        else:
                            # HuggingFace tokenizer
                            tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    else:
                        raise ValueError("Tokenizer must have encode method")

                    # For generation mode, we need longer sequences
                    min_length = self.min_generation_length if self.enable_generation else self.context_length // 2

                    if len(tokens) >= min_length:
                        self.samples.append({
                            'tokens': tokens,
                            'text': text,
                            'shard_idx': shard_idx
                        })
                        samples_from_shard += 1

            print(f"  Loaded {samples_from_shard} samples from shard {shard_idx}")

        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample with both PPL and generation data.

        Returns:
            dict with keys:
                - 'input_ids': Full sequence for PPL calculation [B, S]
                - 'text': Original text (truncated for logging)
                - 'shard_idx': Which shard this came from
                - 'num_tokens': Total number of tokens

                If enable_generation=True, also includes:
                - 'prompt_ids': First half of sequence for generation
                - 'target_ids': Second half of sequence (ground truth)
                - 'prompt_text': Decoded prompt text
                - 'target_text': Decoded target text
        """
        sample = self.samples[idx]
        tokens = sample['tokens']

        # Truncate to context_length if needed
        if len(tokens) > self.context_length:
            tokens = tokens[:self.context_length]

        # Convert to tensor for PPL evaluation
        input_ids = torch.tensor(tokens, dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'text': sample['text'][:500],  # Truncate text for logging
            'shard_idx': sample['shard_idx'],
            'num_tokens': len(tokens)
        }

        # Add generation data if enabled
        if self.enable_generation and len(tokens) >= self.min_generation_length:
            # Split tokens: first half as prompt, second half as target
            split_point = int(len(tokens) * self.generation_split_ratio)

            # Ensure we have at least some tokens in both parts
            split_point = max(split_point, 10)  # At least 10 tokens for prompt
            split_point = min(split_point, len(tokens) - 10)  # At least 10 tokens for target

            prompt_tokens = tokens[:split_point]
            target_tokens = tokens[split_point:]

            result['prompt_ids'] = torch.tensor(prompt_tokens, dtype=torch.long)
            result['target_ids'] = torch.tensor(target_tokens, dtype=torch.long)

            # Decode for human-readable logging
            if hasattr(self.tokenizer, 'decode'):
                try:
                    result['prompt_text'] = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                    result['target_text'] = self.tokenizer.decode(target_tokens, skip_special_tokens=True)
                except:
                    # Fallback if decode fails
                    result['prompt_text'] = sample['text'][:len(sample['text'])//2]
                    result['target_text'] = sample['text'][len(sample['text'])//2:]
            else:
                # Use raw text split as fallback
                text_split = int(len(sample['text']) * self.generation_split_ratio)
                result['prompt_text'] = sample['text'][:text_split]
                result['target_text'] = sample['text'][text_split:]

        return result


def collate_fn_nanochat_eval(batch):
    """
    Collate function for NanoChat evaluation with dual-mode support.

    Returns:
        dict with:
            - 'input_ids': Padded full sequences for PPL [B, 1, S]
            - 'attention_mask': Attention mask [B, S]
            - 'texts': List of original texts
            - 'shard_indices': List of shard indices
            - 'num_tokens': List of token counts

            If generation enabled:
            - 'prompt_ids': Padded prompts [B, S_prompt]
            - 'target_ids': Padded targets [B, S_target]
            - 'prompt_attention_mask': Attention mask for prompts
            - 'prompt_texts': List of prompt texts
            - 'target_texts': List of target texts
    """
    # Find max length in this batch for PPL
    max_len = max(item['input_ids'].shape[0] for item in batch)

    # Pad sequences for PPL evaluation
    input_ids_list = []
    attention_mask_list = []

    for item in batch:
        seq_len = item['input_ids'].shape[0]
        padding_len = max_len - seq_len

        # Pad with 0 (or use a specific pad token if needed)
        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.zeros(padding_len, dtype=torch.long)
        ])

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(padding_len, dtype=torch.long)
        ])

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(attention_mask)

    # Stack into batch
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)

    # Add channel dimension for compatibility: [B, S] -> [B, 1, S]
    input_ids = input_ids.unsqueeze(1)

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': [item['text'] for item in batch],
        'shard_indices': [item['shard_idx'] for item in batch],
        'num_tokens': [item['num_tokens'] for item in batch]
    }

    # Add generation data if available
    if 'prompt_ids' in batch[0]:
        # Pad prompts
        max_prompt_len = max(item['prompt_ids'].shape[0] for item in batch)
        prompt_ids_list = []
        prompt_attention_mask_list = []

        for item in batch:
            prompt_len = item['prompt_ids'].shape[0]
            padding_len = max_prompt_len - prompt_len

            padded_prompt = torch.cat([
                item['prompt_ids'],
                torch.zeros(padding_len, dtype=torch.long)
            ])

            prompt_mask = torch.cat([
                torch.ones(prompt_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ])

            prompt_ids_list.append(padded_prompt)
            prompt_attention_mask_list.append(prompt_mask)

        result['prompt_ids'] = torch.stack(prompt_ids_list)
        result['prompt_attention_mask'] = torch.stack(prompt_attention_mask_list)
        result['prompt_texts'] = [item['prompt_text'] for item in batch]
        result['target_texts'] = [item['target_text'] for item in batch]

        # Also include target_ids for reference (no need to pad, just for logging)
        result['target_ids'] = [item['target_ids'] for item in batch]

    return result


import os
from data.fineweb.dataset import parquets_iter_batched as fineweb_parquets_iter_batched
from data.climbmix.dataset import parquets_iter_batched as climbmix_parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer

# -----------------------------------------------------------------------------
# Dataset selection
def dataset_parquets(dataset_name):
    if dataset_name == "fineweb":
        return fineweb_parquets_iter_batched
    elif dataset_name == "climbmix":
        return climbmix_parquets_iter_batched
    else:
        raise ValueError(f"Unknown dataset name: {args.dataset_name}")


def get_tokenizer(dataset_name):
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer", dataset_name)
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(dataset_name, device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer", dataset_name)
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes

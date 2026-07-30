"""
Adapter to use nanochat's RustBPETokenizer in EBT with HuggingFace-compatible interface.
This implementation loads the tokenizer directly from the serialized pkl file
without requiring the nanochat package.
"""
import os
import sys
import pickle


class RustBPETokenizer:
    """
    Drop-in replacement for nanochat.tokenizer.RustBPETokenizer.
    Loads the tiktoken-based tokenizer directly from the serialized tokenizer.pkl.
    """

    def __init__(self, enc):
        self._enc = enc
        self._special_tokens = enc._special_tokens

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pkl_path = os.path.join(tokenizer_dir, 'tokenizer.pkl')
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"tokenizer.pkl not found in {tokenizer_dir}")
        with open(pkl_path, 'rb') as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self._enc.n_vocab

    def get_bos_token_id(self):
        return self._special_tokens.get('<|bos|>', 0)

    def encode(self, text):
        return list(self._enc.encode_ordinary(text))

    def decode(self, ids):
        return self._enc.decode(ids)

    def encode_special(self, special_token_name):
        return self._special_tokens.get(special_token_name, None)

    def render_for_completion(self, conversation):
        """
        Render a conversation dict into a token ID list ready for completion.
        The last message should be an empty assistant turn (content="") to mark
        the open slot where the model will generate.
        """
        bos = self.get_bos_token_id()
        user_start = self.encode_special('<|user_start|>')
        user_end = self.encode_special('<|user_end|>')
        asst_start = self.encode_special('<|assistant_start|>')
        asst_end = self.encode_special('<|assistant_end|>')

        messages = conversation.get('messages', [])
        tokens = [bos]
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'user':
                tokens.append(user_start)
                tokens.extend(self.encode(content))
                tokens.append(user_end)
            elif role == 'assistant':
                tokens.append(asst_start)
                if content:
                    tokens.extend(self.encode(content))
                    tokens.append(asst_end)
                # empty content = open completion slot; leave after asst_start
        return tokens


class NanoChatTokenizerWrapper:
    """
    Wrapper around RustBPETokenizer to provide HuggingFace-compatible interface.
    """

    def __init__(self, tokenizer_obj=None, tokenizer_dir=None):
        if tokenizer_obj is not None:
            if isinstance(tokenizer_obj, RustBPETokenizer):
                self.tokenizer = tokenizer_obj
            else:
                # Legacy: wrap a raw tiktoken Encoding passed as tokenizer_obj
                self.tokenizer = RustBPETokenizer(tokenizer_obj)
        else:
            if tokenizer_dir is None:
                raise ValueError("Either tokenizer_obj or tokenizer_dir must be provided")
            self.tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

        self.bos_token_id = self.tokenizer.get_bos_token_id()
        self.eos_token_id = self.bos_token_id  # nanochat uses BOS as EOS for compatibility
        self.pad_token_id = self.eos_token_id
        self.unk_token_id = 0

        print(f"[NanoChatTokenizerWrapper] Vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"[NanoChatTokenizerWrapper] EOS/BOS/PAD token ID: {self.eos_token_id}")

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None, **kwargs):
        import torch

        if isinstance(text, str):
            ids_list = [self.tokenizer.encode(text)]
        elif isinstance(text, (list, tuple)):
            ids_list = [self.tokenizer.encode(t) for t in text]
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

        if truncation and max_length is not None:
            ids_list = [ids[:max_length] for ids in ids_list]

        if padding:
            target_length = max_length if max_length is not None else max(len(ids) for ids in ids_list)
            padded_ids, attention_masks = [], []
            for ids in ids_list:
                seq_len = len(ids)
                if seq_len < target_length:
                    padded_ids.append(ids + [self.pad_token_id] * (target_length - seq_len))
                    attention_masks.append([1] * seq_len + [0] * (target_length - seq_len))
                else:
                    padded_ids.append(ids)
                    attention_masks.append([1] * len(ids))
        else:
            padded_ids = ids_list
            attention_masks = [[1] * len(ids) for ids in ids_list]

        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor(padded_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            }
        return {'input_ids': padded_ids, 'attention_mask': attention_masks}

    def encode(self, text, add_special_tokens=False, **kwargs):
        if isinstance(text, str):
            return self.tokenizer.encode(text)
        elif isinstance(text, (list, tuple)):
            return [self.tokenizer.encode(t) for t in text]
        raise ValueError(f"Unsupported text type: {type(text)}")

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        import torch
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if skip_special_tokens:
            special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id}
            for name in ("<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                         "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"):
                tid = self.tokenizer.encode_special(name)
                if tid is not None:
                    special_ids.add(tid)
            token_ids = [t for t in token_ids if t not in special_ids]
        return self.tokenizer.decode(token_ids)

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        import torch
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [self.decode(seq if not isinstance(seq, torch.Tensor) else seq.tolist(),
                            skip_special_tokens=skip_special_tokens)
                for seq in sequences]


def get_nanochat_tokenizer(tokenizer_dir=None):
    return NanoChatTokenizerWrapper(tokenizer_dir=tokenizer_dir)

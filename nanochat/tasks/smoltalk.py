"""
SmolTalk by HuggingFace. Good "general" conversational dataset.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
We use the "smol" version, which is more appropriate for smaller models.
"""

import os
from datasets import load_dataset
from tasks.common import Task

class SmolTalk(Task):
    """ smol-smoltalk dataset. train is 460K rows, test is 24K rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"

        # 支持离线数据集
        cache_dir = None
        download_mode = None
        if os.environ.get("NANOCHAT_SFT_DATA_DIR"):
            cache_dir = os.path.join(os.environ["NANOCHAT_SFT_DATA_DIR"], "smol-smoltalk")
            if os.path.exists(cache_dir):
                download_mode = "reuse_cache_if_exists"
            else:
                cache_dir = None

        self.ds = load_dataset(
            "HuggingFaceTB/smol-smoltalk",
            split=split,
            cache_dir=cache_dir,
            download_mode=download_mode
        ).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # sanity checking asserts here
        # TODO: we could remove these asserts later, for now just don't want any footguns
        # there is an optional system message at the beginning
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:] # optional system message is OK
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "SmolTalk messages must have at least 2 messages"
        for i, message in enumerate(rest_messages):
            # user and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        conversation = {
            "messages": messages,
        }
        return conversation

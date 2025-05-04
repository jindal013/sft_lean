from datasets import load_dataset


def load_sft_dataset(path: str, split: str = "train"):
    """
    Expects a JSONL file with fields: 'prompt', 'response'.
    """
    ds = load_dataset("json", data_files={split: path})[split]
    # Rename keys for TRL SFTTrainer compatibility
    ds = ds.map(lambda x: {"text": x["prompt"] + x["response"]})
    return ds


def load_preference_dataset(path: str, split: str = "train"):
    """
    Expects a JSONL file with fields: 'query', 'chosen', 'rejected'.
    """
    ds = load_dataset("json", data_files={split: path})[split]
    # TRL DPOTrainer expects dicts with 'query', 'chosen', 'rejected'
    return ds


def load_gpro_dataset(path: str, split: str = "train"):
    """
    Expects a JSONL file with fields: 'query' and 'preference_rewards'.
    You can compute 'preference_rewards' externally or use a reward model.
    """
    ds = load_dataset("json", data_files={split: path})[split]
    return ds
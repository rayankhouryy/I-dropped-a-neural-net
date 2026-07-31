"""Data loading for TinyStories and WikiText corpora."""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from typing import Optional, Dict


class TextDataset(Dataset):
    """Tokenized text dataset for language modeling."""

    def __init__(
        self,
        texts: list,
        tokenizer: GPT2Tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for text in texts:
            if not text.strip():
                continue
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if tokens["input_ids"].shape[1] >= 10:  # skip very short
                self.examples.append(tokens["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


def collate_fn(batch, pad_token_id: int = 50256):
    """Collate and pad batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def load_tinystories(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> list:
    """Load TinyStories dataset."""
    ds = load_dataset("roneneldan/TinyStories", split=split)
    texts = ds["text"]
    if max_samples:
        texts = texts[:max_samples]
    return texts


def load_wikitext(
    split: str = "train",
    max_samples: Optional[int] = None,
) -> list:
    """Load WikiText-103 dataset."""
    split_map = {"train": "train", "validation": "validation", "test": "test"}
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split_map[split])
    texts = [t for t in ds["text"] if t.strip()]
    if max_samples:
        texts = texts[:max_samples]
    return texts


def get_tokenizer() -> GPT2Tokenizer:
    """Get GPT-2 tokenizer with padding token."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class CollateFn:
    """Picklable collate function for multiprocessing DataLoader."""

    def __init__(self, pad_token_id: int = 50256):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        return collate_fn(batch, self.pad_token_id)


def create_dataloaders(
    dataset_name: str = "tinystories",
    tokenizer: Optional[GPT2Tokenizer] = None,
    max_length: int = 512,
    batch_size: int = 8,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = 5000,
    num_workers: int = 0,  # Default to 0 for compatibility
) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders."""
    if tokenizer is None:
        tokenizer = get_tokenizer()

    load_fn = load_tinystories if dataset_name == "tinystories" else load_wikitext

    train_texts = load_fn("train", max_train_samples)
    val_texts = load_fn("validation", max_val_samples)

    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)

    collate = CollateFn(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    return {"train": train_loader, "val": val_loader}


def create_domain_shift_loader(
    tokenizer: GPT2Tokenizer,
    max_length: int = 512,
    batch_size: int = 8,
    max_samples: int = 50000,
    num_workers: int = 0,
) -> DataLoader:
    """Create WikiText dataloader for domain-shifted pretraining."""
    texts = load_wikitext("train", max_samples)
    dataset = TextDataset(texts, tokenizer, max_length)
    collate = CollateFn(tokenizer.pad_token_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )

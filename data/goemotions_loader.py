from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from utils.config import Config

tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)

def encode_labels(example):
    multi_hot = torch.zeros(Config.NUM_LABELS)
    for label in example["labels"]:
        multi_hot[label] = 1
    example["multi_labels"] = multi_hot
    return example

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LENGTH
    )

def load_goemotions():
    dataset = load_dataset("go_emotions")

    dataset = dataset.map(encode_labels)
    dataset = dataset.map(tokenize, batched=True)

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "multi_labels"]
    )

    return dataset

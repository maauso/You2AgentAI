import configparser
import os

from datasets import load_from_disk
from transformers import AutoTokenizer


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")

# Load configuration
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

try:
    model_name = config.get("tokenizer", "model_name")
    MAX_LENGTH = int(config.get("tokenizer", "max_length"))
    prepared_dataset_dir = config.get(
        "dataset", "prepared_dataset_dir", fallback="prepared_dataset_chatml")
    tokenized_dataset_dir = config.get(
        "tokenizer", "output_dir", fallback="tokenized_dataset_chatml")
except Exception as e:
    print(f"Error loading configuration: {e}")
    exit(1)

prepared_dataset_path = os.path.join(BASE_DIR, prepared_dataset_dir)
tokenized_dataset_path = os.path.join(BASE_DIR, tokenized_dataset_dir)

# 1. Load the Tokenizer
print(f"🔄 Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure Special Tokens (ChatML)
# We add <|im_start|> and <|im_end|> as special tokens so they are treated as atomic units.
special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
tokenizer.add_special_tokens(special_tokens)
added_special_tokens = tokenizer.special_tokens_map.get(
    "additional_special_tokens",
    special_tokens["additional_special_tokens"],
)

# Define pad_token (Standard practice for Mistral is using eos_token or the new im_end)
tokenizer.pad_token = "<|im_end|>"
tokenizer.padding_side = "right"  # Standard for training causal LMs

# 3. Load Dataset (Prepared in Step 1)
print(f"📦 Loading prepared dataset from {prepared_dataset_path}...")
raw_dataset = load_from_disk(prepared_dataset_path)


def preprocess_function(example):
    """
    Applies label masking to a dataset that was already standardized to ChatML.
    Labels are set to -100 for the user prompt so the model only learns from assistant responses.
    """
    chatml_text = example["text"]

    if "<|im_start|>assistant\n" not in chatml_text:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # Tokenize the full text
    tokenized = tokenizer(
        chatml_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False  # We handle special tokens manually in the text
    )

    input_ids = list(tokenized["input_ids"])
    labels = list(input_ids)

    # --- LABEL MASKING LOGIC ---
    # We want to find where the assistant response starts.
    assistant_start_tag = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False)

    # Find the start index of the assistant response in input_ids
    for i in range(len(input_ids) - len(assistant_start_tag)):
        if input_ids[i:i+len(assistant_start_tag)] == assistant_start_tag:
            # Mask everything before the actual response starts (including the tag)
            for j in range(i + len(assistant_start_tag)):
                labels[j] = -100
            break

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }


print("🛠️ Tokenizing and applying label masking...")
tokenized_dataset = raw_dataset.map(
    preprocess_function,
    remove_columns=raw_dataset.column_names,
    desc="Tokenizing with ChatML Masking"
)

# Save the tokenized dataset
tokenized_dataset.save_to_disk(tokenized_dataset_path)

print(f"\n✅ Tokenization complete!")
print(f"📁 Saved to: {tokenized_dataset_path}")
print(f"📝 Special tokens added: {added_special_tokens}")
print(f"📏 Max length: {MAX_LENGTH}")
print(f"📊 Total examples: {len(tokenized_dataset)}")

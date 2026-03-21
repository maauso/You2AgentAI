from transformers import AutoTokenizer
from datasets import load_dataset
import configparser
import os

# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")

try:
    model_name = config.get("tokenizer", "model_name")
    MAX_LENGTH = int(config.get("tokenizer", "max_length"))
except Exception as e:
    print(f"Error loading configuration: {e}")
    exit(1)

# 1. Load the Tokenizer
print(f"🔄 Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure Special Tokens (ChatML)
# We add <|im_start|> and <|im_end|> as special tokens so they are treated as atomic units.
special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
tokenizer.add_special_tokens(special_tokens)

# Define pad_token (Standard practice for Mistral is using eos_token or the new im_end)
tokenizer.pad_token = "<|im_end|>"
tokenizer.padding_side = "right" # Standard for training causal LMs

# 3. Load Dataset (Prepared in Step 1)
print("📦 Loading and formatting 'timdettmers/openassistant-guanaco'...")
raw_dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

def preprocess_function(example):
    """
    Transforms Guanaco format to ChatML and applies Label Masking.
    Labels are set to -100 for the user prompt so the model only learns from assistant responses.
    """
    raw_text = example['text']
    
    # Standardize to ChatML
    # Guanaco: ### Human: {user_input}### Assistant: {assistant_response}
    parts = raw_text.split("### Assistant:")
    if len(parts) < 2:
        return {"input_ids": [], "attention_mask": [], "labels": []}
        
    user_part = parts[0].replace("### Human:", "<|im_start|>user\n")
    assistant_part = parts[1]
    
    full_text = f"{user_part}<|im_end|>\n<|im_start|>assistant\n{assistant_part}<|im_end|>"
    
    # Tokenize the full text
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False # We handle special tokens manually in the text
    )
    
    input_ids = list(tokenized["input_ids"])
    labels = list(input_ids)
    
    # --- LABEL MASKING LOGIC ---
    # We want to find where the assistant response starts.
    assistant_start_tag = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    
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
output_dir = "tokenized_dataset_chatml"
tokenized_dataset.save_to_disk(output_dir)

print(f"\n✅ Tokenization complete!")
print(f"📁 Saved to: {output_dir}")
print(f"📝 Special tokens added: {tokenizer.additional_special_tokens}")
print(f"📏 Max length: {MAX_LENGTH}")
print(f"📊 Total examples: {len(tokenized_dataset)}")

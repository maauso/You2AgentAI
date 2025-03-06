from transformers import AutoTokenizer
import json
from datasets import load_dataset
from tqdm import tqdm
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")

try:
    model_name = config.get("tokenizer", "model_name")
    MAX_LENGTH = int(config.get("tokenizer", "max_length"))
except Exception as e:
    print(f"Error loading configuration: {e}")
    exit(1)

# Load the Mistral 7B tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset in JSONL format and access the "train" part
dataset = load_dataset("json", data_files="./filtered_telegram_chat.jsonl")["train"]

# Define a padding token to avoid errors
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(example):
    """
    Concatenate the messages within 'messages' in a user/assistant format.
    """
    try:
        messages = example.get("messages", [])

        if not isinstance(messages, list) or len(messages) == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        conversation = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in messages
            if "role" in msg and "content" in msg
        )

        tokens = tokenizer(
            conversation,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"][0].tolist(),
            "attention_mask": tokens["attention_mask"][0].tolist(),
            "labels": tokens["input_ids"][0].tolist(),
        }
    except Exception as e:
        print(f"Error in tokenization: {e}")
        return {"input_ids": [], "attention_mask": [], "labels": []}


# Apply tokenization to the entire dataset without batching
print("🔄 Tokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function, remove_columns=dataset.column_names, desc="Tokenizing"
)

# Filter out empty examples
tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) > 0)

# Verify the content of the tokenized dataset
print("\n📊 Dataset statistics:")
print(f"Number of examples: {len(tokenized_datasets)}")

# Calculate token statistics
total_tokens = sum(len(example["input_ids"]) for example in tokenized_datasets)
avg_tokens = (
    total_tokens / len(tokenized_datasets) if len(tokenized_datasets) > 0 else 0
)

# Save the tokenized dataset
tokenized_datasets.save_to_disk("tokenized_telegram_chat")

print("\n✅ Process completed:")
print(f"💾 Dataset saved in 'tokenized_telegram_chat'")
print(f"🔢 Total tokens: {total_tokens:,}")
print(f"📈 Average tokens per example: {avg_tokens:.2f}")

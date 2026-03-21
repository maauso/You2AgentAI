import torch
import configparser
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")


def resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(BASE_DIR, path_value)


# 1. Load configuration
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
# Using the base model name from tokenizer section
model_name = config['tokenizer']['model_name']

# 2. Implementation of QLoRA (4-bit Quantization)
# Optimized for RTX 4090 to fit 7B model in ~5GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    # Crucial for Ampere/Ada Lovelace architectures
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 3. Load Base Model with Flash Attention 2
print(f"🚀 Loading base model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    # Hardware acceleration for RTX 40 series
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16
)

# Prepare model for k-bit training (handles gradient checkpointing etc.)
model = prepare_model_for_kbit_training(model)

# 4. High-Capacity LoRA Configuration
# Increased 'r' to learn ChatML format from a Base model
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,  # 2x the value of r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # Covering more projections
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Resize embeddings to account for new ChatML tokens added in Step 2
# Note: Tokenizer should have been saved or re-loaded with added tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)
special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

model.print_trainable_parameters()

# 5. Load Tokenized Dataset (Prepared with Masking)
dataset_path = resolve_path(
    config['fine_tuning'].get(
        'tokenized_dataset_dir', './tokenized_dataset_chatml')
)
print(f"📦 Loading tokenized dataset from {dataset_path}...")
dataset = load_from_disk(dataset_path)

# 6. Optimized Training Arguments for Base Models
training_args = TrainingArguments(
    output_dir=config['fine_tuning']['output_dir'],
    per_device_train_batch_size=int(config['fine_tuning']['batch_size']),
    gradient_accumulation_steps=int(
        config['fine_tuning']['gradient_accumulation_steps']),
    learning_rate=2e-4,  # More aggressive for Base models
    lr_scheduler_type="cosine",  # Smooth decay
    weight_decay=0.01,
    logging_steps=10,
    num_train_epochs=int(config['fine_tuning']['num_train_epochs']),
    bf16=True,  # RTX 4090 native support
    fp16=False,
    optim="paged_adamw_8bit",  # VRAM saving optimizer
    report_to="wandb",  # Integration for professional monitoring
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,  # Important when using custom labels/masking
)

# 7. Start Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("🔥 Starting fine-tuning...")
trainer.train()

# 8. Save the Peft Adapter
trainer.model.save_pretrained("mistral-7b-chatml-adapter")
print("✅ Training complete. Adapter saved.")

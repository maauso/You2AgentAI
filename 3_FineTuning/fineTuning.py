import torch
import configparser
import os
from importlib.util import find_spec
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")


def resolve_path(path_value):
    # If the path is already complete, use it as-is.
    if os.path.isabs(path_value):
        return path_value
    # If it is a relative path, attach it to the project root.
    return os.path.join(BASE_DIR, path_value)


# 1. Load configuration
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
# We reuse the same base model name that the tokenizer step used, so both parts
# speak the same "language" and expect the same token IDs.
model_name = config['tokenizer']['model_name']

# 2. Implementation of QLoRA (4-bit Quantization)
# Goal: make a 7B model small enough to fine-tune on a single consumer GPU.
# Bigger explanations for words like QLoRA, NF4, and double quantization are in
# 3_FineTuning/GLOSSARY.md.
bnb_config = BitsAndBytesConfig(
    # Store the model in 4-bit form so it uses much less VRAM.
    load_in_4bit=True,
    # NF4 is a 4-bit format that usually keeps model quality better than simpler 4-bit choices.
    bnb_4bit_quant_type="nf4",
    # Do the math in bf16 because modern NVIDIA cards handle it well and it is more stable.
    bnb_4bit_compute_dtype=torch.bfloat16,
    # Compress the quantization information too, so memory use drops a bit more.
    bnb_4bit_use_double_quant=True,
)

# 3. Load Base Model with attention backend fallback
print(f"🚀 Loading base model: {model_name}...")
flash_attention_available = find_spec("flash_attn") is not None
attention_backend = "flash_attention_2" if flash_attention_available else "sdpa"

if flash_attention_available:
    print("⚡ FlashAttention2 detected. Trying the fastest attention backend.")
else:
    print("ℹ️ FlashAttention2 not installed. Using PyTorch SDPA by default.")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # Use the 4-bit setup above so the model fits in memory.
        quantization_config=bnb_config,
        # Let Transformers decide which GPU parts should live on automatically.
        device_map="auto",
        # Use FlashAttention2 only when it is installed, otherwise use SDPA.
        attn_implementation=attention_backend,
        # Keep the compute dtype aligned with the rest of training.
        dtype=torch.bfloat16,
    )
except ImportError as e:
    if attention_backend != "flash_attention_2" or "FlashAttention2" not in str(e):
        raise
    print("⚠️ FlashAttention2 was found but could not be loaded. Falling back to PyTorch SDPA.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        # SDPA is the safe built-in PyTorch fallback. Usually a bit slower, but reliable.
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )

# Make the quantized model ready for training. This does the housekeeping needed
# so LoRA can learn on top of 4-bit weights without us wiring everything by hand.
model = prepare_model_for_kbit_training(model)

# 4. High-Capacity LoRA Configuration
# We give LoRA enough capacity to learn the ChatML style from a base model,
# without paying the price of full fine-tuning.
peft_config = LoraConfig(
    # Bigger rank means the adapter can learn richer changes, but it also costs more memory.
    r=16,
    # Alpha controls how strongly LoRA changes affect the base model. A common starting rule
    # is to keep it around 2x the rank.
    lora_alpha=32,
    # These are the main transformer layers where small changes have a big effect.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    # A little dropout makes the adapter less likely to memorize exact training examples.
    lora_dropout=0.05,
    # Do not train separate bias values. This keeps the adapter smaller and simpler.
    bias="none",
    # Tell PEFT this is a next-token text generation task.
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# Reload the tokenizer and make sure it knows the special ChatML markers.
# The glossary covers why these special tokens matter.
tokenizer = AutoTokenizer.from_pretrained(model_name)
# These tokens mark where each chat message starts and ends.
special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
tokenizer.add_special_tokens(special_tokens)
# Use the end-of-message token as padding so empty space looks less strange to the model.
tokenizer.pad_token = "<|im_end|>"
# Right padding is the common safe choice for this training setup.
tokenizer.padding_side = "right"
# The model needs a slightly bigger token table now that we added new special tokens.
model.resize_token_embeddings(len(tokenizer))

# Print how many parameters will actually learn. This is a quick sanity check that
# LoRA is active and we are not accidentally training the whole base model.
model.print_trainable_parameters()

# 5. Load Tokenized Dataset (Prepared with Masking)
dataset_path = resolve_path(
    config['fine_tuning'].get(
        'tokenized_dataset_dir', './tokenized_dataset_chatml')
)
# Load the tokenized dataset from disk instead of building it again every run.
print(f"📦 Loading tokenized dataset from {dataset_path}...")
dataset = load_from_disk(dataset_path)

# 6. Optimized Training Arguments for Base Models
report_to = "wandb" if find_spec("wandb") is not None else "none"
if report_to == "none":
    print("ℹ️ wandb not installed. Disabling external experiment reporting.")

training_args = TrainingArguments(
    output_dir=config['fine_tuning']['output_dir'],
    # How many samples the GPU studies at once. We read this from config so you can
    # raise or lower it depending on how much VRAM your machine has.
    per_device_train_batch_size=int(config['fine_tuning']['batch_size']),
    # How many small batches we collect before doing one real weight update. This gives
    # us the effect of a bigger batch without needing all that memory at once.
    gradient_accumulation_steps=int(
        config['fine_tuning']['gradient_accumulation_steps']),
    # How big each learning step is. We use a fairly strong value because LoRA on a
    # base model usually needs to learn the chat format quickly.
    learning_rate=2e-4,
    # This slowly turns the learning rate down as training goes on. "Cosine" means we
    # start stronger and finish more gently, which usually makes training smoother.
    lr_scheduler_type="cosine",
    # A small penalty for weights growing too much. This helps the model avoid memorizing
    # random noise instead of learning the useful pattern.
    weight_decay=0.01,
    # Print training numbers every 10 update steps. That is frequent enough to catch
    # problems early without filling the screen with too much noise.
    logging_steps=10,
    # How many full passes we make over the whole dataset. We keep it in config because
    # the right amount depends on how big and how clean your dataset is.
    num_train_epochs=int(config['fine_tuning']['num_train_epochs']),
    # Use bfloat16 math. On an RTX 4090 this is faster, more memory-friendly, and usually
    # more stable than older half-precision training.
    bf16=True,
    # Leave float16 off because bf16 already gives us the speed and memory benefits here,
    # and it is usually the safer choice on this GPU.
    fp16=False,
    # Use an 8-bit AdamW optimizer with paging. In plain English: it needs less VRAM, so
    # training a big model is much more realistic on one consumer GPU.
    optim="paged_adamw_8bit",
    # Decide where training logs go. If wandb is installed we send them there; otherwise
    # we keep reporting off so the script still runs cleanly.
    report_to=report_to,
    # Save checkpoints every few steps instead of waiting until the very end. This gives
    # us recovery points in case training stops halfway through.
    save_strategy="steps",
    # Save a checkpoint every 100 update steps. That is a practical middle ground:
    # often enough to be safe, not so often that saving becomes annoying.
    save_steps=100,
    # Keep only the 2 newest checkpoints. This stops the disk from filling up with many
    # old saves we probably will not use.
    save_total_limit=2,
    # Keep all dataset columns, even if Trainer thinks some are unused. We do this because
    # custom masking/labels can be dropped by mistake otherwise.
    remove_unused_columns=False,
)

# 7. Build the Trainer
# Trainer is the Hugging Face training engine that runs the loop for us.
trainer = Trainer(
    # The model we prepared with quantization + LoRA.
    model=model,
    # All the training choices from the block above.
    args=training_args,
    # The tokenized examples the model will study.
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(
        # Use the tokenizer so batches are padded in a way the model understands.
        tokenizer=tokenizer,
        # Pass the model because the collator can use model-specific padding behavior.
        model=model,
        # Pad examples in each batch to the same length so the GPU can process them together.
        padding=True,
        # Ignore padded label positions when computing loss, so the model is not punished for
        # the fake tokens we added only to make tensor shapes match.
        label_pad_token_id=-100,
        # Return PyTorch tensors because Trainer expects PyTorch inputs.
        return_tensors="pt",
    ),
)

# Start the actual learning process.
print("🔥 Starting fine-tuning...")
trainer.train()

# 8. Save the Peft Adapter
# Save only the learned LoRA adapter, not a full copy of the base model. This is much
# smaller and is exactly what we need to reuse the fine-tuned behavior later.
trainer.model.save_pretrained("mistral-7b-chatml-adapter")
print("✅ Training complete. Adapter saved.")

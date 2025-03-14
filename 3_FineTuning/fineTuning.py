import torch
import configparser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")
# Load the Mistral model with 4-bit quantization
model_name = config['fine_tuning']['model_name']

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load the model in 4 bits to save memory
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computations
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # Use double quantization for higher efficiency
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Load on GPU if available
)

# Load the tokenizer and define the padding token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as padding

print("Model loaded successfully!")
# Load the tokenized dataset
dataset = load_from_disk(config['fine_tuning']['tokenized_telegram_chat'])

# Configure LoRA (train only some layers of the model)
lora_config = LoraConfig(
    r=4,  # Increase r since we have more VRAM on the RTX 4090
    lora_alpha=16,  # Adjusted proportionally to r (usually 2*r)
    lora_dropout=0.1,  # Maintains stability
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # See how many parameters will be trained

# Configure the training arguments
training_args = TrainingArguments(
    output_dir=config['fine_tuning']['output_dir'],
    eval_strategy="no",  # Disable evaluation to speed up training
    learning_rate=float(config['fine_tuning']['learning_rate']),
    per_device_train_batch_size=int(config['fine_tuning']['batch_size']),
    per_device_eval_batch_size=int(config['fine_tuning']['batch_size']),
    num_train_epochs=int(config['fine_tuning']['num_train_epochs']),
    weight_decay=float(config['fine_tuning']['weight_decay']),
    save_total_limit=int(config['fine_tuning']['save_total_limit']),
    save_steps=int(config['fine_tuning']['save_steps']),
    logging_dir=config['fine_tuning']['logging_dir'],
    fp16=True,  # Keep FP16 to save memory
    bf16=False,  # If errors with FP16, change to True
    logging_steps=int(config['fine_tuning']['logging_steps']),
    log_level="info",
    eval_steps = int(config['fine_tuning']['eval_steps']),
    gradient_accumulation_steps=int(
        config['fine_tuning']['gradient_accumulation_steps']),
)

# Configure the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # Use dataset directly without ["train"]
)

# Start training
trainer.train()

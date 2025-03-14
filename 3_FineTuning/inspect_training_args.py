import torch
from transformers import TrainingArguments
from torch.serialization import add_safe_globals

# Add TrainingArguments to the safe globals list
add_safe_globals([TrainingArguments])

# Load with weights_only=False (this is safe if you trust the source of the file)
args = torch.load(
    "mistral-finetuned/checkpoint-228/training_args.bin", weights_only=False)
print(args)

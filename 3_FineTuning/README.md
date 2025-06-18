# Fine-Tuning

This step trains a language model using the tokenized dataset to replicate the user's conversational style.

## 📌 How It Works

1. Loads the base model and the tokenized dataset.
2. Configures the training parameters.
3. Fine-tunes the model using LoRA (Low-Rank Adaptation) techniques.

## 🚀 Running the Script

To train the model, run:

```bash
python 3_FineTuning/fineTuning.py
```

The trained model will be saved in the directory specified in the configuration file (`config.ini`).

## 📜 Configuration File Structure

The fine-tuning script uses the following parameters from the `[fine_tuning     ]` section in `config.ini`:

| Parameter | Description |
|-----------|-------------|
| `model_name` | Name of the base model for fine-tuning. |
| `output_dir` | Directory where the trained model will be saved. |
| `learning_rate` | Learning rate for training. |
| `batch_size` | Batch size per device. |
| `num_train_epochs` | Number of training epochs. |
| `weight_decay` | Weight decay factor for regularization. |
| `save_total_limit` | Maximum number of checkpoints to save. |
| `save_steps` | Number of steps between each checkpoint save. |
| `logging_dir` | Directory to save training logs. |
| `gradient_accumulation_steps` | Number of gradient accumulation steps. |

## 📊 Output of the Process

After completing the training, the script generates a checkpoint directory containing the following files:

### 📂 Checkpoint Files (`checkpoint-XXX/`)

Each checkpoint represents a saved state of the model at a specific training step `(e.g., checkpoint-228/)`

| File | Description |
|------|-------------|
| `adapter_config.json` | Configuration file for LoRA Adapter, defining modifications applied to the base model. |
| `adapter_model.safetensors` | Trained model weights stored in a safe tensor format (safetensors). |
| `optimizer.pt` | Optimizer state (e.g., AdamW), used for resuming training. |
| `rng_state.pth` | Random Number Generator (RNG) state, ensuring reproducibility. |
| `scaler.pt` | Mixed Precision Scaler state, useful when training with FP16 for memory efficiency. |
| `scheduler.pt` | Learning rate scheduler state, controlling training step adjustments. |
| `trainer_state.json` | Trainer state from Hugging Face, including training progress and metrics. |
| `training_args.bin` | Training arguments file, storing hyperparameter settings (batch size, learning rate, etc.). |

These files allow you to resume training, analyze model performance, or load the fine-tuned model for inference.

## 🔍 Analyzing the Training Process

To understand how the training progressed, check the following:

1️⃣ Training Metrics (trainer_state.json)

```json
{
    "log_history": [
        {
            "loss": 1.234,
            "learning_rate": 5e-5,
            "epoch": 1.0,
            "step": 100
        },
        {
            "loss": 0.987,
            "learning_rate": 4.5e-5,
            "epoch": 2.0,
            "step": 200
        }
    ]
}
```

This file tracks loss reduction and learning rate changes over time.

Helps in diagnosing training stability.

2️⃣ Training Parameters `(training_args.bin)`

To inspect the hyperparameters used during training:

```bash
python 3_FineTuning/inspect_training_args.py --checkpoint_dir checkpoint-XXX/
```

Ensures the correct batch size, learning rate, and training epochs were used.

These files allow you to resume training, analyze model performance, or load the fine-tuned model for inference.

Ensures the correct batch size, learning rate, and training epochs were used.

## ⚠ Why log_history Might Be Empty?

If log_history is missing or empty in trainer_state.json, it could be due to:

- Logging steps (logging_steps) being too high
  - If set higher than global_step, logs may not be recorded.

- Training ended before a logging event occurred 
  - If global_step is lower than logging_steps, no logs will be written.

- Evaluation steps (eval_steps) were not reached 
  - If eval_steps is too high, no evaluations will be logged.

- Dataset is too small 
  - If the dataset has too few samples, training may complete before logging occurs.

- Trainer configuration missing log level 
  - Ensure log_level="info" is set in TrainingArguments.

To fix this, you can:

- Decrease logging_steps and eval_steps in config.ini.

- Ensure `log_level="info"` is set in the trainer configuration.

- Use a larger dataset or increase num_train_epochs.

These files allow you to resume training, analyze model performance, or load the fine-tuned model for inference.

## 🔍 Why This Process?

This process ensures that:

✔ **Personalized Adaptation** – The model learns and replicates the user's conversational style.

✔ **Efficiency** – Utilizes LoRA techniques for efficient and effective training.

✔ **Compatibility** – The trained model is compatible with most fine-tuning frameworks, facilitating its use and deployment.

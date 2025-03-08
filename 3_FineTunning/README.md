# Fine-Tuning

This step trains a language model using the tokenized dataset to replicate the user's conversational style.

## 📌 How It Works

1. Loads the base model and the tokenized dataset.
2. Configures the training parameters.
3. Fine-tunes the model using LoRA (Low-Rank Adaptation) techniques.

## 🚀 Running the Script

To train the model, run:

```bash
python 3_FineTunning/fineTuning.py
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

After completing the training, the script provides:

- The trained model saved in the specified directory.
- Training logs to visualize metrics and progress.

## 🔍 Why This Process?

This process ensures that:

✔ **Personalized Adaptation** – The model learns and replicates the user's conversational style.

✔ **Efficiency** – Utilizes LoRA techniques for efficient and effective training.

✔ **Compatibility** – The trained model is compatible with most fine-tuning frameworks, facilitating its use and deployment.
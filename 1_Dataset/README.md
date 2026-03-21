# Dataset Preparation

This step uses the **OpenAssistant Guanaco** dataset from Hugging Face as the source for a reproducible fine-tuning workflow.

## 📌 Why OpenAssistant Guanaco?

We chose the `timdettmers/openassistant-guanaco` dataset because it is a gold standard for QLoRA fine-tuning tutorials. It offers:
- **Reproducibility**: Anyone can run the script without needing their own private chat logs.
- **High Quality**: Well-balanced technical and conversational data.
- **Ease of Integration**: Perfectly structured for rapid fine-tuning experiments.

## 🛠️ Standardization to ChatML

Since we are training a **Base** model (Mistral-7B-v0.3), we must explicitly define where each speaker begins and ends. We convert the raw dataset format into **ChatML**, which uses `<|im_start|>` and `<|im_end|>` tokens.

## What Is ChatML?

**ChatML** is a plain-text conversation format designed to represent dialog as a sequence of clearly delimited turns. Each message is wrapped with markers that identify:
- **Who is speaking**: `user`, `assistant`, or `system`
- **Where a message starts**: `<|im_start|>`
- **Where a message ends**: `<|im_end|>`

In practice, ChatML turns an unstructured prompt into a format the model can parse more reliably. Instead of seeing a single block of text, the model sees explicit conversational boundaries and roles.

Why this matters in this repository:
- We are not relying on a chat model with a built-in conversation template.
- We want the model to learn the exact structure of a conversation during fine-tuning.
- The same structure must be reused later during inference, otherwise the model receives a different input format than the one it saw during training.

In this pipeline, ChatML acts as the contract between all stages:
- **Step 1** converts Guanaco examples into ChatML
- **Step 2** tokenizes those ChatML sequences
- **Step 3** fine-tunes the model on that exact structure

Example of ChatML format:
```text
<|im_start|>user
What is a distributed system?<|im_end|>
<|im_start|>assistant
It is a collection of autonomous computers that work together...<|im_end|>
```

## 🚀 How It Works

The script `1_Dataset/prepare_dataset.py`:
1. **Loads** the `train` split of `timdettmers/openassistant-guanaco` directly from the Hugging Face Hub.
2. **Standardizes** the format to ChatML using string replacement.
3. **Displays** the first 3 examples for visual inspection, ensuring the prompt template is correctly applied.
4. **Saves** the prepared dataset to disk so step 2 can consume it directly.

## 🏃 Running the Script

To prepare the dataset, execute:

```bash
python 1_Dataset/prepare_dataset.py
```

Upon execution, you should see the first few examples printed in ChatML format, confirming the dataset is ready for training.

By default, the prepared dataset is saved to `prepared_dataset_chatml`. You can change this location in `config.ini` under `[dataset].prepared_dataset_dir`.

# Dataset Preparation

This step migrates from local Telegram data to a professional workflow using the **OpenAssistant Guanaco** dataset from Hugging Face.

## 📌 Why OpenAssistant Guanaco?

We chose the `timdettmers/openassistant-guanaco` dataset because it is a gold standard for QLoRA fine-tuning tutorials. It offers:
- **Reproducibility**: Anyone can run the script without needing their own private chat logs.
- **High Quality**: Well-balanced technical and conversational data.
- **Ease of Integration**: Perfectly structured for rapid fine-tuning experiments.

## 🛠️ Standardization to ChatML

Since we are training a **Base** model (Mistral-7B-v0.3), we must explicitly define where each speaker begins and ends. We convert the raw dataset format into **ChatML**, which uses `<|im_start|>` and `<|im_end|>` tokens.

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

## 🏃 Running the Script

To prepare the dataset, execute:

```bash
python 1_Dataset/prepare_dataset.py
```

Upon execution, you should see the first few examples printed in ChatML format, confirming the dataset is ready for training.

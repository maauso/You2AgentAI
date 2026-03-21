# Tokenization & ChatML Formatting

This module defines the technical core of the fine-tuning process: how the model perceives the conversation structure and how we optimize its learning.

## 🛠️ ChatML Special Tokens

When using a **Base** model like Mistral-7B-v0.3, the tokenizer doesn't natively recognize ChatML delimiters. We explicitly add `<|im_start|>` and `<|im_end|>` as special tokens. This ensures:
- They are treated as **atomic units** (not split into sub-words).
- The model learns the exact boundaries of user and assistant turns.

## 🧠 Label Masking (Expert Logic)

This is a critical optimization for high-quality fine-tuning. Instead of training the model to predict the entire conversation, we apply **Label Masking**:
1. We locate the transition from user to assistant.
2. We set the `labels` for the user's prompt to `-100`.
3. Since PyTorch's loss functions ignore `-100`, the model only calculates loss (and learns) from the **assistant's responses**.

This prevents the model from wasting capacity "learning" how to repeat the user's input and focuses it entirely on generating the correct response.

## ⚙️ Configuration

The script uses parameters from `config.ini`:
- `model_name`: The base model to load (e.g., `mistralai/Mistral-7B-v0.3`).
- `max_length`: The context window limit (e.g., `512` or `1024`).

## 🚀 How It Works

The `2_Tokenizer/tokenizer.py` script:
1. **Initializes** the tokenizer and adds ChatML tokens.
2. **Resizes** model embeddings (handled in the fine-tuning stage but prepared here).
3. **Processes** the Guanaco dataset into ChatML sequences.
4. **Applies** the masking logic to the labels.
5. **Saves** the processed tensors to `tokenized_dataset_chatml`.

## 🏃 Running the Tokenizer

Execute the following command to process your data:

```bash
python 2_Tokenizer/tokenizer.py
```

## 🛑 Stop Tokens (Note for Inference)

By defining `<|im_end|>` here, we ensure that during inference, the model knows exactly when to stop generating. Without this, a Base model would continue hallucinating text indefinitely.

# Tokenization Module

This module transforms the filtered dataset of conversations into a tokenized format suitable for model training.

## 🔍 What is Tokenization?

Tokenization is the process of converting raw text into numerical tokens that machine learning models can understand. For language models, this involves:

1. Breaking text into smaller units (tokens) such as words, subwords, or characters
2. Converting these tokens into numerical IDs using a vocabulary
3. Creating additional metadata like attention masks to help the model process the input correctly

Tokenization is a critical preprocessing step that bridges human language and the numerical representation needed for neural networks.

## 🛠️ How the Tokenizer Works

The `tokenizer.py` script performs the following operations:

1. **Configuration Loading**: Reads parameters from `config.ini`
2. **Tokenizer Initialization**: Loads a pre-trained tokenizer (Mistral 7B by default)
3. **Dataset Loading**: Imports the filtered conversation dataset
4. **Conversation Processing**: For each conversation:
   - Concatenates messages in a user/assistant format
   - Applies padding and truncation to ensure uniform length
   - Generates input IDs, attention masks, and labels
5. **Statistics Generation**: Calculates and displays metrics about the tokenized dataset
6. **Dataset Saving**: Stores the tokenized data for the fine-tuning process

## ⚙️ Configuration Parameters

The tokenizer uses the following parameters from the `[tokenizer]` section in `config.ini`:

| Parameter | Description |
|-----------|-------------|
| `max_length` | Maximum sequence length for tokens (default: 512). Shorter values use less memory but might truncate longer conversations. |
| `model_name` | The pre-trained model whose tokenizer will be used (default: `"mistralai/Mistral-7B-v0.1"`). This should match the model you plan to fine-tune. |

## 🚀 Running the Tokenizer

To tokenize your filtered dataset, run:

```bash
python 2_Tokenizer/tokenizer.py
```

The script will:
1. Process the filtered dataset from the previous step
2. Display statistics about the tokenized data
3. Save the tokenized dataset to `tokenized_telegram_chat` directory

## 📊 Output

After successful execution, the script provides:
- Number of examples processed
- Total token count
- Average tokens per example
- Stored tokenized dataset ready for fine-tuning

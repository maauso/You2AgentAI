# Dataset Preparation

This step processes raw Telegram chat data to generate a structured dataset for training.

## 📌 How It Works

1. Reads exported Telegram JSON chat files from the `telegram_chats/` directory.
2. Filters messages based on:
   - Whether they were sent by the target user.
   - Whether they are responses to a detected question.
   - Avoiding messages that contain only emojis.
3. Saves the processed dataset as a structured JSONL file.

## 🚀 Running the Script

To generate the dataset, run:

```bash
python 1_Dataset/generate_filtered_dataset.py
```

The resulting dataset will be saved as `filtered_dataset.jsonl` in the root directory.
This file will follow a structured format suitable for fine-tuning AI models.

## 📜 Output File Structure

After processing the dataset, the script generates a JSONL file where each entry represents a structured conversation between a user and an assistant.

This format is widely used in fine-tuning frameworks (e.g., Hugging Face's Trainer, OpenAI models) as it defines clear input-response relationships.

Fine-tuning a model with this structure allows it to learn and mimic specific conversational patterns.

As a goal for this project, we want to train a model to speak like us.
This dataset helps capture our style and responses.

Below is an example of the expected output format:

```json
{
    "messages": [
        {
            "role": "user",
            "content": "Hey, can you help me with something?"
        },
        {
            "role": "assistant",
            "content": "Sure! What do you need?"
        }
    ]
}
```

## 🔍 Why This Structure?

This format ensures that:

✔ **Contextual Understanding** – The model learns relationships between user inputs and responses.

✔ **Personalized Conversations** – The assistant's replies reflect the conversational patterns of the target user.

✔ **Compatibility** – The dataset aligns with most fine-tuning frameworks, reducing preprocessing effort.

#!/usr/bin/env python3
"""
generate_filtered_dataset.py

This script processes and filters Telegram chat data to create a dataset suitable for fine-tuning.
It reads configuration parameters from config.ini (section [filtered_dataset]).
"""

import configparser
import json
import glob
import emoji
import unicodedata
import logging
import subprocess
from datetime import datetime
import spacy
from tqdm import tqdm
import os

# Configure logging for tracking
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("🚀 Starting dataset filtering script...")

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

# Retrieve configuration parameters from the 'filtered_dataset' section
user_target = config.get("filtered_dataset", "user_target")
spacy_model = config.get("filtered_dataset", "spacy_model")
output_file = config.get("filtered_dataset", "json_file")
question_words_config = {
    word.strip() for word in config.get("filtered_dataset", "question_words").split(",")
}

# Load the specified spaCy model; if not available, install it.
try:
    nlp = spacy.load(spacy_model)
    logger.info(f"✅ Model '{spacy_model}' loaded successfully.")
except OSError:
    logger.warning(f"🔹 Model '{spacy_model}' not found. Installing...")
    subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
    nlp = spacy.load(spacy_model)

# Check if spaCy is using GPU (via cupy) for acceleration
if spacy.prefer_gpu():
    logger.info("⚡ spaCy is using the GPU for acceleration.")
else:
    logger.warning("🔹 spaCy is NOT using the GPU. Running on CPU.")

# Search for JSON files in the chats directory (as per original script)
json_files = glob.glob("./telegram_chats/*.json")
logger.info(f"📂 Found {len(json_files)} chat files.")


# Function to remove accents from text for normalization
def remove_accents(text):
    """Removes accents from text."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


# Function to check if text contains only emojis (i.e., no words)
def contains_only_emojis(text):
    """Returns True if the text contains only emojis."""
    return all(char in emoji.EMOJI_DATA for char in text.strip())


# Function to determine if a text has a question structure using spaCy NLP
def is_question(text):
    """Determines if the text appears to be a question without requiring a '?' symbol."""
    # Normalize text: lowercase, remove accents, and trim whitespace
    text = remove_accents(text.lower().strip())
    doc = nlp(text)

    # If the sentence starts with a question word, consider it a question
    if len(doc) > 0 and doc[0].text in question_words_config:
        return True

    # If any token in the text is a question word, consider it a question
    if any(token.text in question_words_config for token in doc):
        return True

    # Detect question structures using auxiliary verbs
    for token in doc:
        if token.dep_ in {"aux", "auxpass"} and token.head.pos_ in {"VERB", "AUX"}:
            return True

    return False


# Process each JSON file
with open(output_file, "w", encoding="utf-8") as f_out:
    # Calculate total number of messages across all files for the progress bar
    total_messages = sum(len(json.load(open(f, "r"))["messages"]) for f in json_files)
    pbar = tqdm(total=total_messages, desc="Processing conversations")

    for file in json_files:
        logger.info(f"📖 Processing file: {file}")
        with open(file, "r", encoding="utf-8") as f:
            chat_data = json.load(f)
            messages = chat_data.get("messages", [])
            processed_count = 0

            # Iterate over messages, except the last one
            for i in range(len(messages) - 1):
                pbar.update(1)
                current_msg = messages[i]
                next_msg = messages[i + 1]

                # Filter valid text messages that are not only emojis
                if (
                    current_msg["type"] == "message"
                    and isinstance(current_msg["text"], str)
                    and current_msg["text"].strip()
                    and not contains_only_emojis(current_msg["text"])
                ):
                    if (
                        next_msg["type"] == "message"
                        and isinstance(next_msg["text"], str)
                        and next_msg["text"].strip()
                        and not contains_only_emojis(next_msg["text"])
                    ):
                        # If the current message is not from the target user and appears to be a question...
                        if current_msg.get("from") != user_target and is_question(
                            current_msg["text"]
                        ):
                            # ...and the next message is from the target user, then record the conversation.
                            if next_msg.get("from") == user_target:
                                user_text = current_msg["text"]
                                assistant_text = next_msg["text"]

                                # Save the conversation as an independent JSONL line
                                conversation_data = {
                                    "messages": [
                                        {"role": "user", "content": user_text},
                                        {
                                            "role": "assistant",
                                            "content": assistant_text,
                                        },
                                    ]
                                }
                                f_out.write(
                                    json.dumps(conversation_data, ensure_ascii=False)
                                    + "\n"
                                )
                                processed_count += 1

            logger.info(
                f"✅ File processed: {file} ({processed_count} conversations added)"
            )

logger.info(f"✅ Filtered dataset saved in {output_file}")

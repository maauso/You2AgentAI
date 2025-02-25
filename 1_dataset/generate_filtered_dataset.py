#!/usr/bin/env python3
"""
generate_filtered_dataset.py

This script processes and filters Telegram chat data to create a dataset suitable for fine-tuning.
Reads configuration parameters from config.ini (section [filtered_dataset]).
"""

import configparser
import json
import glob
import unicodedata
import logging
import subprocess
import emoji
import spacy
from tqdm import tqdm


def remove_accents(text):
    """Removes accents from the text."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def contains_only_emojis(text):
    """Returns True if the text contains only emojis."""
    return all(char in emoji.EMOJI_DATA for char in text.strip())


def is_question(text, nlp, question_words):
    """Determines if the text appears to be a question without requiring the '?' sign."""
    # Normalize the text: lowercase, no accents, and no extra spaces
    text_norm = remove_accents(text.lower().strip())
    doc = nlp(text_norm)

    # If the sentence starts with a question word, it is considered a question
    if len(doc) > 0 and doc[0].text in question_words:
        return True

    # If any token is a question word, it is considered a question
    if any(token.text in question_words for token in doc):
        return True

    # Detect question structures using auxiliary verbs
    for token in doc:
        if token.dep_ in {"aux", "auxpass"} and token.head.pos_ in {"VERB", "AUX"}:
            return True

    return False


def count_total_pairs(json_files):
    """Calculates the total number of consecutive message pairs to process in all files."""
    total = 0
    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                messages = data.get("messages", [])
                if len(messages) > 1:
                    total += len(messages) - 1
        except Exception as e:
            logger.error("Error reading %s: %s", file, e)
    return total


# Define logger at the module level
logger = logging.getLogger(__name__)


def main():
    """Main function to process and filter chat data."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("🚀 Starting dataset filtering script...")

    # Load configuration from config.ini
    config = configparser.ConfigParser()
    config.read("config.ini")

    try:
        user_target = config.get("filtered_dataset", "user_target")
        spacy_model = config.get("filtered_dataset", "spacy_model")
        output_file = config.get("filtered_dataset", "json_file")
        question_words = {
            word.strip() for word in config.get("filtered_dataset", "question_words").split(",")
        }
    except Exception as e:
        logger.error("Error loading configuration: %s", e)
        return

    # Load the spaCy model; if it doesn't exist, install it
    try:
        nlp = spacy.load(spacy_model)
        logger.info("✅ Model '%s' loaded successfully.", spacy_model)
    except OSError:
        logger.warning(
            "🔹 Model '%s' not found. Installing...", spacy_model)
        subprocess.run(
            ["python", "-m", "spacy", "download", spacy_model], check=True)
        nlp = spacy.load(spacy_model)

    # Check if spaCy is using GPU for acceleration
    if spacy.prefer_gpu():
        logger.info("⚡ spaCy is using GPU for acceleration.")
    else:
        logger.warning("🔹 spaCy is NOT using GPU. Running on CPU.")

    # Search for JSON files in the telegram_chats directory
    json_files = glob.glob("./telegram_chats/*.json")
    logger.info("📂 Found %d chat files.", len(json_files))

    # Calculate total message pairs for the progress bar
    total_pairs = count_total_pairs(json_files)
    pbar = tqdm(total=total_pairs, desc="Processing conversations")

    with open(output_file, "w", encoding="utf-8") as f_out:
        for file in json_files:
            logger.info("📖 Processing file: %s", file)
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    messages = data.get("messages", [])
            except Exception as e:
                logger.error("Error loading %s: %s", file, e)
                continue

            processed_count = 0
            # Iterate over consecutive message pairs using zip
            for current_msg, next_msg in zip(messages, messages[1:]):
                pbar.update(1)
                # Validate that both messages are of type 'message' and have valid text (not just emojis)
                if (
                    current_msg.get("type") == "message"
                    and isinstance(current_msg.get("text"), str)
                    and current_msg.get("text").strip()
                    and not contains_only_emojis(current_msg.get("text"))
                ):
                    if (
                        next_msg.get("type") == "message"
                        and isinstance(next_msg.get("text"), str)
                        and next_msg.get("text").strip()
                        and not contains_only_emojis(next_msg.get("text"))
                    ):
                        # If the current message is not from the target user and appears to be a question...
                        if (
                            current_msg.get("from") != user_target
                            and is_question(current_msg.get("text"), nlp, question_words)
                        ):
                            # ... and the next message is from the target user, save the conversation.
                            if next_msg.get("from") == user_target:
                                conversation_data = {
                                    "messages": [
                                        {"role": "user",
                                            "content": current_msg.get("text")},
                                        {"role": "assistant",
                                            "content": next_msg.get("text")},
                                    ]
                                }
                                f_out.write(json.dumps(
                                    conversation_data, ensure_ascii=False) + "\n")
                                processed_count += 1
            logger.info(
                "✅ File processed: %s (%d conversations added)", file, processed_count)
    pbar.close()
    logger.info("✅ Filtered dataset saved to %s", output_file)


if __name__ == "__main__":
    main()

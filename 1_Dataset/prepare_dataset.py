import configparser
import os

from datasets import load_dataset


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


def format_chatml(example):
    """
    Converts the original Human/Assistant format to ChatML.
    The 'timdettmers/openassistant-guanaco' dataset usually has a 'text' field
    with '### Human: ... ### Assistant: ...'
    """
    text = example['text']
    # Replace the markers with ChatML tags
    # Guanaco format is: ### Human: {prompt}### Assistant: {response}
    text = text.replace("### Human:", "<|im_start|>user\n")
    text = text.replace(
        "### Assistant:", "<|im_end|>\n<|im_start|>assistant\n")
    text += "<|im_end|>"
    return {"text": text}


def preview_dataset(dataset, num_samples=3):
    """Prints the first few examples of the dataset for visual inspection."""
    print(f"\n--- Previewing {num_samples} samples from the dataset ---\n")
    for i in range(num_samples):
        print(f"--- Example {i+1} ---")
        print(dataset[i]['text'])
        print("-" * 20 + "\n")


def main():
    output_dir = config.get("dataset", "prepared_dataset_dir",
                            fallback="prepared_dataset_chatml")
    output_path = os.path.join(BASE_DIR, output_dir)

    print("🚀 Loading 'timdettmers/openassistant-guanaco' dataset from Hugging Face...")
    # Load the training split
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

    print("🛠️ Transforming dataset to ChatML format...")
    # Apply the formatting function
    dataset = dataset.map(format_chatml)

    # Visual inspection
    preview_dataset(dataset)

    print(f"💾 Saving prepared dataset to {output_path}...")
    dataset.save_to_disk(output_path)

    print("✅ Dataset preparation complete. 100% of examples are now in ChatML format.")
    print(f"📁 Saved to: {output_path}")


if __name__ == "__main__":
    main()

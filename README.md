# You2AgentAI

You2AgentAI is a project designed to provide a hands-on approach to fine-tuning AI models using real-world conversational data. The goal is to explore how a language model can be adapted to mimic a specific user's conversational style by leveraging Telegram chat history.

## 🚀 Project Overview

This project follows a structured pipeline to transform raw chat data into a fine-tuned AI model. The pipeline consists of three main stages:

1. **Dataset Preparation** – Extracts and filters Telegram chat messages to create a structured dataset suitable for training.
2. **Tokenization** – Processes the dataset into a tokenized format for model training.
3. **Fine-Tuning** – Trains a language model using LoRA (Low-Rank Adaptation) techniques for efficient adaptation.

By following these steps, the project provides insights into customizing language models for personalized conversational experiences.

## 📂 Project Structure

The project consists of three main stages, each with its own dedicated documentation:

1. **[Dataset Preparation](1_Dataset/README.md)** → Extracts and filters Telegram chat messages to create a structured dataset.
2. **[Tokenization](2_Tokenizer/README.md)** → Converts the dataset into a format suitable for training. *(Coming soon)*
3. **[Fine-Tuning](3_FineTuning/README.md)** → Trains a custom model to replicate the user's conversational style. *(Coming soon)*
3. **[Testing Agent](4_Testing_agent/README.md)** → Interacts with the fine-tuned model using GPU acceleration and tests its conversational capabilities. *(Coming soon)*
4. **[Config](config.ini)** → Configuration file for the project.
5. **[Environment](environment.yml)** → Conda environment file for setting up the project.
6. **[Telegram Chats](telegram_chats/)** → Directory containing raw Telegram chat data.

## ⚙ Environment Setup

Before running any scripts, you need to set up the Conda environment.

### 1️⃣ Install Conda (if not installed)

If you haven't installed Conda yet, follow the [official Miniconda installation guide](https://docs.conda.io/en/latest/miniconda.html).

### 2️⃣ Create the Conda Environment

To create and activate the environment, run:

```bash
conda env update --file environment.yml  --prune
conda activate You2AgentAI
```

### 3️⃣ Configure Hugging Face Authentication

To download the Mistral model, you need a Hugging Face account and API token:

1. Create an account on [Hugging Face](https://huggingface.co/join) if you don't have one
2. Generate an access token at [Hugging Face Access Tokens](https://huggingface.co/docs/hub/security-tokens)
3. Login using the huggingface-cli:

```bash
huggingface-cli login
```

When prompted, input your token. This authentication is necessary for downloading and using the Mistral model for fine-tuning.

## Steps to run the project

⚠ Before running the scripts, make sure to update `config.ini` to match your dataset and model preferences.

1. **[Dataset Preparation](1_Dataset/README.md)** → Extracts and filters Telegram chat messages to create a structured dataset.
2. **[Tokenization](2_Tokenizer/README.md)** → Converts the dataset into a format suitable for training. *
3. **[Fine-Tuning](3_FineTuning/README.md)** → Trains a custom model to replicate the user's conversational style. 
4. **[Testing Agent](4_Testing_agent/README.md)** → Interacts with the fine-tuned model using GPU acceleration and tests its conversational capabilities.


### 💻 Using VSCode Dev Container

A VSCode dev container configuration is available for development. This allows you to set up a consistent development environment using Docker containers.

To use the dev container, open the project in VSCode and follow the prompts to reopen the project in the container.


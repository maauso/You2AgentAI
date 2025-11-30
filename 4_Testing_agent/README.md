# Model Testing - You2AgentAI

This section explains how to interact with the fine-tuned model using GPU acceleration and test its conversational capabilities.

## 🚀 Running the Chat Agent

After completing the fine-tuning process, you can test the model by running the interactive chatbot.

### 1️⃣ Activate the Conda Environment
```bash
conda activate You2AgentAI
```

### 2️⃣ Run the Chat Agent with GPU
```bash
python 4_Testing/chat_agent.py
```

Once the script is running, you can start chatting with the model. Type a message and press **Enter** to get a response.

### 3️⃣ Exit the Chat
To exit, type:
```bash
exit
```

## ⚙ Prerequisites
- Ensure the fine-tuned model is saved in `./mistral-finetuned/checkpoint-XXX/`.
- The base model `mistralai/Mistral-7B-Instruct-v0.3` should be available.
- The `config.ini` file should be properly set up.
- The Conda environment should be activated before running the script.
- The system should have enough VRAM available to run the model on **GPU**.

## 🔍 Troubleshooting
If you encounter issues, check the following:
- **CUDA Out of Memory (OOM):** Reduce `max_length` in `chat_agent.py`:
  ```python
  outputs = model.generate(**inputs, max_length=100)
  ```
- **Slow performance:** Ensure you are running on GPU (`device_map="auto"`).
- **Incorrect responses:** Verify that the fine-tuning process completed correctly and that `adapter_path` is set to the correct checkpoint.

---

This step allows you to validate how well the model replicates the conversational style based on the fine-tuning process. 🚀


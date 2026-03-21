# Testing & Inference

This final stage is where you see the results of the fine-tuning process. Since we used a **Base** model, inference requires precise manual control to maintain the conversational structure.

## 🛠️ ChatML Inference Template

To ensure the model responds correctly, every input is wrapped in the exact **ChatML** template used during training:
```text
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
```
The script automatically handles this formatting, ensuring the model identifies the beginning of its turn.

## 🧠 Stopping Criteria (Preventing Hallucinations)

Base models do not intuitively know when to stop talking. Without a stopping criterion, the model might start simulating new user turns or generating endless text.
- We implement a `StopOnToken` class that monitors the output.
- Generation terminates immediately when the `<|im_end|>` token is emitted.

## 🎲 Decoding Strategies

We provide a balance between coherence and creativity through sampling parameters:
- **Temperature (0.8)**: Adds enough variability for a "human-like" feel without losing track of the topic.
- **Top-p (0.9)**: Uses nucleus sampling to focus on the most probable tokens, ensuring technical accuracy.

If you want a simple explanation plus internal details of each generation parameter, read the [inference glossary](GLOSSARY.md).

## 🚀 How to Run

1. Ensure the fine-tuning from Step 3 completed and the adapter is saved in `mistral-7b-chatml-adapter`.
2. Run the interactive chat script:
   ```bash
   python 4_Testing_agent/chat_agent.py
   ```

## 📝 Debugging Tip

The script includes a **Debug Mode** that prints the exact prompt sent to the model. This allows you to verify that the ChatML tags are present and correctly formatted, which is the most common point of failure in fine-tuning inference.

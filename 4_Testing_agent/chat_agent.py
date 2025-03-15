from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Base model from Hugging Face (same as in fine-tuning)
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./mistral-finetuned/checkpoint-387"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model with explicit offload directory
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="./offload"
)

# Load fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)


def chat():
    """Interactive chat with the fine-tuned model."""
    print("🤖 You2AgentAI Chat | Type 'exit' to stop")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"AI: {response}")


if __name__ == "__main__":
    chat()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model
model_name = "./mistral-finetuned"  # Path where the fine-tuned model is saved
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def chat():
    """Interactive chat with the fine-tuned model."""
    print("🤖 You2AgentAI Chat | Type 'exit' to stop")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        inputs = tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"AI: {response}")


if __name__ == "__main__":
    chat()

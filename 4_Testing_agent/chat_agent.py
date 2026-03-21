import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from peft import PeftModel
import configparser

# 1. Load Configuration
config = configparser.ConfigParser()
config.read("config.ini")
base_model_name = config['tokenizer']['model_name']
adapter_path = "mistral-7b-chatml-adapter" # Path where the adapter was saved in Step 3

# 2. Configure Stopping Criteria for ChatML
# Base models don't know when to stop unless we tell them to stop at <|im_end|>
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1] == self.stop_token_id

# 3. Load Tokenizer and Model
print(f"🔄 Loading tokenizer and base model: {base_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# Add special tokens if not already there (essential for ChatML)
special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
tokenizer.add_special_tokens(special_tokens)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager",
    torch_dtype=torch.bfloat16
)

# Resize for added special tokens
base_model.resize_token_embeddings(len(tokenizer))

# 4. Load Fine-tuned Adapter
print(f"🪄 Applying LoRA adapter from {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

stop_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])

def generate_response(user_input):
    """Encapsulates input in ChatML template and generates a response."""
    # Strict ChatML Template structure
    prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print(f"\n--- [DEBUG: Prompt Sent to Model] ---\n{prompt}\n------------------------------------\n")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8, # Balanced creativity
            top_p=0.9,       # Nucleus sampling for coherence
            stopping_criteria=stopping_criteria,
            pad_token_id=stop_token_id
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][len(inputs["input_ids"][0]):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response

def chat():
    """Interactive chat loop."""
    print("🤖 You2AgentAI | ChatML Inference Engine")
    print("Type 'exit' to quit. Mode: Fine-tuned Agent\n")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        
        if not user_query.strip():
            continue

        try:
            response = generate_response(user_query)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"❌ Error during generation: {e}")

if __name__ == "__main__":
    chat()

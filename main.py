import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "microsoft/phi-3-mini-4k-instruct"
# HF_TOKEN = "your_hf_token_here"

# Use MPS (Apple Silicon GPU) if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load model & tokenizer (downloads ~16GB on first run)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map=device
)

def chat(prompt):
    messages = [{"role": "user", "content": prompt}]
    
    # Add return_dict=False to get a plain tensor
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=False  # <-- add this
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Model loaded! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        print(f"\nLLaMA: {chat(user_input)}\n")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pdf_vector_embedding import rag_context
from prompt_format import json_format_instruction


model_id = "microsoft/Phi-4-mini-instruct"

# Use MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

def pretrain_model_ouput(model_id:str, device:str, message:str, generation_args:dict):

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device
    )

    # Create model pipeline
    pipe = pipeline( 
        "text-generation", 
        model = model, 
        tokenizer = tokenizer, 
    )

    output = pipe(message, **generation_args)

    return output[0]["generated_text"]

### archive ###

#print sample message
# sample = pipe.tokenizer.apply_chat_template(
#     messages, 
#     tokenize=False, 
#     add_generation_prompt=True
# )
# print(sample)

# def chat(prompt):

#     messages = [{"role": "system", "content": "Talk like a pirate. Always give answers in separate lines for easier reading"},
#             {"role": "user", "content": prompt}]
#     # Add return_dict=False to get a plain tensor
#     inputs = tokenizer.apply_chat_template(
#         messages,
#         return_tensors="pt",
#         add_generation_prompt=True,
#         return_dict=False
#     ).to(device)

#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_new_tokens=512,
#             temperature=0.7,
#             do_sample=True
#         )

#     response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
#     return response
    
# if __name__ == "__main__":
#     print("Model loaded! Type 'exit' to quit.\n")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break
#         print(f"\nPhi3: {chat(user_input)}\n")

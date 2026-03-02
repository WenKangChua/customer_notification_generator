import datetime
import torch
from pdf_vector_embedding import rag_context
from prompt_format import json_format_instruction
from llm_model import pretrain_model_ouput

# The goal of this file is to extracts the fee data from pdf in a json format


file_path = "./input/m_an11539_en-us 2025-04-15.pdf"
rag_query = "Extract all acquirer fees."
rag_context = rag_context(file_path = file_path, rag_query = rag_query)

question = """
Follow the system instruction and use only information from context to extract the fees. 
The information in context are new fees, even though the effective date is set in the past, unless otherwise specified.
"""
message = [
    {"role": "system", "content": json_format_instruction()},
    {"role": "user", "content": f"{rag_context}. \n\n Question:{question}" }
]
model_id = "microsoft/Phi-4-mini-instruct"

# Use MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# define kwarg
generation_args = { 
    "max_new_tokens": 1000, 
    "return_full_text": False, 
    "do_sample" : False
}

# generate output
model_result = pretrain_model_ouput(model_id, device, message, generation_args)

# write results into file
now = datetime.datetime.now()

file_name = f"output_{now}.txt"
file_path = f"./output/{file_name}"


with open(file_path, "w") as file:
    file.write(model_result)


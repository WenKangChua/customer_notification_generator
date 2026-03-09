from vector_store import vector_store
from local_llm import mini_instruct_model
from prompt_templates import fee_names_json_prompt_instructions, notification_article_prompt_template
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from fee_lookup import fee_lookup


base_path = Path(__file__).parent.parent
output_file = base_path / "database" / "new_fees_announced" / "fee_announcement.csv"
query = "Please extract all relevant acquirer fees that was announced, including the country affected."
input_file_path = "./input/m_an11539_en-us 2025-04-15.pdf"
# From my PDF, get relevant context through similiarity search
context = vector_store(file_path=input_file_path, rag_query=query)
# Format instruction prompt with system user tag to ensure json output
prompt = fee_names_json_prompt_instructions(context=context, query=query)
# Pass the prompt and context into phi4 chat model from huggingface
response = mini_instruct_model(prompt=prompt, question=query, context=context)
json_response = json.loads(response.content)["fee_names"]

print("\npreparing csv file\n")
new_fees = pd.DataFrame(json_response)
datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
new_fees.insert(column="extracted_at", loc=0, value=datetime_now)
new_fees.to_csv(output_file, index=False, mode='a', header=False)
print("\nupdated fee_announcement.csv\n")

print("\nbeginning to generate output\n")
updated_fee_table_markdown = fee_lookup()
print(updated_fee_table_markdown)
file_path = "/Users/wenkangchua/Documents/GitHub/customer_notification_generator/extract_fees/input/m_an11539_en-us 2025-04-15.pdf"
question = "Give me details on what had change and the retional behind it."
context = vector_store(file_path=file_path, rag_query=question)

prompt = notification_article_prompt_template(context=context, updated_fee_table_markdown=updated_fee_table_markdown)
response = mini_instruct_model(prompt=prompt)

article_notification = response.content

with open("./results.md", "w", encoding="utf-8") as f:
    f.write(article_notification)

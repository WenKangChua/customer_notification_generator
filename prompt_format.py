from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pdf_vector_embedding import rag_context
from typing import List

# Generates a format instruction to produce json result
def json_format_instruction():

    # Using pydantic BaseModel, define a list of fields and dtype
    class fee_name(BaseModel):
        fee_name:str = Field(description = "The name of the interchange fee or scheme fee")
        old_rate:float | None = Field(description = "The old interchange fee rate or old scheme fee rate. Contains only numbers without %")
        new_rate:float = Field(description = "The new interchange fee rate or new scheme fee rate. Contains only numbers without %")
        effective_date:str = Field(description = "The effective start date or end date of the fee. In YYYY-MM-DD formt.")

    # Enables a list of fee_name
    class fee_name_list(BaseModel):
        fee_names: List[fee_name]

    # Defines json format
    pydantic_parser = PydanticOutputParser(pydantic_object=fee_name_list)
    format_instructions = pydantic_parser.get_format_instructions()

    # Generates the instruction
    format_instruction_prompt = PromptTemplate(
        template = """
        
        {format_instructions}
        
        Generate ONLY raw JSON, do not include markdown code fences, backticks, or any conversational text.
        """,
        partial_variables = {"format_instructions":format_instructions}
    )

    return format_instruction_prompt.format()

if __name__ == "__main__":
    print(json_format_instruction())
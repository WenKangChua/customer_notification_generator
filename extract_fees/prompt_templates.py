from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from typing import Literal


def fee_names_json_prompt_instructions(query:str, context:str):

    # Using pydantic BaseModel, define a list of fields and dtype
    class fee_name(BaseModel):
        fee_name:str = Field(description = "The name of the interchange fee or scheme fee")
        new_rate:float | None = Field(description = "The new interchange fee rate or new scheme fee rate. Contains only numbers without %")
        effective_date:str = Field(description = "The effective start date or end date of the fee. STRICTLY in YYYY-MM-DD format.")
        country:str | None = Field(description = "The name of the country or countries affecte. For example, Malaysia, Singapore, etc.")
        change_type: Literal["new_fee", "updated_fee"] = Field(description = "The type of fee change. The only values allowed are new_fee and updated_fee.")

    # Enables a list of fee_name
    class fee_name_list(BaseModel):
        fee_names: List[fee_name]

    # Defines json format
    pydantic_parser = PydanticOutputParser(pydantic_object=fee_name_list)

    # Generates the instruction
    format_instructions = pydantic_parser.get_format_instructions()
    format_instruction_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", """
        \nYou are a data analyst with good knowledge in the payments industry working at Stripe. Your role is to extract all relevant acquirer fees into a json format to enable further processing by another agent.
        
        \n{format_instructions}

        \nSTRICTLY generate ONLY raw JSON, do not include markdown code fences, backticks, or any conversational text.The context given are new fees, eventhough the effective date is set in the past.
        """
        ),
        ("user", "Context:\n{context}. \n\nQuestion:\n{query}")  
        ]
    ).partial(format_instructions=format_instructions, context=context, query=query)

    return format_instruction_prompt


def notification_article_prompt_template(updated_fee_table_markdown:str, context:str):
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", 
        """
        You are a professional financial communications expert from stripe. Your task is to write a detailed, article-style announcement you have received regarding fee changes.

        LOGICAL RULES FOR THE ARTICLE:
        1. NARRATIVE SOURCE: Use the "Context" extracted from the PDF to explain the business rationale.
        2. ACTION IDENTIFIER: 
        - If `fee_change` is "updated_fee", describe it as a REVISION of an existing fee.
        - If `fee_change` is "new_fee", describe it as the INTRODUCTION of a new fee.
        3. DATA HANDLING:
        - For REVISIONS, include both the Current Rate and New Rate.
        - For INTRODUCTIONS, list the Current Rate as "N/A" or "-" and state that this is a new billing event.
        4. TONE: Maintain a professional, corporate tone suitable for an official customer notification.
        5. Do not include these information
        - Anything about Technical Resource Center or Pricing Guide

        REQUIRED STRUCTURE:
        1. Headline: A clear, professional title.
        2. BODY:
            - Always start the paragraph with effective date, the region or country, and the business purpose. Keep it to one short paragraph.
            - A clean Markdown table with columns | Country | Effective Date | | Fee Name | Current Rate | New Rate |. DO NOT include fee_change column.
            - A brief closing mentioning that if there are any other questions, please contact us.
        """
        ),
        ("user", 
        """
        ###Context:
        {context}
        ###Fee Table:
        {updated_fee_table_markdown}
        """
        )  
        ]
    ).partial(context=context, updated_fee_table_markdown=updated_fee_table_markdown )

    return prompt

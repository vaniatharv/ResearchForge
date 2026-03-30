import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    temperature=0,
)

parser = StrOutputParser()

def run_synthesis(summarized_research: str, critique: str):
    prompt_template = PromptTemplate(
        input_variables=["summarized_research", "critique"],
        template="""
    You are a synthesis assistant. Your task is to synthesize information from multiple sources into a comprehensive document.
    You will receive content from a summarizer and a critique. Combine these inputs to create a final synthesized document.
    
    The synthesized document should:
    - Integrate key points from the summary
    - Address critiques and concerns raised
    - Provide a balanced, comprehensive perspective
    - Organize information coherently
    
    Output your synthesis in JSON format with keys: executive_summary, main_content, addressed_critiques, conclusions, recommendations.
    
    Summarized Research: {summarized_research}
    Critique: {critique}
    """
    )
    
    chain = prompt_template | llm | parser
    result = chain.invoke({"summarized_research": summarized_research, "critique": critique})
    return result
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

def run_critique(summarized_research: str):
    prompt_template = PromptTemplate(
        input_variables=["summarized_research"],
        template="""
    You are a research critic. Analyze the following summarized research and provide constructive criticism.
    Identify strengths, weaknesses, gaps, and areas for improvement.
    Output your critique in JSON format with keys: strengths, weaknesses, gaps, recommendations.

    Summarized Research: {summarized_research}
    """
    )
    
    chain = prompt_template | llm | parser
    result = chain.invoke({"summarized_research": summarized_research})
    return result

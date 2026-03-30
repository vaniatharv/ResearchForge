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

def run_summarizer(research_data: str):
    prompt_template = PromptTemplate(
        input_variables=["research_data"],
        template="""
    You are a research summarizer. Analyze the following research data and create a comprehensive summary.
    
    The summary should:
    - Highlight key findings and insights
    - Identify main themes and patterns
    - Present information in a clear, structured manner
    - Maintain accuracy while being concise
    
    Output your summary in JSON format with keys: key_findings, themes, insights, summary.
    
    Research Data: {research_data}
    """
    )
    
    chain = prompt_template | llm | parser
    result = chain.invoke({"research_data": research_data})
    return result
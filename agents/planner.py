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


def run_planner(query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""
    You are a research planner. Break down the following research query into smaller, atomic tasks that can be executed independently.
    
    Requirements:
    - Each task should be specific and actionable
    - Tasks should be ordered logically
    - Output must be valid JSON format with a "tasks" array
    - Each task should have "id", "description", and "dependencies" fields

    Query: {query}
    """
    )
    
    chain = prompt | llm | parser
    plan = chain.invoke({"query": query})
    return plan
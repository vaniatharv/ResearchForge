import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import json

# Add parent directory to path to import memory module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory import store_in_vectordb, retrieve_from_vectordb

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    temperature=0,
)

parser = StrOutputParser()



########################################################################################################3
def search_internet(query: str, max_results: int = 5) -> str:
    """
    Search the internet using DuckDuckGo.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
    
    Returns:
        Search results as formatted string
    """
    try:
        print(f"\n🔍 Searching internet for: {query}")
        search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
        search_tool = DuckDuckGoSearchResults(api_wrapper=search_wrapper)
        results = search_tool.invoke(query)
        print(f"✓ Found search results")
        return results
    except Exception as e:
        print(f"⚠️ Search error: {e}")
        return f"Search failed: {str(e)}"
########################################################################################################3



def run_research(plan: str, use_internet: bool = True) -> Document:
    """
    Execute research based on plan with real-time internet search and return as Document object.
    
    Args:
        plan: Research plan to execute
        use_internet: Whether to search the internet for real-time information (default: True)
    
    Returns:
        Document object with research results and metadata
    """
    
    ########################################################################################################3
    # Parse plan to extract search queries
    internet_data = ""
    if use_internet:
        try:
            # Try to parse plan as JSON to extract tasks
            plan_json = json.loads(plan)
            if isinstance(plan_json, dict) and "tasks" in plan_json:
                tasks = plan_json["tasks"]
                for task in tasks[:3]:  # Limit to first 3 tasks to avoid too many searches
                    query = task.get("description", "")
                    if query:
                        search_results = search_internet(query)
                        internet_data += f"\n\n=== Search Results for: {query} ===\n{search_results}\n"
        except json.JSONDecodeError:
            # If plan is not JSON, use it directly as search query
            search_results = search_internet(plan[:200])  # Limit query length
            internet_data = f"\n\n=== Internet Search Results ===\n{search_results}\n"
    ########################################################################################################3



    prompt = PromptTemplate(
        input_variables=["plan", "internet_data"],
        template="""
    You are a research assistant with access to real-time internet information. Execute the following research plan.
    Provide detailed information for each item in the plan including key concepts, important findings, current trends, and relevant sources.
    
    Use the internet search results provided to enhance your research with current, factual information.
    Output your research in JSON format with keys: overview, key_concepts, findings, trends, sources.

    Research Plan: {plan}
    
    Internet Search Data:
    {internet_data}
    """
    )
    
    chain = prompt | llm | parser
    result = chain.invoke({"plan": plan, "internet_data": internet_data})
    
    metadata = {
        "plan": plan,
        "type": "research_result",
        "stage": "research",
        "timestamp": datetime.now().isoformat(),
        "agent": "researcher",
        "word_count": len(result.split()),
        "internet_search_used": use_internet,
        "search_queries_count": internet_data.count("=== Search Results for:") if internet_data else 0
    }
    
    document = Document(
        page_content=result,
        metadata=metadata
    )
    
    return document


# def retrieve_similar_research(query: str, k: int = 3):
#     """
#     Retrieve similar research from vector store based on query.
    
#     Args:
#         query: Search query
#         k: Number of results to return
    
#     Returns:
#         List of relevant documents
#     """
#     filter_metadata = {"type": "research_result"}
#     results = retrieve_from_vectordb(query, k=k, filter_metadata=filter_metadata)
#     return results
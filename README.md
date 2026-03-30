**ResearchForge** is a multi-agent AI research assistant powered by LangChain and Meta Llama 3.1. 
It breaks down a research query into structured tasks, searches the web for real-time information, summarizes findings, critiques them, and synthesizes a final report — all while persisting knowledge in a vector database for semantic retrieval.
---
## Features
- **Query Planning** — Decomposes a research topic into ordered, atomic tasks
- **Internet Search** — Retrieves real-time web results via DuckDuckGo
- **Research Execution** — Combines search results with LLM reasoning to produce detailed findings
- **Summarization** — Distills research into key findings, themes, and insights
- **Critique** — Evaluates summaries for strengths, weaknesses, gaps, and recommendations
- **Synthesis** — Integrates the summary and critique into a coherent final report
- **Vector Memory** — Stores and retrieves research documents using ChromaDB and sentence-transformer embeddings
---
## Architecture
```
User Query
    │
    ▼
┌─────────┐     ┌────────────┐     ┌────────────┐
│ Planner │────▶│ Researcher │────▶│ Summarizer │
└─────────┘     └────────────┘     └────────────┘
                      │                   │
               DuckDuckGo Search    ChromaDB Store
                                          │
                                    ┌─────▼──────┐
                                    │  Critique  │
                                    └─────┬──────┘
                                          │
                                   ┌──────▼──────┐
                                   │ Synthesizer │
                                   └──────┬──────┘
                                          │
                                   Final Report
```
### Agents
| Agent | File | Responsibility |
|---|---|---|
| **Planner** | `agents/planner.py` | Breaks the query into a JSON task list with dependencies |
| **Researcher** | `agents/researcher.py` | Executes the plan, searches the web, and returns structured findings |
| **Summarizer** | `agents/summarizer.py` | Produces a structured JSON summary of the research |
| **Critique** | `agents/critique.py` | Evaluates the summary and outputs strengths, weaknesses, gaps, and recommendations |
| **Synthesizer** | `agents/synthesizer.py` | Merges the summary and critique into a final comprehensive report |
### Supporting Modules
| Module | File | Responsibility |
|---|---|---|
| **Memory** | `memory.py` | Stores and retrieves research documents in ChromaDB using HuggingFace embeddings |
| **State** | `state.py` | Defines the `ResearchState` TypedDict passed through the agent pipeline |
---
## Tech Stack
- **LLM** — [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) via HuggingFace Inference API
- **Orchestration** — [LangChain](https://www.langchain.com/)
- **Web Search** — [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)
- **Vector Store** — [ChromaDB](https://www.trychroma.com/)
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
---
## Prerequisites
- Python 3.9+
- A [HuggingFace](https://huggingface.co/) account with access to **meta-llama/Llama-3.1-8B-Instruct**
- A HuggingFace API token
---
## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/vaniatharv/ResearchForge.git
   cd ResearchForge
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   HUGGINGFACE_TOKEN=hf_your_token_here
   ```
---
## Usage
Run the main application:
```bash
python app.py
```
Each agent in the pipeline can also be used independently:
```python
from agents.planner import run_planner
from agents.researcher import run_research
from agents.summarizer import run_summarizer
from agents.critique import run_critique
from agents.synthesizer import run_synthesis
# 1. Plan
plan = run_planner("What are the latest advances in quantum computing?")
# 2. Research (with internet search)
research_doc = run_research(plan, use_internet=True)
# 3. Summarize
summary = run_summarizer(research_doc.page_content)
# 4. Critique
critique = run_critique(summary)
# 5. Synthesize final report
report = run_synthesis(summary, critique)
print(report)
```
### Vector Memory
```python
from memory import store_in_vectordb, retrieve_from_vectordb
# Store a document
store_in_vectordb("Your research content here", metadata={"topic": "AI"})
# Retrieve similar documents
docs = retrieve_from_vectordb("quantum entanglement", k=3)
```
---
## Project Structure
```
ResearchForge/
├── agents/
│   ├── planner.py       # Query decomposition agent
│   ├── researcher.py    # Web search + research agent
│   ├── summarizer.py    # Summarization agent
│   ├── critique.py      # Critique agent
│   └── synthesizer.py   # Final synthesis agent
├── app.py               # Application entry point
├── graph.py             # Agent workflow / graph definition
├── memory.py            # ChromaDB vector store utilities
├── state.py             # ResearchState TypedDict
├── tools.py             # Shared tool definitions
├── ui.py                # User interface
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables (not committed)
```
---
## Output Format
Each agent outputs structured JSON. The final synthesized report includes:
- `executive_summary` — High-level overview of the research
- `main_content` — Detailed findings and analysis
- `addressed_critiques` — How weaknesses and gaps were resolved
- `conclusions` — Key takeaways
- `recommendations` — Suggested next steps
---
## License
This project is open source. See [LICENSE](LICENSE) for details.

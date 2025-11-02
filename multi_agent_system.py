# ------------------------------------------------------------
# 1️⃣ Imports
# ------------------------------------------------------------
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# ------------------------------------------------------------
# 2️⃣ Load environment variables (API key)
# ------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------
# 3️⃣ Create base LLM
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------------------------------------------------
# 4️⃣ Finder Agent (Search + Collect)
# ------------------------------------------------------------
# Load tools for web search and Wikipedia
tools = load_tools(["ddg-search", "wikipedia"])

finder_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ------------------------------------------------------------
# 5️⃣ Summarizer Agent
# ------------------------------------------------------------
# Custom summarizer template
summarizer_prompt = PromptTemplate.from_template("""
You are a professional summarizer.
Summarize the following text in under 150 words, keeping only key facts and clarity:
{text}
""")

def summarize_text(text):
    prompt = summarizer_prompt.format(text=text)
    response = llm.invoke(prompt)
    return response.content

# ------------------------------------------------------------
# 6️⃣ Multi-Agent Workflow
# ------------------------------------------------------------
def multi_agent_research(topic):
    print(f"🔍 Finder Agent: Searching for information on '{topic}'...")
    info = finder_agent.run(topic)

    print("\n🧾 Summarizer Agent: Summarizing the findings...")
    summary = summarize_text(info)

    return {
        "topic": topic,
        "raw_info": info,
        "summary": summary
    }

# ------------------------------------------------------------
# 7️⃣ Example run
# ------------------------------------------------------------
if __name__ == "__main__":
    result = multi_agent_research("How did the Chandrayaan-3 mission succeed?")
    print("\n✅ Final Summary:\n", result["summary"])

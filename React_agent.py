from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.agent_toolkits.load_tools import load_tools


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv() 

# 1️⃣ Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2️⃣ Load the web search tool (DuckDuckGo)
tools = load_tools(["ddg-search","wikipedia"])

# 3️⃣ Create the ReAct Agent
agent = initialize_agent(
    tools, 
    llm, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4️⃣ Ask question that needs reasoning + web search
query = "Explain how the Indian space program started and who founded ISRO."
result = agent.run(query)

print("\n🔍 Final Answer:", result)

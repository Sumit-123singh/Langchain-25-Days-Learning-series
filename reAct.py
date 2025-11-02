from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_experimental.utilities.python import PythonREPL
  # ✅ updated import
from dotenv import load_dotenv
import os

# 1️⃣ Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2️⃣ Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)

# 3️⃣ Define a Calculator Tool using Python REPL
python_repl = PythonREPL()

def calculator_tool(query: str) -> str:
    """Use Python REPL to evaluate math expressions."""
    return python_repl.run(query)

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for solving math problems."
    )
]

# 4️⃣ Create ReAct Agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True
)

# 5️⃣ Run queries
print(agent.run("What is 25 * 17?"))
print(agent.run("What is (15 + 3) * (7 - 2)?"))


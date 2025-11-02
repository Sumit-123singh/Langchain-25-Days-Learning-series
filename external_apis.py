from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import Tool, initialize_agent, AgentType


# Initialize text-generation pipeline
hf_pipeline = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",  # This will automatically download the model
    max_new_tokens=256
)

# Test the model
output = hf_pipeline("Write a haiku about AI:")[0]['generated_text']
print(output)

# Wrap pipeline in LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Example tool (NewsAPI) - replace with your function
news_tool = Tool(
    name="NewsAPI",
    func=lambda topic: f"Top news about {topic}",
    description="Fetches latest news headlines"
)

# Initialize agent
agent = initialize_agent([news_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Run a query
query = "Get top news about Artificial Intelligence"
result = agent.invoke({"input": query})
print(result)

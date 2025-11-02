from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
# Step 1: Initialize Gemini LLM
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define prompt
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a catchy blog title for the topic: '{topic}'."
)

# New way: use RunnableSequence (prompt | llm)
chain = title_prompt | llm

# Call the chain
response = chain.invoke({"topic": "LangChain"})

print("Generated Blog Title:", response.content)

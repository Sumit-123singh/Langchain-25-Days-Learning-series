# basic_demo.py

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Step 1: Define a simple prompt
prompt = PromptTemplate.from_template("Translate this sentence into French: {text}")

# Step 2: Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Turbo model
    temperature=0
)

# Step 3: Create a simple chain
chain = LLMChain(prompt=prompt, llm=llm, verbose=True)  # verbose=True shows console logs

# Step 4: Run the chain
result = chain.run(text="Good morning, do you know about langchain?")
print("\nFinal Output:", result)

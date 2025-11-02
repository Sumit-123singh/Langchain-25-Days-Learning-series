from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os

from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Step 1: Initialize Gemini LLM
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

#system and human templates
system_msg=SystemMessagePromptTemplate
system_msg=SystemMessagePromptTemplate.from_template("You are a helpful assistant that translates {input_language} to {output_language}.")
human_msg=HumanMessagePromptTemplate.from_template("{text}")

chat_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
final_prompt=chat_prompt.format_messages(topic='langchain',input_language='English',output_language="english", text="Hello, please explain the langchain?")

response = llm.invoke(final_prompt[1].content)
print(response.content)


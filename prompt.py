from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

template = """Explain the topic "{topic}" in exactly 3 sentences, using simple language."""
prompt = PromptTemplate(input_variables=["topic"], template=template)

# Fill the template
final_prompt = prompt.format(topic="LangChain")

# Call the LLM
response = llm.invoke(final_prompt)
print(response.content)
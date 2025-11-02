# 1️⃣ Import necessary classes
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 2️⃣ Load the PDF file using PyPDFLoader
# Provide the full path to your PDF
loader = PyPDFLoader(r"C:\Users\hp\OneDrive\Desktop\Langchain Tutorial\LangChain_20_Day_Roadmap.pdf")

# 3️⃣ Load pages as Document objects
# Each page in PDF becomes one Document object with page_content
documents = loader.load()

# 4️⃣ Check the first page content (first 500 characters)
print(documents[0].page_content[:500])  # Helps verify that PDF is loaded correctly

# 5️⃣ Extract text from all Document objects
# Because split_text() expects a string, not a list
document_text = ""
for doc in documents:
    document_text += doc.page_content + "\n"  # Add newline between pages

# 6️⃣ Initialize the text splitter
# chunk_size = number of characters in each chunk
# chunk_overlap = number of characters that overlap between chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 7️⃣ Split the concatenated text into smaller chunks
chunks = text_splitter.split_text(document_text)  # ✅ Pass a string

# 8️⃣ Check the first 2 chunks
print(chunks[:2])

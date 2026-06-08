import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

os.environ["GOOGLE_API_KEY"] = "AIzaSyCcdDpVxSsVLDD8rUk_u2swQg-k6xhzXYQ"
# 1. Prepare your "Private" data
# 1. Load your data (You can also upload a .txt file to Colab and read it here)
context_data = """
CyberCare is an AI-powered system designed for phishing and fraud detection.
It utilizes XGBoost for classification and is integrated with a FastAPI backend.
The training dataset includes over 10,000 verified phishing URLs and legitimate sites.
The system was developed as part of a graduate research project at UAEU.
"""

# 2. Advanced Chunking (Recursive handles punctuation better)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.create_documents([context_data])

# 3. Initialize Gemini Embeddings & Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma.from_documents(docs, embeddings)

# 4. Setup the LLM (Gemini 1.5 Flash is perfect for Colab)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 5. Create the QA Chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

# 6. Test it
query = "What dataset was used for CyberCare?"
print(rag_chain.invoke(query)["result"])
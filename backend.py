import os
import pinecone
import openai
import fitz  # PyMuPDF for PDF processing
import time
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env
load_dotenv()

# Retrieve keys and configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client (using the correct import for pinecone-client==2.2.0)
pc = pinecone.Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# Check if the index exists; if not, create it.
index_list = [index.name for index in pc.list_indexes()]

if pinecone_index_name not in index_list:
    pc.create_index(name=pinecone_index_name, dimension=1536, metric="cosine") # Added metric

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of the PDF and returns it as a string."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    print("Extracted text preview:", text[:200])
    return text

def process_batch(batch, embeddings):
    """Processes a batch of text chunks with exponential backoff on rate limit errors."""
    success = False
    wait_time = 10  # initial wait time in seconds
    while not success:
        try:
            LangchainPinecone.from_texts(batch, embeddings, index_name=pinecone_index_name)
            success = True
        except Exception as e:
            if "429" in str(e) or "insufficient_quota" in str(e):
                print(f"Rate limit error encountered: {e}")
                print(f"Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                wait_time *= 2  # exponential backoff
            else:
                raise

def embed_and_store_text(pdf_path):
    """
    Extracts text from a PDF, splits it into chunks, and stores embeddings in Pinecone.
    The `pdf_path` is provided by the Flask endpoint from the uploaded file.
    """
    text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    print(f"Number of text chunks: {len(texts)}")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, max_retries=5)

    batch_size = 10  # Adjust batch size based on your rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        process_batch(batch, embeddings)
        time.sleep(2)  # Delay between batches

def initialize_rag():
    """
    Initializes the retrieval chain using RetrievalQA.from_chain_type with the "stuff" chain.
    A custom Arabic prompt template is provided.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = LangchainPinecone.from_existing_index(
        index_name=pinecone_index_name,
        embedding=embeddings
    )
    retriever = vector_store.as_retriever()

    # Define your Arabic prompt template.
    system_prompt = """
    أجب على السؤال بناءً على السياق التالي باللغة العربية:
    {context}

    السؤال: {input}
    الإجابة:
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
 
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

    # Create a RetrievalQA chain using the "stuff" chain type.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
 
    # Create retrieval chain
    chain = create_retrieval_chain(retriever, question_answer_chain)
 
    return chain

def ask_chatgpt_arabic(query):
    """Queries the retrieval chain with an Arabic query and returns the raw response."""
    chain = initialize_rag()
    # Pass the query as a string; RetrievalQA expects a string input.
    response = chain.invoke({'input': query})
    return response['answer'] # Accessing the result correctly.

# Optional: Testing from the command line.
if __name__ == "__main__":
    ask_chatgpt_arabic("Hello")
    print("File processed successfully.")
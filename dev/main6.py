from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import boto3
import os
import time
import concurrent.futures
import logging
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import Pinecone as PineconeLang
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone, PodSpec
from tqdm.autonotebook import tqdm

# Constants
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = "gcp-starter"
INDEX_NAME = "gsasubset"
MODEL_ID = 'anthropic.claude-v2:1'
DOCUMENT_PATH = "/home/ubuntu/gsa-rag-backend/dev/GSA-buyers-guide/"

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to the appropriate origins if needed
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class Question(BaseModel):
    user_question: str

# Initialize Bedrock client globally
def initialize_bedrock_client():
    role_arn = 'arn:aws:iam::992382738258:role/gsa-rag'
    session_name = 'AssumedRoleSession'
    region_name = 'us-east-1'
    sts_client = boto3.client('sts', region_name=region_name)

    response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)

    credentials = response['Credentials']
    logger.info("Initializing Bedrock client...")
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
        region_name=region_name
    ).client(service_name="bedrock-runtime", region_name="us-east-1")

def initialize_pinecone():
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["PINECONE_API_ENV"] = PINECONE_API_ENV
    logger.info("Pinecone credentials initialized")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Pinecone initialized ✅")
    return pc

def load_documents(document_path):
    logger.info(f"Loading documents from: {document_path}...")
    loader = PyPDFDirectoryLoader(document_path)
    documents = loader.load()
    if documents:
        logger.info("Documents loading complete ✅")
        logger.info(f"Number of documents loaded: {len(documents)}")
    else:
        logger.warning("No documents loaded!")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

def initialize_qa_chain():
    bedrock_runtime = initialize_bedrock_client()
    llm = Bedrock(
        model_id=MODEL_ID,
        client=bedrock_runtime,
        model_kwargs={
            'max_tokens_to_sample': 4096,
            'temperature': 1.0,
            'top_k': 250,
            'top_p': 0.999
        }
    )
    bedrock_embeddings = BedrockEmbeddings(client=bedrock_runtime)
    doc_texts = [t.page_content for t in load_documents(DOCUMENT_PATH)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        logger.info("Initializing Pinecone for document search...")
        docsearch_future = executor.submit(PineconeLang.from_texts, doc_texts, bedrock_embeddings, index_name=INDEX_NAME)
        chain_future = executor.submit(load_qa_chain, llm, chain_type="stuff")
    docsearch = docsearch_future.result()
    chain = chain_future.result()
    return docsearch, chain

docsearch, chain = initialize_qa_chain()

@app.get("/hello")
async def read_root():
    return {"message": "Hello, from GSA-RL-RAG V2"}

@app.post("/generate_answer/")
async def generate_answer(question: Question):
    try:
        user_question = question.user_question
        query = f"You are an AI assistant. {user_question}. Use provided context only."
        logger.info("Query: %s", query)
        logger.info("Searching for similar documents")
        
        # Search for similar documents
        docs = docsearch.similarity_search(query, k=80)
        
        # Run QA chain
        output = chain.run(input_documents=docs, question=query)
        logger.info("Output: %s", output)
        return output
    except Exception as e:
        logger.exception("An error occurred while generating answer.")
        return {"error": "An error occurred while generating answer. Please try again later."}

# Run the FastAPI app using uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("main6:app", host="0.0.0.0", port=8000)

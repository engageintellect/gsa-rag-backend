from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import Pinecone as PineconeLang
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone, PodSpec
from tqdm.autonotebook import tqdm
import boto3
import os
import time
import concurrent.futures
import logging
import sys

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class Question(BaseModel):
    user_question: str

def initialize_bedrock_client():
    role_arn = 'arn:aws:iam::992382738258:role/gsa-rag'
    session_name = 'AssumedRoleSession'
    region_name = 'us-east-1'
    sts_client = boto3.client('sts', region_name=region_name)

    response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)

    credentials = response['Credentials']
    logging.info("Initializing Bedrock client...")
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
        region_name=region_name
    ).client(service_name="bedrock-runtime", region_name="us-east-1")

def load_documents(document_path):
    logging.info(f"Loading documents from: {document_path}...")
    loader = PyPDFDirectoryLoader(document_path)
    documents = loader.load()
    if documents:
        logging.info("Documents loading complete ✅")
        logging.info(f"Number of documents loaded: {len(documents)}")
    else:
        logging.info("No documents loaded!")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

# Pinecone credentials
os.environ["PINECONE_API_ENV"] = "gcp-starter"
index_name = "gsasubset"
logging.info("Pinecone credentials initialized")

# Initialize Pinecone
try:
    logging.info("Initializing Pinecone...")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    logging.info("Pinecone initialized ✅")
except Exception as e:
    logging.error(f"Error initializing Pinecone: {e}")
    sys.exit(1)

# Initialize Bedrock client
try:
    bedrock_runtime = initialize_bedrock_client()
    logging.info("Bedrock client initialized ✅")
except Exception as e:
    logging.error(f"Error initializing Bedrock client: {e}")
    sys.exit(1)

# Retry mechanism for initializing services
retry_attempts = 3
for attempt in range(retry_attempts):
    try:
        # Initialize Pinecone index
        if index_name not in pc.list_indexes().names():
            logging.info("Creating Pinecone index...")
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='dotproduct',
                spec=PodSpec(environment='gcp-starter')
            )
        
        # Wait for index to finish initialization
        while not pc.describe_index(index_name).status["ready"]:
            logging.info("Waiting for Pinecone index to finish initialization")
            time.sleep(1)
        
        break  # Exit retry loop if successful
    except Exception as e:
        logging.warning(f"Error initializing Pinecone index (attempt {attempt+1}/{retry_attempts}): {e}")
        if attempt < retry_attempts - 1:
            time.sleep(5)  # Wait before retrying
        else:
            logging.error("Failed to initialize Pinecone index after multiple attempts")
            sys.exit(1)

# Additional logging for clarity
logging.info("Initialization complete ✅")

# Load documents
document_path = "/home/ubuntu/gsa-rag-backend/dev/GSA-buyers-guide/"
docs = load_documents(document_path)

# Initialize Bedrock and embeddings
llm = Bedrock(
    model_id='anthropic.claude-v2:1',
    client=bedrock_runtime,
    model_kwargs={
        'max_tokens_to_sample': 4096,
        'temperature': 1.0,
        'top_k': 250,
        'top_p': 0.999
    }
)
bedrock_embeddings = BedrockEmbeddings(client=bedrock_runtime)

# Initialize Pinecone for document search
doc_texts = [t.page_content for t in docs]
with concurrent.futures.ThreadPoolExecutor() as executor:
    logging.info("Initializing Pinecone for document search...")
    docsearch_future = executor.submit(PineconeLang.from_texts, doc_texts, bedrock_embeddings, index_name=index_name)
    chain_future = executor.submit(load_qa_chain, llm, chain_type="stuff")

docsearch = docsearch_future.result()
chain = chain_future.result()

# Example query
user_query = "How can GSA help me in selecting the right MFD? In particular, what does GSA recommend for picking the right maintainance plan?"

@app.get("/hello")
async def read_root():
      return {"message": "Hello, from GSA-RL-RAG V2"}

@app.post("/generate_answer/")
async def generate_answer(question: Question):
    try:
        user_question = question.user_question
        query = f"You are an AI assistant. {user_question}. Use provided context only."
        logging.info("Query:", query)
        logging.info("Searching for similar documents...")

        # Search for similar documents
        docs = await docsearch.similarity_search(query, k=80)
        
        # Run QA chain
        output = await chain.run(input_documents=docs, question=query)
        logging.info("Output:", output)
        return output
    except Exception as e:
        logging.error(f"An error occurred while generating answer: {e}")
        return {"error": "An error occurred while generating answer. API may be down. Please try again later."}

# Run the FastAPI app using uvicorn when the script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main5:app", host="0.0.0.0", port=8000)


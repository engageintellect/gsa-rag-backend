from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import boto3
import os
import concurrent.futures
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import Pinecone as PineconeLang
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone, PodSpec

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

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] = "7f2bbe68-ec0e-4e28-9575-b5da2c4ffdc3"
os.environ["PINECONE_API_ENV"] = "gcp-starter"
index_name = "gsasubset"
print("Pinecone credentials initialized")

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
print("Pinecone initialized ✅")

# Initialize Bedrock client
def initialize_bedrock_client():
    role_arn = 'arn:aws:iam::992382738258:role/gsa-rag'
    session_name = 'AssumedRoleSession'
    region_name = 'us-east-1'
    sts_client = boto3.client('sts', region_name=region_name)

    response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)

    credentials = response['Credentials']
    print("Initializing Bedrock client...")
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
        region_name=region_name
    ).client(service_name="bedrock-runtime", region_name="us-east-1")

bedrock_runtime = initialize_bedrock_client()
print("Bedrock client initialized ✅")

# Initialize Pinecone index
if index_name not in pc.list_indexes().names():
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='dotproduct',
        spec=PodSpec(environment='gcp-starter')
    )

# Wait for index to finish initialization
while not pc.describe_index(index_name).status["ready"]:
    print("Waiting for Pinecone index to finish initialization")
    time.sleep(1)

# Load documents
def load_documents(document_path):
    print(f"Loading documents from: {document_path}...")
    loader = PyPDFDirectoryLoader(document_path)
    documents = loader.load()
    if documents:
        print("Documents loading complete ✅")
        print(f"Number of documents loaded: {len(documents)}")
    else:
        print("No documents loaded!")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

document_path = "/home/ubuntu/gsa-rag-backend/dev/GSA-buyers-guide/"
with concurrent.futures.ThreadPoolExecutor() as executor:
    doc_texts = [t.page_content for t in load_documents(document_path)]
    print("Initializing Pinecone for document search...")
    docsearch_future = executor.submit(PineconeLang.from_texts, doc_texts, bedrock_embeddings, index_name=index_name)
    chain_future = executor.submit(load_qa_chain, llm, chain_type="stuff")

docsearch = docsearch_future.result()
chain = chain_future.result()

# Example query
@app.get("/hello")
async def read_root():
    return {"message": "Hello, from GSA-RL-RAG V2"}

@app.post("/generate_answer/")
async def generate_answer(question: Question):
    try:
        user_question = question.user_question
        query = f"You are an AI assistant. {user_question}. Use provided context only."
        print("Query:", query)
        print("Searching for similar documents")
        
        # Search for similar documents
        docs = docsearch.similarity_search(query, k=80)
        # print("DOCS", docs)
        
        # Run QA chain
        output = chain.run(input_documents=docs, question=query)
        print("Output:", output)
        return output
    except Exception as e:
        return {"error": "An error occurred while generating answer. Please try again later."}


# Run the FastAPI app using uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_config=logging_config)


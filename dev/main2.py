import boto3
import os
import time
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.vectorstores import Pinecone as PineconeLang
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone, PodSpec
from tqdm.autonotebook import tqdm

# Initialize Bedrock client globally
def initialize_bedrock_client():
    role_arn = 'arn:aws:iam::992382738258:role/gsa-rag'
    session_name = 'AssumedRoleSession'
    region_name = 'us-east-1'
    sts_client = boto3.client('sts', region_name=region_name)

    response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)

    credentials = response['Credentials']
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
        region_name=region_name
    ).client(service_name="bedrock-runtime", region_name="us-east-1")

def load_documents():
    document_path = "/home/ubuntu/gsa-rag-backend/GSA-buyers-guide"
    loader = PyPDFDirectoryLoader(document_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

# Pinecone credentials
os.environ["PINECONE_API_KEY"] = "bb1c3c9c-55e5-4685-8d85-a2f646bf4d63"
os.environ["PINECONE_API_ENV"] = "gcp-starter"
index_name = "gsasubset"

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Initialize Bedrock client
bedrock_runtime = initialize_bedrock_client()

# Initialize Pinecone index
if index_name in pc.list_indexes().names():
    pc.delete_index(name=index_name)

# Create an index if not already there
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='dotproduct',
        spec=PodSpec(environment='gcp-starter')
    )

# Wait for index to finish initialization
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)

# Model ID for Bedrock
modelId = 'anthropic.claude-v2:1'

# Load documents
docs = load_documents()

# Initialize Bedrock and embeddings
llm = Bedrock(
    model_id=modelId,
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
docsearch = PineconeLang.from_texts(
    [t.page_content for t in docs],
    bedrock_embeddings,
    index_name=index_name
)

# Load QA chain
chain = load_qa_chain(llm, chain_type="stuff")

# Example query
query = "You are an AI assistant. I am planning to implement a zero trust architecture. Can you provide implementation guidance? Who can I contact in GSA? Use provided context only."

# Search for similar documents
docs = docsearch.similarity_search(query, k=40)

# Run QA chain
output = chain.run(input_documents=docs, question=query)
print(output)


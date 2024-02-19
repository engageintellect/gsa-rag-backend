import boto3
import json
import os



from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import PodSpec
from tqdm.autonotebook import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np

bedrock_runtime = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-west-2"
)

modelId = 'anthropic.claude-v2:1'
#modelId = 'meta.llama2-13b-v1'

accept = 'application/json'
contentType = 'application/json'


import numpy as np
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("/Users/r3dux/gsa-rag-rl-backend/GSA-buyers-guide")

#loader= DirectoryLoader("/home/ec2-user/SageMaker/text")
                        
documents = loader.load()
# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2000,
    chunk_overlap=100,
)
docs = text_splitter.split_documents(documents)
len(docs)

os.environ["PINECONE_API_KEY"] = "bb1c3c9c-55e5-4685-8d85-a2f646bf4d63"
os.environ["PINECONE_API_ENV"] = "gcp-starter"
index_name = "gsasubset"

from pinecone import Pinecone, PodSpec
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
if index_name in pc.list_indexes().names():
    pc.delete_index(name=index_name)
# Now create an index if not already there
if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='dotproduct',
            spec=PodSpec(
                environment='gcp-starter'
            )
        )
# wait for index to finish initialization
while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
        
llm = Bedrock(
    model_id=modelId,
    client=bedrock_runtime,
    model_kwargs={
                    'max_tokens_to_sample': 4096,
                    'temperature': 1.0,
                    'top_k': 250,
                    'top_p': 0.999}
)
bedrock_embeddings = BedrockEmbeddings(client=bedrock_runtime) 


from langchain_community.vectorstores import Pinecone as PineconeLang
docsearch = PineconeLang.from_texts(
    [t.page_content for t in docs],
    bedrock_embeddings,
    index_name = index_name
)


from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type = "stuff")


query = "You are an AI assistant.  I am planning to implement a zero trust architecture. Can you provide implementation guidance? who can I contact in GSA ?Use provided context only."
docs = docsearch.similarity_search(query,k=40)
output = chain.run(input_documents = docs, question = query)
print(output)



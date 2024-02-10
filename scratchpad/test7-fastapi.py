from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
from langchain import FewShotPromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain_community.document_loaders import UnstructuredHTMLLoader

app = FastAPI()

class Question(BaseModel):
    user_question: str

# Define AWS credentials globally
role_arn = 'arn:aws:iam::992382738258:role/gsa-rag'
session_name = 'AssumedRoleSession'
region_name = 'us-east-1'
sts_client = boto3.client('sts', region_name=region_name)

# Initialize Bedrock client globally
def initialize_bedrock_client():
    response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
    credentials = response['Credentials']
    return boto3.Session(
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
        region_name=region_name
    ).client(service_name="bedrock-runtime", region_name="us-east-1")

bedrock_runtime = initialize_bedrock_client()

# List of document paths
document_paths = [
    "/home/ubuntu/gsa-rag-backend/docs/0VV35P.3RLG4G_47QTCA20D00B2_INDUCTIVEHEALTH123020.html",
    "/home/ubuntu/gsa-rag-backend/docs/0WG872.3S6L5T_47QTCA20D006Y_TMETRICSIFSSPRICELISTREV.html",
    "/home/ubuntu/gsa-rag-backend/docs/0WTKWJ.3SJXV8_47QTCA20D00EY_47QTCA20D00EY.html"
]

# Load documents and concatenate their contents
fulltext = ""
for path in document_paths:
    loader = UnstructuredHTMLLoader(path)
    document = loader.load()
    for page in document:
        fulltext += page.page_content
    fulltext += "\n****************End of this Document**************\n"
    fulltext += "****************Start of next Document**************\n"

# Model ID for Bedrock
model_id = 'anthropic.claude-v2:1'

# Hello World endpoint
@app.get("/hello")
async def read_root():
    return {"message": "Hello, World!"}

@app.post("/generate_answer/")
async def generate_answer(question: Question):
    # Q&A prompt template
    user_question = question.user_question
    question_template = """
    You are a helpful assistant. The {doc_text} contains quotes from 4 companies.
    Answer the following questions only on provided info.
    List all company names.
    """

    question_template = question_template + user_question
    prompt = PromptTemplate(template=question_template, input_variables=[""])

    # Initialize LLM chain and run
    bedrock_llm = Bedrock(
        model_id=model_id,
        client=bedrock_runtime,
        model_kwargs={
            'max_tokens_to_sample': 4096,
            'temperature': 0.0,
            'top_k': 250,
            'top_p': 0.999
        }
    )

    llm_chain = LLMChain(prompt=prompt, llm=bedrock_llm)
    output = llm_chain.run({"doc_text": fulltext})
    return output


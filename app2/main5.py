from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from langchain import FewShotPromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain_community.document_loaders import UnstructuredHTMLLoader
import uvicorn
import logging.config

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

document_path = "/home/ubuntu/gsa-rag/docs/"
document_names = [
    "0VV35P.3RLG4G_47QTCA20D00B2_INDUCTIVEHEALTH123020.html",
    "0WG872.3S6L5T_47QTCA20D006Y_TMETRICSIFSSPRICELISTREV.html",
    "0WTKWJ.3SJXV8_47QTCA20D00EY_47QTCA20D00EY.html",
    "0VVTSA.3RM6R1_GS-35F-148DA_GSAPRECISEDIGITALGS35F148DAIFSS6002021REV.html",
]

document_paths = []
for document_name in document_names:
    document_paths.append(document_path + document_name)

print(document_paths)



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
    uvicorn_logger.info("Hello endpoint was accessed.")
    return {"message": "Hello, World!"}

@app.post("/generate_answer/")
async def generate_answer(question: Question):
    try:
        uvicorn_logger.info("Received question: %s", question.user_question)

        # Rest of your code...

        return output
    except Exception as e:
        uvicorn_logger.exception("An error occurred while generating answer: %s", str(e))
        return {"error": "An error occurred"}

# Configure UVicorn logging
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": "/home/ubuntu/gsa-rag/logs/uvicorn.log",
        },
        "stream": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        }
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["file", "stream"],  # Add the stream handler here
            "level": "INFO",
        }
    }
}


# Run the FastAPI app using uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_config=logging_config)

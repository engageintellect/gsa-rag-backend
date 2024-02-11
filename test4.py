import boto3
from langchain import FewShotPromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain_community.document_loaders import UnstructuredHTMLLoader

# Define the ARN of the role to assume
role_arn = 'arn:aws:iam::992382738258:role/gsa-rag'
# Define a session name for the assumed role session
session_name = 'AssumedRoleSession'
# Define the AWS region
region_name = 'us-east-1'

# Create an STS client and assume the role
sts_client = boto3.client('sts', region_name=region_name)
response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
credentials = response['Credentials']
session = boto3.Session(
    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken'],
    region_name=region_name
)

# Create Bedrock client
bedrock_runtime = session.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken']
)

# List of document paths
document_paths = [
    "/home/ubuntu/gsa-rag/docs/0VV35P.3RLG4G_47QTCA20D00B2_INDUCTIVEHEALTH123020.html",
    "/home/ubuntu/gsa-rag/docs/0WG872.3S6L5T_47QTCA20D006Y_TMETRICSIFSSPRICELISTREV.html",
    "/home/ubuntu/gsa-rag/docs/0WTKWJ.3SJXV8_47QTCA20D00EY_47QTCA20D00EY.html"
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


# Q&A prompt template
user_input = "which company provides the best prompt payment terms?"
question_template = """
You are a helpful assistant. The {doc_text} contains quotes from 4 companies.
Answer the following questions only on provided info.
List all company names.

{user_input}
"""
prompt = PromptTemplate(template=question_template, input_variables=["user_input"])




# Initialize Bedrock client
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

# Initialize LLM chain and run
llm_chain = LLMChain(prompt=prompt, llm=bedrock_llm)
output = llm_chain.run({"user_input": user_input, "doc_text": fulltext})
print(output)


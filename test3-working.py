import boto3
from langchain import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader



# Define the ARN of the role to assume
role_arn = 'arn:aws:iam::992382738258:role/gsa-rag'

# Define a session name for the assumed role session
session_name = 'AssumedRoleSession'

# Define the AWS region
region_name = 'us-east-1'

# Create an STS client
sts_client = boto3.client('sts', region_name=region_name)

# Assume the role
response = sts_client.assume_role(
    RoleArn=role_arn,
    RoleSessionName=session_name
)

# Extract the temporary credentials from the response
credentials = response['Credentials']

# Use the temporary credentials to create a new session
session = boto3.Session(
    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken'],
    region_name=region_name  # Specify the region
)

# Now you can use the new session to make AWS API calls with the permissions of the assumed role
bedrock_client = session.client('bedrock-runtime')
print(bedrock_client)

# Use the client to make API calls
# For example:
# response = bedrock_client.describe_runtime_instances()


bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",


    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken'],


    #aws_access_key_id=aws_access_key_id,
    #aws_secret_access_key=aws_secret_access_key,
    #aws_session_token=aws_session_token
)


print(bedrock_runtime)







# List of document paths
document_paths = [
    "/home/ubuntu/gsa-rag-backend/docs/0VV35P.3RLG4G_47QTCA20D00B2_INDUCTIVEHEALTH123020.html",
    "/home/ubuntu/gsa-rag-backend/docs/0WG872.3S6L5T_47QTCA20D006Y_TMETRICSIFSSPRICELISTREV.html",
    "/home/ubuntu/gsa-rag-backend/docs/0WTKWJ.3SJXV8_47QTCA20D00EY_47QTCA20D00EY.html"       
]

# Initialize fulltext
fulltext = ""

# Load documents in a loop
for path in document_paths:
    loader = UnstructuredHTMLLoader(path)
    document = loader.load()
    for page in document:
        fulltext += page.page_content
    fulltext += "\n****************End of this Document**************\n"
    fulltext += "****************Start of next Document**************\n"


len(fulltext)


model_id = 'anthropic.claude-v2:1'
#model_id = 'meta.llama2-13b-chat-v1'

# Q&A
question_template = """

You are a helpful assistant. The {doc_text} contains quotes from 4 companies.Answer the following questions only on provided info.

List all company names.
which company provides the best prompt payment terms?

"""
prompt = PromptTemplate(template=question_template,input_variables=[""])

#Change this to OpenAI call if you want to test with OpenAI
bedrock_llm = Bedrock(
    model_id=model_id,
    client=bedrock_runtime,
    model_kwargs={
                    'max_tokens_to_sample': 4096,
                    'temperature': 0.0,
                    'top_k': 250,
                    'top_p': 0.999}
)
llm_chain = LLMChain(prompt=prompt, llm=bedrock_llm)
output = llm_chain.run({"doc_text": fulltext})
print(output)

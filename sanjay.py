import boto3
from langchain import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader

bedrock_runtime = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-west-2"
)

import sys
sys.path.append('/home/ec2-user/anaconda3/envs/python3/lib/python3.10/site-packages/unstructured')
print(sys.path)

#Initialize to null and start appending docs
fulltext = ""

loader = UnstructuredHTMLLoader("/home/ec2-user/SageMaker/html/0VV35P.3RLG4G_47QTCA20D00B2_INDUCTIVEHEALTH123020.html")
document = loader.load()
for page in document:
    #print(page.page_content)
    fulltext += page.page_content
fulltext += "\n****************End of this Document**************\n"
fulltext += "****************Start of next Document**************\n"
#print(fulltext)

#Second doc

loader = UnstructuredHTMLLoader("/home/ec2-user/SageMaker/html/0WG872.3S6L5T_47QTCA20D006Y_TMETRICSIFSSPRICELISTREV.html")
document = loader.load()
for page in document:
    #print(page.page_content)
    fulltext += page.page_content
fulltext += "\n****************End of this Document**************\n"
fulltext += "****************Start of next Document**************\n"
#print(fulltext)

#Third doc

loader = UnstructuredHTMLLoader("/home/ec2-user/SageMaker/html/0WTKWJ.3SJXV8_47QTCA20D00EY_47QTCA20D00EY.html")
document = loader.load()
for page in document:
    #print(page.page_content)
    fulltext += page.page_content
fulltext += "\n****************End of this Document**************\n"
fulltext += "****************Start of next Document**************\n"
#print(fulltext)


len(fulltext)


model_id = 'anthropic.claude-v2:1'

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

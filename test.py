import boto3

# Define the ARN of the role to assume
role_arn = 'arn:aws:iam::992382738258:role/gsa-rag'

# Define a session name for the assumed role session
session_name = 'AssumedRoleSession'

# Create an STS client
sts_client = boto3.client('sts')

# Assume the role
response = sts_client.assume_role(
    RoleArn=role_arn,
    RoleSessionName=session_name
)

# Extract the temporary credentials from the response
credentials = response['Credentials']
print(credentials)

# Use the temporary credentials to create a new session
session = boto3.Session(
    aws_access_key_id=credentials['AccessKeyId'],
    aws_secret_access_key=credentials['SecretAccessKey'],
    aws_session_token=credentials['SessionToken']
)

# Now you can use the new session to make AWS API calls with the permissions of the assumed role
s3_client = session.client('s3')
buckets = s3_client.list_buckets()

# Print the list of S3 buckets (just an example)
print(buckets)




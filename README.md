
# Finetuning Azure OpenAI

This repository contains a notebook for fine-tuning an Azure OpenAI model. The notebook provides instructions and code examples for preparing, uploading, and fine-tuning data on Azure OpenAI.

## Table of Contents

1. [Prepare Your Training and Validation Data](#prepare-your-training-and-validation-data)
2. [Upload Your Training Data](#upload-your-training-data)
3. [Create a Customized Model](#create-a-customized-model)
4. [Check Fine-Tuning Job Status](#check-fine-tuning-job-status)
5. [List Fine-Tuning Events](#list-fine-tuning-events)
6. [Checkpoints](#checkpoints)
7. [Deploy a Fine-Tuned Model](#deploy-a-fine-tuned-model)
8. [Cross Region Deployment](#cross-region-deployment)
9. [Cross Tenant Deployment](#cross-tenant-deployment)
10. [Use a Deployed Customised Model](#use-a-deployed-customised-model)
11. [Analyze Your Customised Model](#analyze-your-customised-model)

## Prepare Your Training and Validation Data

Your training and validation data sets consist of input and output examples for how you would like the model to perform.

The more training examples you have, the better. Fine-tuning jobs will not proceed without at least 10 training examples, but such a small number is not enough to noticeably influence model responses. It is best practice to provide hundreds, if not thousands, of training examples to be successful.

In general, doubling the dataset size can lead to a linear increase in model quality. But keep in mind, low-quality examples can negatively impact performance. If you train the model on a large amount of internal data, without first pruning the dataset for only the highest quality examples, you could end up with a model that performs much worse than expected.

The training and validation data you use must be formatted as a JSON Lines (JSONL) document. For gpt-35-turbo-0613, the fine-tuning dataset must be formatted in the conversational format that is used by the Chat completions API.

### Example File Format

See [example.jsonl](data/example.jsonl) for JSONL example.

### Multi-Turn Chat File Format

Multiple turns of a conversation in a single line of your JSONL training file is also supported. To skip fine-tuning on specific assistant messages, add the optional weight key-value pair. Currently, weight can be set to 0 or 1.

See [multi-turn-example.jsonl](data/multi-turn-example.jsonl) for JSONL example.

### Chat Completions with Vision

See [vision-example.jsonl](data/vision-example.jsonl) for JSONL example.

In addition to the JSONL format, training and validation data files must be encoded in UTF-8 and include a byte-order mark (BOM). The file must be less than 512 MB in size.

## Upload Your Training Data

There are two ways to upload training data:

* [From a local file](https://learn.microsoft.com/en-us/rest/api/azureopenai/files/upload?view=rest-azureopenai-2024-10-21&tabs=HTTP)
* [Import from an Azure Blob store or other web location](https://learn.microsoft.com/en-us/rest/api/azureopenai/files/import?view=rest-azureopenai-2024-10-21&tabs=HTTP)

For large data files, it's recommended that you import from an Azure Blob store. Large files can become unstable when uploaded through multipart forms because the requests are atomic and can't be retried or resumed.

The following Python example uploads local training and validation files by using the Python SDK, and retrieves the returned file IDs:

```python
import os
from openai import AzureOpenAI

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version=os.getenv("API_VERSION")  # This API version or later is required to access seed/events/checkpoint capabilities
)

training_file_name = 'training_set.jsonl'
validation_file_name = 'validation_set.jsonl'

# Upload the training and validation dataset files to Azure OpenAI with the SDK.

training_response = client.files.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)
training_file_id = training_response.id

validation_response = client.files.create(
    file=open(validation_file_name, "rb"), purpose="fine-tune"
)
validation_file_id = validation_response.id

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)
```

## Create a Customized Model

After you upload your training and validation files, you're ready to start the fine-tuning job.

The following Python code shows an example of how to create a new fine-tune job with the Python SDK:

```python
response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model=os.getenv("MODEL_NAME"), # Enter base model name. Note that in Azure OpenAI, the model name contains dashes and cannot contain dot/period characters. 
    seed = 105  # seed parameter controls reproducibility of the fine-tuning job. If no seed is specified, one will be generated automatically.
)

job_id = response.id

# You can use the job ID to monitor the status of the fine-tuning job.
# The fine-tuning job will take some time to start and complete.

print("Job ID:", response.id)
print("Status:", response.id)
print(response.model_dump_json(indent=2))
```

## Check Fine-Tuning Job Status

To check the status of your fine-tuning job, use the following code:

```python
response = client.fine_tuning.jobs.retrieve(job_id)

print("Job ID:", response.id)
print("Status:", response.status)
print(response.model_dump_json(indent=2))
```

## List Fine-Tuning Events

To examine the individual fine-tuning events that were generated during training:

```python
response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
print(response.model_dump_json(indent=2))
```

## Checkpoints

When each training epoch completes, a checkpoint is generated. A checkpoint is a fully functional version of a model that can both be deployed and used as the target model for subsequent fine-tuning jobs.

You can retrieve the list of checkpoints associated with an individual fine-tuning job by running the following:

```python
response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
print(response.model_dump_json(indent=2))
```

## Deploy a Fine-Tuned Model

When the fine-tuning job succeeds, the value of the `fine_tuned_model` variable in the response body is set to the name of your customized model. Your model is now also available for discovery from the list Models API. However, you can't issue completion calls to your customized model until it is deployed.

Deploy your customized model using the Azure API:

```python
import json
import requests

token= os.getenv("TOKEN") 
subscription = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP_NAME")
resource_name = os.getenv("AZURE_OPENAI_RESOURCE_NAME")
model_deployment_name ="gpt-35-turbo-ft" # Custom deployment name that you will use to reference the model when making inference calls.

deploy_params = {'api-version': "2023-05-01"} 
deploy_headers = {'Authorization': 'Bearer {}'.format(token), 'Content-Type': 'application/json'}

deploy_data = {
    "sku": {"name": "standard", "capacity": 1}, 
    "properties": {
        "model": {
            "format": "OpenAI",
            "name": "<fine_tuned_model>", # Retrieve this value from the previous call
            "version": "1"
        }
    }
}

deploy_data = json.dumps(deploy_data)

request_url = f'https://management.azure.com/subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}'

print('Creating a new deployment...')

r = requests.put(request_url, params=deploy_params, headers=deploy_headers, data=deploy_data)

print(r)
print(r.reason)
print(r.json())
```

## Cross Region Deployment

Fine-tuning supports deploying a fine-tuned model to a different region than where the model was originally fine-tuned. You can deploy to a different subscription/region as well.

Below is an example of deploying a model that was fine-tuned in one subscription/region to another:

```python
import json
import requests

token= os.getenv("<TOKEN>") 

subscription = "<DESTINATION_SUBSCRIPTION_ID>"  
resource_group = "<DESTINATION_RESOURCE_GROUP_NAME>"
resource_name = "<DESTINATION_AZURE_OPENAI_RESOURCE_NAME>"

source_subscription = "<SOURCE_SUBSCRIPTION_ID>"
source_resource_group = "<SOURCE_RESOURCE_GROUP>"
source_resource = "<SOURCE_RESOURCE>"

source = f'/subscriptions/{source_subscription}/resourceGroups/{source_resource_group}/providers/Microsoft.CognitiveServices/accounts/{source_resource}'

model_deployment_name ="gpt-35-turbo-ft" # Custom deployment name that you will use to reference the model when making inference calls.

deploy_params = {'api-version': "2023-05-01"} 
deploy_headers = {'Authorization': 'Bearer {}'.format(token), 'Content-Type': 'application/json'}

deploy_data = {
    "sku": {"name": "standard", "capacity": 1}, 
    "properties": {
        "model": {
            "format": "OpenAI",
            "name": "<FINE_TUNED_MODEL_NAME>", # This value will look like gpt-35-turbo-0613.ft-0ab3f80e4f2242929258fff45b56a9ce 
            "version": "1",
            "source": source
        }
    }
}
deploy_data = json.dumps(deploy_data)

request_url = f'https://management.azure.com/subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}'

print('Creating a new deployment...')

r = requests.put(request_url, params=deploy_params, headers=deploy_headers, data=deploy_data)

print(r)
print(r.reason)
print(r.json())
```

## Cross Tenant Deployment

The account used to generate access tokens with `az account get...

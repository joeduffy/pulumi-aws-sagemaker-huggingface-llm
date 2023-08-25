[![Deploy](https://get.pulumi.com/new/button.svg)](https://app.pulumi.com/new?template=https://github.com/joeduffy/pulumi-aws-sagemaker-huggingface-llm/blob/master/README.md)

# Hugging Face LLM on AWS SageMaker using Pulumi

This repo contains a simple `HuggingFaceLlm` component, defined in Python using Pulumi, which configures
and deploys the necessary machinery to run a Hugging Face LLM model to Amazon SageMaker. It uses the Hugging Face
LLM Inference Container for Amazon SageMaker [announced here](https://huggingface.co/blog/sagemaker-huggingface-llm).
This leverages [Amazon Deep Learning Containers (DLCs)](
https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/what-is-dlc.html) which are pre-built Docker
containers for TensorFlow, PyTorch, etc.

An example of using this component is included in the `__main__.py` file:

```python
import pulumi
from hugging_face_llm import HuggingFaceLlm

llm = HuggingFaceLlm('Llama2Llm',
    instance_type = 'ml.g5.2xlarge',
    environment_variables = {
        'HF_MODEL_ID': 'NousResearch/Llama-2-7b-chat-hf',
        'SM_NUM_GPUS': '1',
        'MAX_INPUT_LENGTH': '2048',
        'MAX_TOTAL_TOKENS': '4096',
        'MAX_BATCH_TOTAL_TOKENS': '8192',
    },
)

pulumi.export('EndpointName', llm.endpoint.name)
```

All you need to do is initialize a Pulumi stack with `pulumi init`, configure the AWS region
(e.g., `pulumi config set aws:region us-west-2`), and then run `pulumi up`. Magic should happen:

```
Updating (dev)

View in Browser (Ctrl+O): https://app.pulumi.com/joeduffy/aws-sagemaker-huggingface-llm/dev/updates/1

     Type                                       Name                               Status                  Info
 +   pulumi:pulumi:Stack                        aws-sagemaker-huggingface-llm-dev  created
 +   └─ huggingface:llm:HuggingFaceLlm          Llama2Llm                          created
 +      ├─ aws:iam:Role                         Llama2Llm-role                     created (1s)
 +      ├─ aws:sagemaker:Model                  Llama2Llm-model                    created (6s)
 +      ├─ aws:sagemaker:EndpointConfiguration  Llama2Llm-config                   created (0.57s)
 +      └─ aws:sagemaker:Endpoint               Llama2Llm-endpoint                 created (7m8s)


Resources:
    + 5 created

Duration: 7m15s
```


There are two todos. This is not yet a multi-language component so can only be used in Python. It's also not
distributed in any package managers yet, so you need to bundle the code.

This repo was inspired by https://github.com/philschmid/aws-sagemaker-huggingface-llm.

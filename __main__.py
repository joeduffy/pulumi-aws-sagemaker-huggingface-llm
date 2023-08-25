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

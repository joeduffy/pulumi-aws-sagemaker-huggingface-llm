"""Hugging Face LLM on AWS SageMaker."""
"""Inspired by https://github.com/philschmid/aws-sagemaker-huggingface-llm"""

import json
import pulumi
from pulumi_aws import config
from pulumi_aws import iam
from pulumi_aws import sagemaker
from sagemaker import huggingface
from typing import Mapping, Optional

class HuggingFaceLlm(pulumi.ComponentResource):
    def __init__(
            self,
            name: str,
            instance_type: str,
            environment_variables: Mapping[str, str],
            s3_model_path: Optional[str] = None,
            tgi_version: str = '0.9.3',
            pytorch_version: str = '2.0.1',
            startup_health_check_timeout_in_seconds: int = 600,
            opts: Optional[pulumi.ResourceOptions] = None):
        # Initialize the parent resource state.
        super().__init__('huggingface:llm:HuggingFaceLlm', name, None, opts)

        # Initialize all of this object's properties.
        self.instance_type = instance_type
        self.environment_variables = environment_variables
        self.name = name
        self.s3_model_path = s3_model_path
        self.tgi_version = tgi_version
        self.pytorch_version = pytorch_version
        self.startup_health_check_timeout_in_seconds = startup_health_check_timeout_in_seconds

        # Now initialize all associated AWS resources.
        is_s3_model = s3_model_path is not None
        is_huggingface_hub_model = environment_variables['HF_MODEL_ID']
        if is_s3_model and is_huggingface_hub_model:
            raise Exception('Cannot specify both `environment_variables["HF_MODEL_ID"]` and `s3_model_path`')
        elif not is_s3_model and not is_huggingface_hub_model:
            raise Exception('Must specify either `environment_variables["HF_MODEL_ID"]` or `s3_model_path`')

        # Get Hugging Face LLM container image.
        container_image = huggingface.get_huggingface_llm_image_uri(
            backend = 'huggingface',
            region = config.region,
            version = tgi_version,
            # pytorch_version = pytorch_version,
        )

        # Create the IAM role.
        role = iam.Role(f'{name}-role',
            assume_role_policy=json.dumps({
                'Version': '2012-10-17',
                'Statement': [{
                    'Effect': 'Allow',
                    'Principal': { 'Service': 'sagemaker.amazonaws.com' },
                    'Action': 'sts:AssumeRole',
                }],
            }),
            managed_policy_arns = [ 'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess' ],
            opts = pulumi.ResourceOptions(parent = self),
        )

        # Create the SageMaker model:
        sage_maker_model = sagemaker.Model(f'{name}-model',
            execution_role_arn = role.arn,
            primary_container=sagemaker.ModelContainerArgs(
                image = container_image,
                environment = environment_variables,
                model_data_url = s3_model_path,
            ),
            opts = pulumi.ResourceOptions(parent = self),
        )

        # Create the SageMaker endpoint configuration.
        cfn_endpoint_config = sagemaker.EndpointConfiguration(f'{name}-config',
            production_variants = [
                sagemaker.EndpointConfigurationProductionVariantArgs(
                    model_name = sage_maker_model.name,
                    variant_name = 'primary',
                    initial_variant_weight = 1.0,
                    initial_instance_count = 1,
                    instance_type = instance_type,
                    container_startup_health_check_timeout_in_seconds = startup_health_check_timeout_in_seconds,
                ),
            ],
            opts = pulumi.ResourceOptions(parent = self),
        )

        # Deploy SageMaker endpoint.
        self.endpoint = sagemaker.Endpoint(f'{name}-endpoint',
            endpoint_config_name = cfn_endpoint_config.name,
            opts = pulumi.ResourceOptions(parent = self),
        )
        # TODO: what about BridgeEndpointConfig?

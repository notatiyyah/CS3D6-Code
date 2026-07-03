'''
Training / Processing Job Orchestrator (Requires AWS Access). 
Bundles up the src/ folder and sends it to the docker container (ml.g5.xlarge) to run.
Note: Can be tested locally using Docker if you set config.is_local to True. 
However, you still need AWS credentials to run locally since it requires AWS ECR (Elastic Container Repository) to download the container images.
'''

from boto3 import session as boto_session

from sagemaker.local import LocalSession
from sagemaker import session
from sagemaker.pytorch import PyTorch, PyTorchProcessor
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

from dataclasses import dataclass

@dataclass
class Config:
    # AWS Config
    is_local: bool                 = False
    aws_profile_name: str          = "data-platform-admin-dev"
    aws_region: str                = "eu-west-2"
    role_arn: str                  = "arn:aws:iam::484466746276:role/service-role/AmazonSageMaker-ExecutionRole-20260617T135400"
    s3_bucket: str                 = "s3://additional-needs-data-dev/"
    tags                           = [
        {"Key": "Application", "Value": "housing-additional-needs"},
        {"Key": "TeamEmail", "Value": "shared.services@hackney.gov.uk"},
        {"Key": "Environment", "Value": "dev"},
        {"Key": "Confidentiality", "Value": "Internal"}
    ]

    # Paths
    source_dir: str                = "src"
    
    # Training URIs
    s3_train_data: str             = f"{s3_bucket}data/train_data.json"
    s3_val_data: str               = f"{s3_bucket}data/val_data.json"
    s3_output_path: str            = f"{s3_bucket}output/"
    
    # Inference URIs
    s3_span_model_uri: str         = f"{s3_bucket}models/needs-span-classifier"
    s3_relation_model_uri: str     = f"{s3_bucket}models/needs-relation-classifier"
    s3_output_predictions_uri: str = f"{s3_bucket}predictions/"



def _setup_session(config: Config):
    aws_session = boto_session.Session(profile_name=config.aws_profile_name, region_name=config.aws_region)
    if config.is_local:
        sagemaker_session = LocalSession(boto_session=aws_session)
        sagemaker_session.config = {'local': {'local_code': True}}
    else:
        sagemaker_session = session.Session(boto_session=aws_session)
    return sagemaker_session

def run_train():
    config = Config()
    sagemaker_session = _setup_session(config)

    # Set up training job
    estimator = PyTorch(
        entry_point="sagemaker_entrypoint_train.py",
        source_dir=config.source_dir,
        sagemaker_session=sagemaker_session,
        role=config.role_arn,
        framework_version='2.1.0',
        py_version='py310',
        instance_count=1,
        instance_type='local' if config.is_local else 'ml.g5.xlarge',
        tags=config.tags,
        output_path= config.s3_output_path,
    )

    # Kick off the job
    inputs = {
        "train": config.s3_train_data,
        "val": config.s3_val_data,
    }
    estimator.fit(inputs=inputs)

def run_inference():
    config = Config()
    sagemaker_session = _setup_session(config)

    # Set up processing job
    processor = PyTorchProcessor(
        framework_version="2.1.0",
        py_version="py310", 
        role=config.role_arn,
        instance_count=1,
        instance_type='local' if config.is_local else 'ml.m5.xlarge',
        base_job_name="an-e2e-inference-poc",
        tags=config.tags
    )

    # Run the sequential pipeline
    processor.run(
        code="sagemaker_entrypoint_inference.py",
        source_dir=config.source_dir,
        inputs=[
            ProcessingInput(
                source=config.s3_val_data,
                destination="/opt/ml/processing/input/raw_data"
            ),
            ProcessingInput(
                source=config.s3_span_model_uri,
                destination="/opt/ml/processing/input/span_model",
            ),
            ProcessingInput(
                source=config.s3_relation_model_uri,
                destination="/opt/ml/processing/input/relation_model",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=config.s3_output_predictions_uri
            )
        ]
    )


if __name__ == "__main__":
    run_train()
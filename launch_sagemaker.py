'''
Training Job Orchestrator (Requires AWS Access). 
Bundles up the src/ folder and sends it to the docker container (ml.g5.xlarge) to run.
Note: Can be tested locally using Docker if you set config.is_local to True. 
However, you still need AWS credentials to run locally since it requires AWS ECR (Elastic Container Repository) to download the container images.
'''

from boto3 import session as boto_session
from sagemaker.local import LocalSession
from sagemaker import session
from sagemaker.pytorch import PyTorch

from dataclasses import dataclass

@dataclass
class Config:
    job_name: str                  = "e2e-inference-pipeline"
    # AWS Config
    is_local: bool                 = False
    aws_profile_name: str          = "data-platform-dev-admin"
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
    entry_point_file: str          = "sagemaker_entrypoint.py" # Wrapper file - update in this file to change which script is called.
    
    # Training URIs
    s3_train_data: str             = f"{s3_bucket}data/train_data.json"
    s3_val_data: str               = f"{s3_bucket}data/val_data.json"
    s3_output_path: str            = f"{s3_bucket}output/"
    
    # Inference URIs
    s3_span_model_uri: str         = f"{s3_bucket}models/needs-span-classifier"
    s3_relation_model_uri: str     = f"{s3_bucket}models/needs-relation-classifier"
    s3_output_predictions_uri: str = f"{s3_bucket}predictions/"


def main():
    config = Config()

    # Set up session
    aws_session = boto_session.Session(profile_name=config.aws_profile_name)
    if config.is_local:
        sagemaker_session = LocalSession(boto_session=aws_session)
        sagemaker_session.config = {'local': {'local_code': True}}
    else:
        sagemaker_session = session.Session(boto_session=aws_session)

    # Set up training job
    estimator = PyTorch(
        entry_point=config.entry_point_file,
        source_dir=config.source_dir,
        sagemaker_session=sagemaker_session,
        role=config.role_arn,
        framework_version='2.1.0',
        py_version='py310',
        instance_count=1,
        instance_type='local' if config.is_local else 'ml.g5.xlarge',
        tags=config.tags,
        output_path= config.s3_output_predictions_uri,
    )

    # Kick off the job
    inputs = {
        "train": config.s3_train_data,
        "val": config.s3_val_data,
        "span_model": config.s3_span_model_uri,
        "relation_model": config.s3_relation_model_uri
    }
    estimator.fit(inputs=inputs, job_name=config.job_name)

if __name__ == "__main__":
    main()
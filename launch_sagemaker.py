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
from src.common.logging import setup_logger

@dataclass
class Config:
    logger                = setup_logger('sagemaker.training', 'sagemaker.training.log')

    # AWS Config
    is_local: bool        = True
    aws_profile_name: str = "data-platform-dev-admin"
    role_arn: str         = "arn:aws:iam::484466746276:role/service-role/AmazonSageMaker-ExecutionRole-20260617T135400"
    s3_bucket: str        = "s3://additional-needs-data-dev/"
    tags: list[dict]      = [
        {"Key": "Application", "Value": "housing-additional-needs"},
        {"Key": "TeamEmail", "Value": "shared.services@hackney.gov.uk"},
        {"Key": "Environment", "Value": "dev"},
        {"Key": "Confidentiality", "Value": "Internal"}
    ]

    # Paths
    source_dir: str       = "src"
    entry_point_file: str = "training/span_level/train_span.py" # Change to whatever script you want
    s3_train_data: str    = f"{s3_bucket}data/train/"
    s3_val_data: str      = f"{s3_bucket}data/val/"
    s3_output_path: str   = f"{s3_bucket}output/"


def main():
    config = Config()

    # Set up session
    boto_session = boto_session.Session(profile_name=config.aws_profile_name)
    if config.is_local:
        sagemaker_session = LocalSession(boto_session=boto_session)
        sagemaker_session.config = {'local': {'local_code': True}}
    else:
        sagemaker_session = session.Session(boto_session=boto_session)

    # Set up training job
    pt_estimator = PyTorch(
        entry_point=config.entry_point_file,
        source_dir=config.source_dir,
        sagemaker_session=sagemaker_session,
        role=role,
        framework_version='2.1.0',
        py_version='py310',
        instance_count=1,
        instance_type='local' if IS_LOCAL else 'ml.g5.xlarge',
        tags=tags,
        output_path= config.s3_output_path,
    )

    # Kick off the job
    estimator.fit({
        "train": config.s3_train_data,
        "val": config.s3_val_data
    })

if __name__ == "__main__":
    main()
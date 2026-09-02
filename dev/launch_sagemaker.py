'''
Training / Processing Job Orchestrator (Requires AWS Access).
Bundles this `dev/` package and sends it to the docker container (ml.g5.xlarge).
Note: Can be tested locally using Docker if you set config.is_local to True.
However, you still need AWS credentials to run locally since it requires AWS ECR
(Elastic Container Repository) to download the container images.
'''
from dataclasses import dataclass
from pathlib import Path
import logging
from boto3 import session as boto_session

from sagemaker.local import LocalSession
from sagemaker import session
from sagemaker.pytorch import PyTorch

# This file lives inside the `dev/` package — always upload this directory,
# regardless of the caller's working directory.
DEV_PACKAGE_DIR = str(Path(__file__).resolve().parent)

@dataclass
class Config:
    # Training config
    model_name: str         = "housing-additional-needs-spans"
    model_desc: str         = "DeBERTa span model"
    source_dir: str         = DEV_PACKAGE_DIR
    entry_point: str        = "spans/training/train_span.py"
    instance_type: str      = "ml.g5.xlarge"

    # AWS Config
    is_local: bool          = False
    aws_profile_name: str   = "data-platform-admin-dev"
    aws_region: str         = "eu-west-2"
    role_arn: str           = "arn:aws:iam::484466746276:role/service-role/AmazonSageMaker-ExecutionRole-20260617T135400"
    s3_bucket: str          = "s3://additional-needs-data-dev/"
    tags                    = [
        {"Key": "Application", "Value": "housing-additional-needs"},
        {"Key": "TeamEmail", "Value": "shared.services@hackney.gov.uk"},
        {"Key": "Environment", "Value": "dev"},
        {"Key": "Confidentiality", "Value": "Internal"}
    ]

    # S3 URIs
    s3_train_data: str      = f"{s3_bucket}data/train_data.json"
    s3_val_data: str        = f"{s3_bucket}data/val_data.json"
    s3_output_path: str     = f"{s3_bucket}output/"


def _setup_session(config: Config):
    aws_session = boto_session.Session(profile_name=config.aws_profile_name, region_name=config.aws_region)
    if config.is_local:
        sagemaker_session = LocalSession(boto_session=aws_session)
        sagemaker_session.config = {'local': {'local_code': True}}
    else:
        sagemaker_session = session.Session(boto_session=aws_session)
    return sagemaker_session


def main():
    config = Config()
    sagemaker_session = _setup_session(config)

    logger = logging.getLogger('train_sagemaker')
    logger.setLevel(logging.INFO)

    # Set up training job
    estimator = PyTorch(
        entry_point=config.entry_point,
        source_dir=config.source_dir,
        base_job_name=f"train_{config.model_name}",
        requirements="requirements-sagemaker.txt",
        sagemaker_session=sagemaker_session,
        role=config.role_arn,
        framework_version='2.1.0',
        py_version='py310',
        instance_count=1,
        instance_type='local' if config.is_local else config.instance_type,
        tags=config.tags,
        output_path= config.s3_output_path,
    )

    # Kick off the job
    logger.info("Starting training...")
    inputs = {
        "train": config.s3_train_data,
        "val": config.s3_val_data,
    }
    estimator.fit(inputs=inputs)

    # Register
    logger.info("Registering trained model version...")
    model_package = estimator.register(
        model_package_group_name=config.model_name,
        approval_status="PendingManualApproval",
        content_types=["application/json"],
        response_types=["application/json"],
        description=config.model_desc,
    )
    logger.info("Model registered successfully: %s", model_package.model_package_arn)

if __name__ == "__main__":
    main()

"""
This script ingests MTFH notes data and MTFH tenure data combines them,
and writes the results to S3 as Parquet, registering the output table in the AWS Glue Data Catalog.

The pipeline handles:
- Loading and executing the notes_ingest.sql query via AWS Athena
- Joining notes to tenures via tenure ID, asset ID, or person ID
- Appending data quality and business logic flags (e.g., inactive tenancies, organizational targets, missing targets) for downstream metrics
- Writing the results to S3 as Snappy-compressed Parquet
- Registering the output in the Glue Data Catalog under the target database
"""

import logging
import os
from pathlib import Path

import awswrangler as wr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SOURCE_DATABASE = os.environ.get("SOURCE_DATABASE")
TARGET_DATABASE = os.environ.get("TARGET_DATABASE")
TARGET_TABLE_NAME = os.environ.get("TARGET_TABLE_NAME")

S3_OUTPUT_BASE = os.environ.get("S3_OUTPUT_BASE")
KMS_KEY = os.environ.get("KMS_KEY")
WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "housing")
S3_ATHENA_STORAGE = os.environ.get("S3_ATHENA_STORAGE")


# Helpers
def validate_environment_vars() -> bool:
    required_vars = [
        "SOURCE_DATABASE",
        "TARGET_DATABASE",
        "TARGET_TABLE_NAME",
        "S3_OUTPUT_BASE",
        "KMS_KEY",
        "ATHENA_WORKGROUP",
        "S3_ATHENA_STORAGE"
    ]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logger.error("Missing required environment variables: %s", ", ".join(missing_vars))
        return False
    return True


def load_sql_from_file(sql_file: str) -> str:
    """Loads SQL from file."""
    sql_path = Path(f"/app/housing/additional_needs/sql/{sql_file}")
    logger.info(f"Loading SQL from: {sql_path}",)

    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    return sql_path.read_text()


# Entry point
def main() -> int:
    if not validate_environment_vars():
        return 1

    logger.info("Starting MTFH Additional Needs Notes Reshaping.")
    sql = load_sql_from_file("notes_ingest.sql")

    logger.info(f"Executing Athena query for tables: {SOURCE_DATABASE}.mtfh_notes, {SOURCE_DATABASE}.mtfh_tenureinformation")
    df = wr.athena.read_sql_query(
        sql=sql,
        database=str(SOURCE_DATABASE),
        ctas_approach=False,
        s3_output=S3_ATHENA_STORAGE,
        workgroup=WORKGROUP,
    )
    logger.info(f"Query returned {len(df)} rows")

    # Log summaries of the flag columns
    logger.info(f"Summary of note targets matfched:\n{df['flag_no_matching_target'].value_counts()}")
    logger.info(f"Summary of notes with date parse failures:\n{df['flag_note_date_parse_failed'].value_counts()}")
    logger.info(f"Summary of notes linked to inactive tenancies:\n{df['flag_inactive'].value_counts()}")
    logger.info(f"Summary of notes linked to organisations:\n{df['flag_is_organisation'].value_counts()}")

    logger.info(f"Writing table to {S3_OUTPUT_BASE}")
    wr.s3.to_parquet(
        df=df,
        path=f"{S3_OUTPUT_BASE}/{TARGET_TABLE_NAME}/",
        dataset=True,
        database=TARGET_DATABASE,
        table=TARGET_TABLE_NAME,
        partition_cols=["import_year", "import_month", "import_day", "import_date"],
        mode="overwrite",
        compression="snappy",
        s3_additional_kwargs={
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": KMS_KEY,
        },
        glue_table_settings={
            "description": "MTFH Notes reshaped (Additional Needs)",
            "parameters": {
                "source_system": "mtfh",
                "classification": "parquet",
            },
        },
    )

    logger.info(f"Successfully wrote table '{TARGET_DATABASE}'.'{TARGET_TABLE_NAME}' to {S3_OUTPUT_BASE}.")
    return 0


if __name__ == "__main__":
    exit(main())

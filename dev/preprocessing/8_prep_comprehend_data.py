"""
Converts training, validation, and test data into CSV and TXT files for AWS Comprehend.
Comprehend NER can only do a maximum of 25 classes, and does not handle overlaps, so we split into 
three separate classifiers (see Config), and resolve overlaps.
"""

import json
import pandas as pd

from common.json_helpers import load_json, save_json
from common.logging import setup_logger
from common.paths import PROCESSED, TRAIN_DATA, VAL_DATA, TEST_DATA
from common.data_utils import resolve_overlaps_longest_span

class Config:
    logger = setup_logger("preprocessing.format_comprehend", "preprocess.format_comprehend.log")

    train_data_path = TRAIN_DATA
    val_data_path = VAL_DATA
    test_data_path = TEST_DATA
    
    # Define both JSONL patterns and TXT output paths
    train_anno_output = "train_data_comprehend_{partition}.csv"
    val_anno_output = "val_data_comprehend_{partition}.csv"
    train_docs_output = PROCESSED / "train_data_comprehend.txt"
    val_docs_output = PROCESSED / "val_data_comprehend.txt"
    val_docs_output = PROCESSED / "val_data_comprehend.txt"
    test_docs_output = PROCESSED / "test_data_comprehend.txt"

    # Split labels evenly
    # Note: When setting up in AWS Comprehend, these must be uppercase snake case.
    MODEL_A_LABELS = {
        # overlap categories
        "safety_risk_antisocial_behaviour", 
        "cautions_verbal_abuse_or_threat_of", 
        "cautions_asbo_or_injunction_obtained",

        # other behavior/enforcement labels
        "cautions_physical_abuse_or_threat_of",
        "cautions_unclean_unsafe_living_environment",
        "life_events_life_events",
        "life_events_temporary",
        "safety_risk_firerelated_risks",
        "safety_risk_gas_capped",
        "property_level_property_adapted",
        "property_level_disrepair_damp_mould",
        "property_level_infestation",
        "housing_conditions_utilities",
        "housing_conditions_hoarding",
    }

    MODEL_B_LABELS = {
        # overlap categories
        "safety_risk_domestic_abuse",
        "safety_risk_risk_of_exploitation", 
        "health_medical_condition", 
        "health_mental_health",

        # other health/safeguarding labels
        "care_care_setting",
        "care_has_caring_responsibility",
        "care_social_care_involvement",
        "reasonable_adjustments_communication_needs",
        "communication_digital_exclusion",
        "communication_fluency_in_english",
        "disability_requires_adapted_property",
        "disability_sensory",
        "health_substance_misuse",
        "health_breathing_respiratory_problems",
        "health_care_setting",
        "health_cognitive_impairment",
        "health_neurodiversity_learning_disability",
        "health_terminally_ill",
        "mobility_mobility_physical",
    }

    # Just person references
    MODEL_C_LABELS = {
        "person_role",
        "person_name",
    }

def export_records_file(records):
    '''Flatten the text for ONE_DOC_PER_LINE (remove any newlines)'''
    return [
        r.get("text", "").replace('\n', ' ')
        for r in records
    ]

def export_clean_comprehend_dataset(records, target_labels, output_csv, docs_filename, logger):
    """
    Filters the dataset for the target model group and writes to a compliant csv annotations file (using pandas)
    """
    raw_texts = []
    annotations_data = [] # List to hold all our row dictionaries
    
    for idx, record in enumerate(records):
        raw_entities = [] + record.get("needs", []) + record.get("persons", [])
        
        pool = []
        for ent in raw_entities:
            if ent.get("label") in target_labels:
                pool.append(ent)
                
        clean_entities = resolve_overlaps_longest_span(pool)
        
        # Append a separate row for every entity in this record
        for ent in clean_entities:
            annotations_data.append({
                "File": docs_filename,
                "Line": idx,
                "Begin Offset": ent["start"],
                "End Offset": ent["end"],
                "Type": ent["label"].upper() # Uppercase for comprehend
            })

    # Write Annotations CSV using pandas
    logger.info("Saving annotations to %s...", output_csv)
    df = pd.DataFrame(annotations_data, columns=["File", "Line", "Begin Offset", "End Offset", "Type"])
    df.to_csv(output_csv, index=False, encoding='utf-8')

def main():
    config = Config()
    config.logger.info('Starting AWS Comprehend Data Split...')

    raw_train = load_json(config.train_data_path, config.logger)
    raw_val = load_json(config.val_data_path, config.logger)
    raw_test = load_json(config.test_data_path, config.logger)

    # --- Training Data ---
    # Model A
    export_clean_comprehend_dataset(
        records=raw_train, 
        target_labels=config.MODEL_A_LABELS, 
        output_csv=PROCESSED / config.train_anno_output.format(partition='a'),
        docs_filename=config.train_docs_output,
        logger=config.logger
    )
    # Model B
    export_clean_comprehend_dataset(
        records=raw_train, 
        target_labels=config.MODEL_B_LABELS, 
        output_csv=PROCESSED / config.train_anno_output.format(partition='b'),
        docs_filename=config.train_docs_output,
        logger=config.logger
    )
    # Model c
    export_clean_comprehend_dataset(
        records=raw_train, 
        target_labels=config.MODEL_C_LABELS, 
        output_csv=PROCESSED / config.train_anno_output.format(partition='c'),
        docs_filename=config.train_docs_output,
        logger=config.logger
    )
    
    # Write Train Documents TXT once
    train_docs = export_records_file(raw_train)
    config.logger.info("Saving documents to %s...", config.train_docs_output)
    with open(config.train_docs_output, 'w', encoding='utf-8') as f_txt:
        for text in train_docs:
            f_txt.write(text + '\n')

    # --- Validation Data ---
    # Model A
    export_clean_comprehend_dataset(
        records=raw_val, 
        target_labels=config.MODEL_A_LABELS, 
        output_csv=PROCESSED / config.val_anno_output.format(partition='a'),
        docs_filename=config.val_docs_output,
        logger=config.logger
    )
    # Model B
    export_clean_comprehend_dataset(
        records=raw_val, 
        target_labels=config.MODEL_B_LABELS, 
        output_csv=PROCESSED / config.val_anno_output.format(partition='b'),
        docs_filename=config.val_docs_output,
        logger=config.logger
    )
    # Model C
    export_clean_comprehend_dataset(
        records=raw_val, 
        target_labels=config.MODEL_C_LABELS, 
        output_csv=PROCESSED / config.val_anno_output.format(partition='c'),
        docs_filename=config.val_docs_output,
        logger=config.logger
    )

    # Write Val Documents TXT once
    val_docs = export_records_file(raw_val)
    config.logger.info("Saving documents to %s...", config.val_docs_output)
    with open(config.val_docs_output, 'w', encoding='utf-8') as f_txt:
        for text in val_docs:
            f_txt.write(text + '\n')

    # --- Test Data ---
    # Write Test Documents TXT once
    test_docs = export_records_file(raw_test)
    config.logger.info("Saving documents to %s...", config.test_docs_output)
    with open(config.test_docs_output, 'w', encoding='utf-8') as f_txt:
        for text in test_docs:
            f_txt.write(text + '\n')
    
    config.logger.info('Data Split Completed.')
    config.logger.info('LABELS: \nA: %s \nB: %s \nC: %s', [l.upper() for l in config.MODEL_A_LABELS], [l.upper() for l in config.MODEL_B_LABELS], [l.upper() for l in config.MODEL_C_LABELS])

if __name__ == '__main__':
    main()

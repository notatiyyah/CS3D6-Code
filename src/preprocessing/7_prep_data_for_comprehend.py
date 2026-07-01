import json
import pandas as pd

from common.json_helpers import load_json, save_json
from common.logging import setup_logger
from common.paths import PROCESSED
from common.data_utils import resolve_overlaps_longest_span

class Config:
    LOGGER = setup_logger("preprocessing.format_comprehend", "preprocess.format_comprehend.log")

    TRAIN_DATA_PATH = PROCESSED / "train_data.json"
    VAL_DATA_PATH = PROCESSED / "val_data.json"
    
    # Define both JSONL and TXT output patterns
    TRAIN_ANNO_OUTPUT = "train_data_comprehend_{partition}.csv"
    VAL_ANNO_OUTPUT = "val_data_comprehend_{partition}.csv"
    TRAIN_DOCS_OUTPUT = PROCESSED / "train_data_comprehend.txt"
    VAL_DOCS_OUTPUT = PROCESSED / "val_data_comprehend.txt"

    # Split labels evenly (excluding person ref)
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

def export_clean_comprehend_dataset(records, target_labels, output_csv, docs_filename, logger):
    """
    Filters the dataset for the target model group and writes to a compliant csv annotations file (using pandas)
    """
    raw_texts = []
    annotations_data = [] # List to hold all our row dictionaries
    
    for idx, record in enumerate(records):
        # 1. Flatten the text for ONE_DOC_PER_LINE (remove any newlines)
        text = record.get("text", "").replace('\n', ' ')
        raw_texts.append(text)
        
        raw_entities = record.get("needs", []) + record.get("persons", [])
        
        pool = []
        for ent in raw_entities:
            if ent["label"] == "person_ref":
                # Skip
                continue
            if ent["label"] in target_labels:
                pool.append(ent)
                
        clean_entities = resolve_overlaps_longest_span(pool)
        
        # 2. Append a separate row for EVERY entity found in this line
        # This solves the "multiple entities per document" requirement
        for ent in clean_entities:
            annotations_data.append({
                "File": docs_filename,
                "Line": idx,
                "Begin Offset": ent["start"],
                "End Offset": ent["end"],
                "Type": ent["label"]
            })

    # 3. Write Annotations CSV using pandas
    logger.info("Saving annotations to %s...", output_csv)
    df = pd.DataFrame(annotations_data, columns=["File", "Line", "Begin Offset", "End Offset", "Type"])
    # index=False ensures pandas doesn't write an extra row-number column that AWS would choke on
    df.to_csv(output_csv, index=False, encoding='utf-8')

    return raw_texts


def main():
    config = Config()
    config.LOGGER.info('Starting AWS Comprehend Data Split...')

    raw_train = load_json(config.TRAIN_DATA_PATH, config.LOGGER)
    raw_val = load_json(config.VAL_DATA_PATH, config.LOGGER)

    # --- Training Data ---
    # Model A
    raw_texts = export_clean_comprehend_dataset(
        records=raw_train, 
        target_labels=config.MODEL_A_LABELS, 
        output_csv=PROCESSED / config.TRAIN_ANNO_OUTPUT.format(partition='a'),
        docs_filename=config.TRAIN_DOCS_OUTPUT.name,
        logger=config.LOGGER
    )
    # Model B
    export_clean_comprehend_dataset(
        records=raw_train, 
        target_labels=config.MODEL_B_LABELS, 
        output_csv=PROCESSED / config.TRAIN_ANNO_OUTPUT.format(partition='b'),
        docs_filename=config.TRAIN_DOCS_OUTPUT.name,
        logger=config.LOGGER
    )
    
    # Write Train Documents TXT once
    config.LOGGER.info("Saving documents to %s...", config.TRAIN_DOCS_OUTPUT)
    with open(config.TRAIN_DOCS_OUTPUT, 'w', encoding='utf-8') as f_txt:
        for text in raw_texts:
            f_txt.write(text + '\n')

    # --- Validation Data ---
    # Model A
    raw_texts = export_clean_comprehend_dataset(
        records=raw_val, 
        target_labels=config.MODEL_A_LABELS, 
        output_csv=PROCESSED / config.VAL_ANNO_OUTPUT.format(partition='a'),
        docs_filename=config.VAL_DOCS_OUTPUT.name,
        logger=config.LOGGER
    )
    # Model B
    export_clean_comprehend_dataset(
        records=raw_val, 
        target_labels=config.MODEL_B_LABELS, 
        output_csv=PROCESSED / config.VAL_ANNO_OUTPUT.format(partition='b'),
        docs_filename=config.VAL_DOCS_OUTPUT.name,
        logger=config.LOGGER
    )

    # Write Val Documents TXT once
    config.LOGGER.info("Saving documents to %s...", config.VAL_DOCS_OUTPUT)
    with open(config.VAL_DOCS_OUTPUT, 'w', encoding='utf-8') as f_txt:
        for text in raw_texts:
            f_txt.write(text + '\n')
    
    config.LOGGER.info('Data Split Completed.')
    config.LOGGER.info('LABELS: \nA: %s \nB: %s', [l.upper() for l in config.MODEL_A_LABELS], [l.upper() for l in config.MODEL_B_LABELS])

if __name__ == '__main__':
    main()

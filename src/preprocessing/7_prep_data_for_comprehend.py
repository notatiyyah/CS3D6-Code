import json

from common.json_helpers import load_json, save_json
from common.logging import setup_logger
from common.paths import PROCESSED

class Config:
    LOGGER = setup_logger("preprocessing.format_comprehend", "preprocess.format_comprehend.log")

    TRAIN_DATA_PATH = PROCESSED / "train_data.json"
    VAL_DATA_PATH = PROCESSED / "val_data.json"
    TRAIN_OUTPUT = "train_data_comprehend_{partition}.jsonl"
    VAL_OUTPUT = "val_data_comprehend_{partition}.jsonl"

    # Split labels evenly (excluding person ref)
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

def resolve_overlaps_longest_span(entities):
    """
    Sorts entities to prioritize the longest span. 
    Drops any nested/shorter entities that intersect with an already-accepted span.
    """
    if not entities:
        return []
        
    # Sort by start index ascending, then by span length DESCENDING.
    # This guarantees the longest span is evaluated and locked in first.
    entities.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    
    resolved_entities = []
    
    for current_ent in entities:
        is_overlapping = False
        c_start, c_end = current_ent["start"], current_ent["end"]
        
        # Check against spans we've already accepted into the clean list
        for accepted_ent in resolved_entities:
            a_start, a_end = accepted_ent["BeginOffset"], accepted_ent["EndOffset"]
            
            # Mathematical condition for span intersection
            if max(c_start, a_start) < min(c_end, a_end):
                is_overlapping = True
                break
                
        # If it doesn't collide with a longer span, keep it
        if not is_overlapping:
            resolved_entities.append({
                "BeginOffset": c_start,
                "EndOffset": c_end,
                "Type": current_ent["label"].upper()  # Comprehend strictly requires uppercase types
            })
            
    return resolved_entities

def export_clean_comprehend_dataset(records, target_labels, output_filepath, logger):
    """
    Filters the dataset for the target model group and writes a compliant .jsonl file.
    """
    comprehend_lines = []
    dropped_person_refs = 0
    
    for idx, record in enumerate(records):
        text = record.get("text", "")
        
        # Gather all raw entities
        raw_entities = record.get("needs", []) + record.get("persons", [])
        
        # Filter for this specific model's labels AND strip out person_ref
        pool = []
        for ent in raw_entities:
            if ent["label"] == "person_ref":
                dropped_person_refs += 1
                continue
            if ent["label"] in target_labels:
                pool.append(ent)
                
        # Apply the Longest Span tie-breaker
        clean_entities = resolve_overlaps_longest_span(pool)
        
        # Format to Comprehend's strict single-line JSON standard
        comprehend_line = {
            "File": f"doc_{idx}.txt",
            "Line": idx,
            "Text": text,
            "Entities": clean_entities
        }
        comprehend_lines.append(comprehend_line)

    # Write JSONL
    logger.info("Saving data to %s...", output_filepath)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for line in comprehend_lines:
            f.write(json.dumps(line) + '\n')

def main():
    config = Config()
    config.LOGGER.info('Starting AWS Comprehend Data Split...')

    # Load data
    raw_train = load_json(config.TRAIN_DATA_PATH, config.LOGGER)
    raw_val = load_json(config.VAL_DATA_PATH, config.LOGGER)

    # Training Data
    export_clean_comprehend_dataset(
        records=raw_train, 
        target_labels=config.MODEL_A_LABELS, 
        output_filepath=PROCESSED / config.TRAIN_OUTPUT.format(partition='a'),
        logger=config.LOGGER
    )
    export_clean_comprehend_dataset(
        records=raw_train, 
        target_labels=config.MODEL_B_LABELS, 
        output_filepath=PROCESSED / config.TRAIN_OUTPUT.format(partition='b'),
        logger=config.LOGGER
    )

    # Validation Data
    export_clean_comprehend_dataset(
        records=raw_val, 
        target_labels=config.MODEL_A_LABELS, 
        output_filepath=PROCESSED / config.VAL_OUTPUT.format(partition='a'),
        logger=config.LOGGER
    )
    export_clean_comprehend_dataset(
        records=raw_val, 
        target_labels=config.MODEL_B_LABELS, 
        output_filepath=PROCESSED / config.VAL_OUTPUT.format(partition='b'),
        logger=config.LOGGER
    )
    config.LOGGER.info('Data Split Completed.')

if __name__ == '__main__':
    main()

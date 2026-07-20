"""
Generate Fake Dataset

Creates synthetic case notes to replace real AWS/Athena data.

Used to demonstrate iterative stratified sampling in preprocessing/2_prep_gold_standard.ipynb

Outputs:
- CSV: fake Athena note export
"""

import random
import json
from uuid import uuid4
from dataclasses import dataclass

import pandas as pd
from faker import Faker
from pandarallel import pandarallel

from common.paths import PROCESSED
from common.logging import setup_logger
from common.json_helpers import save_json


# --- CONSTANTS ---
@dataclass
class Config:
    logger           = setup_logger("preprocessing.create_fake_full_dataset", "preprocessing.create_fake_full_dataset")
    input_path: str  = PROCESSED / 'full_dataset_regex_matches.csv'
    output_path: str = PROCESSED / "fake_athena_notes.csv"
    random_seed: int = 42

    filler_person = [
        "%s contacted regarding current circumstances.",
        "%s advised of available support services.",
        "%s's circumstances reviewed.",
        "%s engagement recorded.",
        "%s provided with relevant information.",
        "%s requested further assistance.",
        "%s circumstances remain under review.",
    ]

    filler = [
        "Case reviewed during tenancy visit.",
        "Support requirements discussed.",
        "Further assessment may be required.",
        "Information recorded during contact.",
        "Notes updated following recent review.",
        "Support needs identified during conversation.",
        "Follow-up action required.",
        "Case discussed with relevant team.",
        "Information gathered during routine tenancy check.",
        "Additional support options explored.",
        "Referral considered based on current situation.",
        "Contact made with resident to discuss needs.",
        "Case notes updated following officer visit.",
        "Current risks and support needs considered.",
        "Assessment completed based on available information.",
        "Relevant information shared with support team.",
        "Ongoing monitoring recommended.",
        "Support plan reviewed and updated.",
        "Actions agreed following discussion with resident.",
        "Officer recorded concerns raised during contact.",
        "Previous support arrangements discussed.",
        "Case progression reviewed by team.",
        "Additional checks may be required.",
        "Contact history reviewed as part of case management."
    ]

    needs_templates = {
        "care_care_experienced": [
            "Tenant is a care leaver.",
            "Resident is care experienced.",
            "Was in foster care as a child.",
            "Previously looked after by social services."
        ],
        "care_care_setting": [
            "Child is in foster care.",
            "Tenant is currently in care.",
            "Resident is in a foster placement.",
            "Living in a social care placement."
        ],
        "care_has_caring_responsibility": [
            "Tenant cares for her elderly mother and two children.",
            "In receipt of carer's allowance.",
            "Provides unpaid care for a family member.",
            "Has caring responsibilities at home."
        ],
        "care_social_care_involvement": [
            "Known by adult social care.",
            "Referred to resident sustainment team.",
            "Adult social care are involved with the resident.",
            "A social worker is supporting the case."
        ],
        "cautions_asbo_or_injunction_obtained": [
            "ASBO alert.",
            "Obtained an injunction.",
            "There is a civil injunction in place.",
            "Subject to a Criminal Behaviour Order."
        ],
        "cautions_dangerous_animals": [
            "Resident has a biting dog.",
            "There is a dangerous dog at the property.",
            "Aggressive dog reported by neighbours.",
            "Resident owns a snake."
        ],
        "cautions_visiting_requirements": [
            "Warning: No lone office visits.",
            "No female officers to attend.",
            "No lone home visits required.",
            "Password scheme required before visits."
        ],
        "cautions_physical_abuse_or_threat_of": [
            "History of physical abuse.",
            "Resident reported being hit.",
            "There has been violent behaviour.",
            "Threats of physical harm reported."
        ],
        "cautions_unclean_unsafe_living_environment": [
            "Property is unclean and unsafe.",
            "There are concerns regarding hoarding and squalor.",
            "Unsafe living environment reported.",
            "Sharps found inside the property."
        ],
        "cautions_verbal_abuse_or_threat_of": [
            "Resident reported verbal abuse.",
            "Aggressive behaviour towards staff.",
            "Threatening language used.",
            "Resident feels intimidated."
        ],
        "reasonable_adjustments_communication_needs": [
            "Resident requires BSL support.",
            "Needs large print documents.",
            "Requires a translator.",
            "Communication support needed."
        ],
        "communication_digital_exclusion": [
            "Resident has no access to the internet.",
            "No wifi available at home.",
            "Digitally excluded.",
            "Does not have access to a computer or smartphone."
        ],
        "communication_fluency_in_english": [
            "Resident requires an interpreter.",
            "English is not their first language.",
            "Limited English spoken.",
            "Language barrier identified."
        ],
        "disability_requires_adapted_property": [
            "Property requires adaptations.",
            "Wet room installation required.",
            "Stairlift needed.",
            "Level access required."
        ],
        "disability_sensory": [
            "Resident has hearing impairment.",
            "Uses hearing aids.",
            "Resident is visually impaired.",
            "Requires support due to sensory impairment."
        ],
        "health_substance_misuse": [
            "Concerns around alcohol misuse.",
            "History of drug use.",
            "Resident struggles with addiction.",
            "Support required around substance misuse."
        ],
        "health_breathing_respiratory_problems": [
            "Resident has asthma.",
            "Uses an inhaler.",
            "COPD affects daily living.",
            "Experiences breathing difficulties."
        ],
        "health_care_setting": [
            "Resident is currently in hospital.",
            "Staying in a care home.",
            "Requires hospice support.",
            "Recovering after hospital admission."
        ],
        "health_cognitive_impairment": [
            "Resident has dementia.",
            "Memory loss affecting daily life.",
            "Cognitive impairment identified.",
            "Mild cognitive impairment recorded."
        ],
        "reasonable_adjustments_mental_capacity": [
            "Has an advocate supporting them.",
            "Power of attorney in place.",
            "Capacity assessment completed.",
            "Court appointed advocate involved."
        ],
        "health_neurodiversity_learning_disability": [
            "Resident has autism.",
            "ADHD diagnosis recorded.",
            "Learning disability identified.",
            "Dyslexia affects communication."
        ],
        "health_medical_condition": [
            "Resident has a long-term health condition.",
            "Chronic illness affects daily living.",
            "Requires ongoing medical support.",
            "Has diabetes and requires treatment."
        ],
        "health_mental_health": [
            "Resident experiences anxiety.",
            "Depression affecting wellbeing.",
            "Mental health support required.",
            "History of self harm."
        ],
        "health_terminally_ill": [
            "Resident is terminally ill.",
            "Receiving end of life care.",
            "Palliative care involved."
        ],
        "health_medical_life_sustaining": [
            "Uses oxygen concentrator.",
            "Requires dialysis.",
            "Uses medical equipment at home.",
            "Requires ventilator support."
        ],
        "housing_conditions_utilities": [
            "No gas supply.",
            "Electricity disconnected.",
            "No water available."
        ],
        "housing_conditions_hoarding": [
            "Property has significant clutter.",
            "Hoarding concerns identified.",
            "Many items stored throughout the home."
        ],
        "life_events_social_isolation": [
            "Resident feels lonely.",
            "Socially isolated.",
            "Has no social contact."
        ],
        "life_events_life_events": [
            "Recently released from prison.",
            "Resident is a veteran.",
            "Previously in custody.",
            "Former member of the armed forces."
        ],
        "life_events_temporary": [
            "Recent bereavement.",
            "Family member passed away.",
            "Resident is pregnant.",
            "Recent pregnancy loss."
        ],
        "life_events_identity": [
            "Resident identifies as transgender.",
            "Non-binary resident."
        ],
        "mobility_mobility_physical": [
            "Uses a wheelchair.",
            "History of falls.",
            "Requires walking frame.",
            "Limited mobility."
        ],
        "mobility_service_need": [
            "Unable to answer the door.",
            "Needs extra time to answer.",
            "Cannot access the meter.",
            "Personal emergency evacuation plan required."
        ],
        "property_level_property_adapted": [
            "Property has a stairlift.",
            "Major adaptations completed.",
            "Wet room installed."
        ],
        "property_level_disrepair_damp_mould": [
            "Damp and mould reported.",
            "Property requires repairs.",
            "Leaks causing damage.",
            "Home is in disrepair."
        ],
        "property_level_infestation": [
            "Rat infestation reported.",
            "Bed bugs found.",
            "Cockroach problem.",
            "Pest control required."
        ],
        "safety_risk_antisocial_behaviour": [
            "Anti-social behaviour reported.",
            "Neighbour harassment concerns.",
            "Resident experiencing nuisance behaviour."
        ],
        "safety_risk_domestic_abuse": [
            "Domestic abuse concerns reported.",
            "Resident feels unsafe with partner.",
            "Coercive control suspected."
        ],
        "safety_risk_firerelated_risks": [
            "Fire risk identified.",
            "No smoke alarm present.",
            "Fire hazards in property."
        ],
        "safety_risk_gas_capped": [
            "Gas supply has been capped.",
            "Gas capped due to access issues."
        ],
        "safety_risk_risk_of_exploitation": [
            "Concern about financial exploitation.",
            "Risk of modern slavery.",
            "Possible cuckooing concerns."
        ],
        "safety_risk_gang_activity_serious_youth_violence": [
            "Gang association reported.",
            "Targeted by gang members.",
            "Serious youth violence concerns."
        ]
    }

    def __post_init__(self):
        random.seed(self.random_seed)
        fake = Faker('en')
        Faker.seed(self.random_seed)

        # 50 fake names and 100 of each other sort of reference
        fake_names = [fake.name() for _ in range(50)]
        refs = ["tenant", "tnt", "l/h", "resident", "client", "cnt", "occupant", "no.4", "the gentleman", "he", "she"]
        self.person_references = refs*100 + fake_names


def generate_note(config, row):
    '''Takes a row from df with binary columns for regex matches and generates a fake note containing those categories. 
    Note: Will likely not make sense, so this data is not used for anything other than demonstrating our sampling technique.'''
    # Words / phrases picked up by regex
    matched_cats = row[row].index.to_list()
    need_spans = [random.sample(config.needs_templates.get(c, [""]), 1)[0] for c in matched_cats]

    # create phrases containing residents
    residents = random.sample(config.person_references, random.randint(1,4))
    resident_sentences = [random.sample(config.filler_person, 1)[0] % r for r in residents]

    filler = random.sample(config.filler, random.randint(1,4))

    # combine all the sentence lists into one list & shuffle
    combined_sentences = need_spans + resident_sentences + filler
    random.shuffle(combined_sentences)
    
    # join into new note
    note_content = " ".join(combined_sentences)

    return pd.Series({
        "note_id": uuid4(), # new ID for confidentiality
        "note_content": note_content,
        "note_category": row.note_category,
        "note_date": row.note_date
    })

def main():
    pandarallel.initialize(progress_bar=True)
    config = Config()

    config.logger.info("Reading regex matches from %s...", config.input_path)
    df = pd.read_csv(config.input_path)

    config.logger.info("Generating %s fake notes...", len(df))
    gen_notes = df.parallel_apply(lambda x: generate_note(config, x), axis=1)

    config.logger.info("Saving generated notes out to %s...", config.output_path)
    gen_notes.to_csv(config.output_path, index=False)


if __name__ == "__main__":
    main()
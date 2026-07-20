"""
Retroactively adds pseudonymised household members data to gold standard data from AWS Athena.
(Should have been done in sampling notebook.)
"""

import json
import difflib
import string
from dataclasses import dataclass
from datetime import datetime

import awswrangler as wr
import boto3
import pandas as pd

from common.paths import PROCESSED
from common.json_helpers import load_json, save_json
from common.logging import setup_logger

@dataclass
class Config:
    aws_region = "eu-west-2"
    aws_profile = "data-platform-housing-prod"
    database_name = "housing-refined-zone"
    table_name = "additional_needs_notes_reshaped"

    logger = setup_logger("preprocessing.pseudonymise_hms", "preprocess.pseudonymise_hms.log")
    gold_standard_file = PROCESSED / "gold_standard.json"
    output_file = PROCESSED / "gold_standard_with_households.json"


def flatten_unique_people(series, logger):
        """
        Flattens and deduplicates people (household members) by their UUID Id.
        If a note has multiple tenancies associated with it (through joins), extract all household memebers into one flat list.
        """
        unique_people = {}
        
        for raw_members in series.dropna():
            members_list = json.loads(raw_members) if isinstance(raw_members, str) else raw_members
            for member in members_list:
                person_id = member.get('id')
                if not person_id:
                    logger.warning("No ID for member %s", member)
                # If we haven't seen this person yet, add them to our dictionary
                if person_id not in unique_people:
                    unique_people[person_id] = member
        return list(unique_people.values())

def get_athena_data(config, note_ids, database):
    """Fetch original notes and household members from Athena."""
    session = boto3.Session(region_name=config.aws_region, profile_name=config.aws_profile)

    # Format IDs for the SQL IN clause
    formatted_ids = ",".join([f"'{nid}'" for nid in note_ids])
    
    query = f"""
        SELECT
            note_id, 
            tenure_id,
            note_content AS original_text,
            household_members
        FROM "{config.database_name}".{config.table_name}
        WHERE note_id IN ({formatted_ids})
    """
    
    config.logger.info("Querying Athena %s.%s...", config.database_name, config.table_name)
    df =  wr.athena.read_sql_query(query, database=config.database_name, boto3_session=session, ctas_approach=False)
    config.logger.info("Athena returned %s/%s records.", len(df), len(note_ids))

    # Reduce the fan-out from multiple tenancies per note to one record per note (will need to do this in prod too)
    config.logger.info("Squishing SQL fan-out and extracting unique people per note...")
    df = df.groupby('note_id').agg({
        'original_text': 'first',
        'tenure_id': lambda x: list(x), 
        # Collect all unique person IDs into a list
        'household_members': lambda x: flatten_unique_people(x, config.logger)
    })
    
    config.logger.info("Reduced to %s/%s records.", len(df), len(note_ids))
    
    return df.to_dict('index')
    
def extract_real_to_fake_map(real_text, fake_text):
    """Diffs the texts and creates a {Original Word: Replacement Word} mapping."""
    mapping = {}
        
    real_words = real_text.split()
    fake_words = fake_text.split()
    
    matcher = difflib.SequenceMatcher(None, real_words, fake_words)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            real_slice = real_words[i1:i2]
            fake_slice = fake_words[j1:j2]
            
            # If it's a 1-to-1 word replacement (e.g., ["Atiyyah"] -> ["Bob"])
            if len(real_slice) == len(fake_slice):
                for r_word, f_word in zip(real_slice, fake_slice):
                    r_clean = r_word.strip(string.punctuation)
                    f_clean = f_word.strip(string.punctuation)
                    if r_clean and f_clean:
                        mapping[r_clean] = f_clean
            else:
                # If it's a multi-word phrase mismatch, map the whole chunk
                r_clean = " ".join(real_slice).strip(string.punctuation)
                f_clean = " ".join(fake_slice).strip(string.punctuation)
                if r_clean and f_clean:
                    mapping[r_clean] = f_clean
                    
    return mapping

def pseudonymize_full_name(real_name, translation_map):
    """Replaces parts of the real name with the fake name using the map."""
    if not real_name:
        return real_name
        
    # First check if the entire exact name was mapped as a phrase
    if real_name in translation_map:
        return translation_map[real_name]
        
    # Otherwise, translate word-by-word (e.g., Atiyyah Jones -> Bob Jones)
    safe_words = []
    for word in real_name.split():
        clean_word = word.strip(string.punctuation)
        # Fallback to the original word if it wasn't redacted in the note
        fake_word = translation_map.get(clean_word, clean_word) 
        safe_words.append(fake_word)
        
    return " ".join(safe_words)

def replace_date_of_birth(real_dob, logger):
    '''Replaces date of birth with a new dob with correct year.'''
    if not real_dob:
        return real_dob

    # Convert to datetime if it's a string
    if isinstance(real_dob, str):
        try:
            dob = datetime.fromisoformat(real_dob.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Invalid date format for DOB: %s", real_dob)
            return real_dob  # Return unchanged if parsing fails
    else:
        dob = real_dob

    return dob.replace(day=1, month=1)  # Replace with 01/01/YYYY

def main():
    config = Config()
    
    # Load the local, redacted dataset
    records = load_json(config.gold_standard_file, config.logger)
    note_ids = [r['id'] for r in records if 'id' in r]
    
    # Fetch the unredacted data from Athena
    athena_dict = get_athena_data(config, note_ids, database="my_database")
    
    config.logger.info("Pseudonymizing household members...")
    
    # Process records
    for record in records:
        note_id = record.get('id')
        if note_id not in athena_dict:
            config.logger.warning("Note ID %s not found in Athena.", note_id)
            continue
            
        db_row = athena_dict[note_id]
        real_text = db_row.get('original_text', '')
        fake_text = record.get('text', '')
        
        # Map the real words to the fake words
        real_to_fake_map = extract_real_to_fake_map(real_text, fake_text)
        
        # Attach tenure_ids
        record['tenure_ids'] = db_row.get('tenure_id', [])
        
        # Safely parse and replace the household members list
        safe_members = []
        for member in db_row.get('household_members'):
            safe_member = member.copy()
            
            # Replace the real name with the fake name
            real_name = safe_member.get('fullName', '')
            safe_member['fullName'] = pseudonymize_full_name(real_name, real_to_fake_map)

            # Replace date of birth with approximate age
            real_dob = safe_member.get('dateOfBirth', '')
            safe_member['dateOfBirth'] = replace_date_of_birth(real_dob, config.logger)
            
            safe_members.append(safe_member)
                
        # Attach the pseudonymised list to the record
        record['household_members'] = safe_members

    # Save
    save_json(config.output_file, records, config.logger)
        
    config.logger.info(f"Success! Enriched {len(records)} records with pseudonymized household data.")

if __name__ == "__main__":
    main()
"""
Construct Taxonomy
Maps the hierarchical 'Additional Needs' categories from a downloaded CSV file into snake case labels & 
adds regexes for each category. Exports to a new CSV.
"""

import re
import pandas as pd
from common.paths import RAW, PROCESSED
from common.logging import setup_logger

# --- CONSTANTS ---
class Config:
    LOGGER = setup_logger("preprocessing.construct_taxonomy", "construct_taxonomy.log")
    INPUT_PATH = RAW / "taxonomy" / "Additional Needs Taxonomy.csv"
    OUTPUT_PATH = PROCESSED / "taxonomy_autogen_v1.csv"
    # Map category labels to regexes (keys need to exactly match cat_label)
    CATEGORY_REGEX_MAP = {
        # --- Care ---
        "care_care_experienced": r"\bcare experienced\b|\bcare leaver\b|\baged out of care\b|\blooked after\b|\blooked-after\b|\bfoster\w*\b",
        "care_care_setting": r"\bfostered\b|\bfoster care\b|\bfoster placement\b|\bin care\b|\bcare placement\b|\bsocial care placement\b",
        "care_has_caring_responsibility": r"\bformal carer\b|\bregistered carer\b|\bcarer.s allowance\b|\bunpaid carer\b|\bcaring for\b|\blooking after\b|\binformal carer\b|\bfamily carer\b",
        "care_social_care_involvement": r"\bsocial care\b|\bcare package\b|\bcare plan\b|\bcare coordinator\b|\bchildren.s social care\b|\bCSC\b|\bASC\b",
        
        # --- Cautions ---
        "cautions_asbo_or_injunction_obtained": r"\bASBO\b|\binjunction\b|\bcivil injunction\b|\bCriminal Behaviour Order\b|\bCBO\b",
        "cautions_dangerous_animals": r"\bdangerous dog\b|\bdog attack\b|\bdog bite\b|\baggressive dog\b|\bdangerous (?:pet|animal)\b|\breptile\b|\bsnake\b|\bstaff(?:fordshire)?\b",
        "cautions_visiting_requirements": r"\bno (?:lone|female|male) (?:visit|staff)\b|\bno lone (?:home|office) visit\b|\bpassword scheme\b|\btwo.person visit\b|\bdouble.crewed\b",
        "cautions_physical_abuse_or_threat_of": r"\bphysical abuse\b|\bthreat of physical abuse\b|\bviolent behaviour\b|\baggressive behaviour\b",
        "cautions_unclean_unsafe_living_environment": r"\bunclean\b|\bunsanitary\b|\bsharps\b|\bbiohazard\b|\bhoarder\b|\bfilthy\b|\bsqualor\b|\bunsafe (?:property|home|environment)\b",
        "cautions_verbal_abuse_or_threat_of": r"\bverbal abuse\b|\bverbally abusive\b|\bthreat\w*\b|\babuse (?:language|behaviour)\b|\bintimidati\w*\b",

        # --- Reasonable Adjustments ---
        "reasonable_adjustments_communication_needs": r"\bBSL\b|\bbraille\b|\bsign language\b|\bhearing loop\b|\binduction loop\b|\blarge print\b|\baudio format\b|\bscreen reader\b|\bcommunication support\b|\bspeech impairment\b|\baugmentative\b|\balternative communication\b|\bAAC\b|\beasy read\b|\btranslator\b",
        "reasonable_adjustments_mental_capacity": r"\badvocate\b|\bIMCA\b|\bdeputy\b|\bcourt of protection\b|\bLPA\b|\bpower of attorney\b|\bnext of kin\b|\bCFAT\b|\bclient financial affairs\b|\bcapacity assessment\b",
        
        # --- Communications ---
        "communication_digital_exclusion": r"\bno internet\b|\bno wifi\b|\bdigitally excluded\b|\bno (?:phone|device|computer|laptop|tablet)\b|\bdigital exclusion\b",
        "communication_fluency_in_english": r"\blow literacy\b|\blanguage barrier\b|\binterpreter\b|\btranslation\b|\bno English\b|\blimited English\b|\bESOL\b",

        # --- Disability ---
        "disability_requires_adapted_property": r"\bstairlift\b|\bwet room\b|\blevel access\b|\bminor adaptation\w*\b|\bmajor adaptation\w*\b",
        "disability_sensory": r"\bdeaf\b|\bBSL\b|\bhearing (?:loss|impairment|aid)\b|\bhard of hearing\b|\bblind\w*\b|\bpartially sighted\b|\bvisual impairment\b|\bdeafblind\w*\b|\bspeech impairment\b|\bsense of smell\b|\banosmia\b",
        
        # --- Health ---
        "health_substance_misuse": r"\balcohol\w*\b|\bdrinking\b|\bdrunk\w*\b|\bbooze\b|\balcoholic\b|\bdrug\w*\b|\bcocaine\b|\bheroin\b|\bcrack\b|\bcannabis\b|\bmethadone\b|\baddic\w*\b",
        "health_breathing_respiratory_problems": r"\basthma\b|\binhaler\b|\bCOPD\b|\bchronic obstructive\b|\ballerg\w*\b|\bbreathless\w*\b",
        "health_care_setting": r"\bhospice\b|\bnursing home\b|\bcare home\b|\bresidential home\b|\bin hospital\b|\bhospitalised\b|\badmitted to hospital\b|\binpatient\b|\bstaying with (?:family|relatives|friends)\b|\bliving with (?:family|relatives|friends)\b",
        "health_cognitive_impairment": r"\bdementia\b|\bAlzheimer\w*\b|\bmemory loss\b|\bcognitive impairment\b|\bmild cognitive impairment\b|\bMCI\b|\bdevelopmental condition\b",
        "health_neurodiversity_learning_disability": r"\bautis\w*\b|\bASD\b|\bAsperger\w*\b|\bADHD\b|\battention deficit\b|\bdyslexia\b|\blearning disabilit\w*\b",
        "health_medical_condition": r"\bfrail\w*\b|\bchronic (?:illness|condition|pain)\b|\bweakened immune\b|\bimmunocompromised\b",
        "health_mental_health": r"\bsuicidal\b|\bself.harm\b|\bdepress\w*\b|\banxiet\w*\b|\bpanic attack\b|\bPTSD\b|\bmental health\b|\blow mood\b",
        "health_terminally_ill": r"\bterminal\w*\b|\bend of life\b|\bpalliative\b|\bhospice\b",
        "health_medical_life_sustaining": r"\bnebuliser\b|\bventilator\b|\bdialysis\b|\boxygen concentrator\b|\bfeeding pump\b|\bapnoea monitor\b|\bmedical equipment\b|\blife.sustaining\b|\bmedical device\b",
        
        # --- Housing Conditions ---
        "housing_conditions_utilities": r"\bno gas\b|\bno electric\w*\b|\bno water\b|\bgas (?:cut off|disconnected)\b|\bpower (?:cut|off)\b",
        "housing_conditions_hoarding": r"\bclutter\w*\b|\bhoarding\b|\bhoarder\b",
        
        # --- Life Events ---
        "life_events_social_isolation": r"\blonely\b|\bloneliness\b|\bsocially isolated\b|\bisolat\w*\b|\bno social contact\b",
        "life_events_life_events": r"\breleased from prison\b|\bon (?:licence|probation)\b|\bprobation\b|\bdischarged from hospital\b|\bleft (?:refuge|supported housing)\b|\bcurrently in prison\b|\bin (?:custody|jail|prison)\b|\bremanded\b|\bHMP\b|\bveteran\b|\bex.service\b|\barmed forces\b|\bformer (?:soldier|military|army|navy|RAF)\b",
        "life_events_temporary": r"\bbereavement\b|\bbereaved\b|\bgrief\b|\bgrieving\b|\bpassed away\b|\bdeath of\b|\bpregnant\b|\bpregnancy\b|\bstillbirth\b|\bmiscarriage\b|\bpostnatal\b",
        "life_events_identity": r"\btransgender\b|\btrans (?:woman|man|person)\b|\bnon.binary\b|\bgender dysphoria\b",
        
        # --- Mobility ---
        "mobility_mobility_physical": r"\bwheelchair\b|\bphysical disabilit\w*\b|\bphysically disabled\b|\bamputee\b|\bparalys\w*\b|\bcrutch\w*\b|\bbroken (?:leg|ankle|hip|arm)\b|\bpost.op\b|\brecovering from (?:surgery|operation)\b|\brollator\b|\bhoist\b|\bcared for in bed\b|\bhistory of falls\b|\bfrail\w*\b|\bZimmer\b|\bwalking (?:stick|frame|aid)\b",
        "mobility_service_need": r"\bunable to answer (?:door|intercom)\b|\ballow (?:extra|more) time\b|\bprepayment meter\b|\bPEEP\b|\bpersonal emergency evacuation\b|\bcannot access meter\b",

        # --- Property Level ---
        "property_level_property_adapted": r"\bstairlift\b|\bwet room\b|\blevel access\b|\bminor adaptation\w*\b",
        "property_level_disrepair_damp_mould": r"\bdisrepair\b|\bunfit\b|\bunhabitable\b|\brepair\w*\b|\bbroken\b|\bdamaged\b|\bleaking\b|\bdamp\b|\bmould\b|\bmold\b|\bcondensation\b|\bleak\w*\b|\bblack mould\b",
        "property_level_infestation": r"\brat\w*\b|\bmice\b|\bmouse\b|\bvermin\b|\bpest\w*\b|\bbed bug\w*\b|\bcockroach\w*\b|\bflea\w*\b|\bdaddy long.?legs\b|\bsilverfish\b",
        
        # --- Safety & Risk --
        "safety_risk_antisocial_behaviour": r"\bASB\b|\banti-social\b|\bharassment\b|\bhate crime\b|\bnuisance\b",
        "safety_risk_domestic_abuse": r"\bdomestic abuse\b|\bdomestic violence\b|\bDV\b|\bcoercive control\b|\bIDVA\b|\bDA\b",
        "safety_risk_firerelated_risks": r"\barson\b|\bfire hazard\b|\bfire risk\b|\bfire safety\b|\bno smoke alarm\b|\bcandles?\b",
        "safety_risk_gas_capped": r"\bgas (?:capped|cap)\b",
        "safety_risk_risk_of_exploitation": r"\bcuckooing\b|\bfinancial (?:exploitation|abuse)\b|\bsexual exploitation\b|\bcriminal exploitation\b|\bmodern slavery\b|\btraffick\w*\b|\bradicalisa\w*\b|\bextremis\w*\b",
        "safety_risk_gang_activity_serious_youth_violence": r"\bgang (?:member|affiliated|involvement)\b|\btargeted by gang\b|\bgang (?:intimidation|threats)\b",
    }

def convert_to_label(row: pd.core.series.Series) -> str:
    '''Strips out whitespace and special characters to make a category label (format: high_level_label_category_description)'''
    raw_label = f"{row.high_level_category}_{row.category_description}".lower()
    clean_label = re.sub(r'[\s]+', '_', raw_label)
    clean_label = re.sub(r'[^a-z0-9_]', '', clean_label)
    clean_label = re.sub(r'_+', '_', clean_label)
    return clean_label

def main():
    config = Config()
    config.LOGGER.info("Starting taxonomy generation...")
    
    # 1. Load and clean original CSV
    df = pd.read_csv(config.INPUT_PATH, usecols=['Person Attribute Category', 'Description (attribute, 40 chars)', 'Value (of data type)'])
    df.dropna(inplace=True)

    df = df.rename(columns={
        'Person Attribute Category': 'high_level_category',
        'Description (attribute, 40 chars)': 'category_description',
        'Value (of data type)': 'values_hint'
    })
    df['values_hint'] = df['values_hint'].str.split("\n")

    # 2. Convert labels to snake_case
    df['cat_label'] = df.apply(convert_to_label, axis=1)

    # 3. Join with regexes and export
    regex_df = pd.DataFrame(list(config.CATEGORY_REGEX_MAP.items()), columns=['cat_label', 'regex'])
    df = df.merge(regex_df, on='cat_label', how='inner')
    if len(regex_df) != len(df):
        raise Exception("Mismatch between autogenerated labels and category regex definition. Please check.")

    config.LOGGER.info("Saving generated taxonomy to %s...", config.OUTPUT_PATH)
    df.to_csv(config.OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
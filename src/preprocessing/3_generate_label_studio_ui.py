"""
Generates Label Studio XML from Additional Needs Taxonomy
Creates an XML frontend layout from the template, dynamically populating the need labels.
"""

import ast
from xml.sax.saxutils import escape
import pandas as pd
from common.paths import ANNOTATIONS, PROCESSED
from common.logging import setup_logger

# --- CONSTANTS ---
class Config:
    LOGGER = setup_logger("preprocessing.generate_label_studio_ui", "generate_label_studio_ui.log")
    INPUT_TEMPLATE_PATH = ANNOTATIONS / "label-studio-ui-template.xml"
    INPUT_TAXONOMY_PATH =  PROCESSED / "taxonomy_autogen_v2.csv"
    OUTPUT_XML_PATH = PROCESSED / "label-studio-ui.xml"

    # High-level colour groups (matches high_level_category from csv)
    GROUP_COLOURS = {
        'Care':                  '#58D68D',  # Bright Green
        'Cautions':              '#CD6155',  # Red
        'Reasonable Adjustments':'#7787EF',  # Dark Blue
        'Communications':        '#F4D03F',  # Bright Yellow
        'Disability':            '#D09DF6',  # Lilac
        'Health':                '#48C9C0',  # Turquoise
        'Housing Conditions':    '#F086F0',  # Pink
        'Life Events':           '#EB984E',  # Soft Orange
        'Mobility':              '#7FB3D5',  # Steel Blue
        'Property Level':        '#B5B9C2',  # Grey
        'Safety & Risk':         '#EC7063',  # Soft Red
    }
    DEFAULT_COLOUR = '#95a5a6'

    TEMPLATE_STRING_NEEDS = "      <AN_PLACEHOLDER>"
    TEMPLATE_STRING_RELATION = "      <RELATION_PLACEHOLDER>"

def main():
    Config.LOGGER.info("Starting label studio UI generation...")

    # 1. Load Template
    Config.LOGGER.info("Loading template from %s...", Config.INPUT_TEMPLATE_PATH)
    with open(Config.INPUT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template_text = f.read()

    # 2. Load Taxonomy
    Config.LOGGER.info("Loading taxonomy from %s...", Config.INPUT_TAXONOMY_PATH)
    taxonomy = pd.read_csv(Config.INPUT_TAXONOMY_PATH)
    category_labels_xml = []

    # 3. Generate HTML Label Tags for each category (grouped by high level category)
    for high_level, group in taxonomy.groupby('high_level_category'):
        category_labels_xml.append('')  # Visual break between groups
        colour = Config.GROUP_COLOURS.get(str(high_level), Config.DEFAULT_COLOUR)
        
        for row in group.itertuples():
            # Parse stringified arrays (pandas doesn't do nested lists well)
            hints_list = ast.literal_eval(str(row.values_hint))
            hint = ", ".join(hints_list) if isinstance(hints_list, list) else str(row.values_hint)

            escaped_desc = escape(str(row.category_description))
            escaped_hint = escape(hint)
            
            category_labels_xml.append(
                f'          <Label value="{row.cat_label}" html="{escaped_desc}" background="{colour}" hint="{escaped_hint}"/>'
            )

    # 4. Construct Relation Block
    labels_joined = ",".join(taxonomy['cat_label'].astype(str).tolist())
    relation_block = (
        f'      <Relation \n'
        f'        value="AFFECTS" \n'
        f'        fromName="need_labels" \n'
        f'        toName="entity_labels" \n'
        f'        label="{labels_joined}"\n'
        f'      />'
    )

    # 5. Inject Dynamic Elements and Save New File
    labels_string = "\n".join(category_labels_xml)
    output_text = template_text.replace(Config.TEMPLATE_STRING_NEEDS, labels_string)
    output_text = output_text.replace(Config.TEMPLATE_STRING_RELATION, relation_block)

    Config.LOGGER.info("Saving XML UI to %s...", Config.OUTPUT_XML_PATH)
    with open(Config.OUTPUT_XML_PATH, "w", encoding="utf-8") as f:
        f.write(output_text)

if __name__ == "__main__":
    main()
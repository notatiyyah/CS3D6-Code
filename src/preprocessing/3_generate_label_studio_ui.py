"""
Generates Label Studio XML from Additional Needs Taxonomy
Creates an XML frontend layout from the template, dynamically populating the need labels.
"""

import ast
from xml.sax.saxutils import escape
import pandas as pd

# --- CONSTANTS ---
INPUT_TEMPLATE_PATH = "../annotation/label-studio-ui-template.xml"
INPUT_TAXONOMY_PATH = "../data/output/taxonomy_autogen_v2.csv"
OUTPUT_XML_PATH = "../annotation/label-studio-ui.xml"

# 1. High-level colour groups (matches high_level_category from csv)
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

# 2. Load Template
with open(INPUT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    template_text = f.read()

taxonomy = pd.read_csv(INPUT_TAXONOMY_PATH)
category_labels_xml = []

# 3. Generate HTML Label Tags for each category (grouped by high level category)
for high_level, group in taxonomy.groupby('high_level_category'):
    category_labels_xml.append('')  # Visual break between groups
    colour = GROUP_COLOURS.get(str(high_level), DEFAULT_COLOUR)
    
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
output_text = template_text.replace("      <AN_PLACEHOLDER>", labels_string)
output_text = output_text.replace("      <RELATION_PLACEHOLDER>", relation_block)

with open(OUTPUT_XML_PATH, "w", encoding="utf-8") as f:
    f.write(output_text)
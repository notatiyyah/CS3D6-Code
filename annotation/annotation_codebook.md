# Annotation Codebook for Additional Needs in MTFH Case Notes.

## Table of Contents

* [Overview](#overview)
* [Entity Types](#entity-types)
* [Additional Needs Types](#additional-needs-types)
    * [<mark style="background-color: #58D68D;">Care</mark>](#care)
    * [<mark style="background-color: #CD6155;">Cautions</mark>](#cautions)
    * [<mark style="background-color: #7787EF;">Reasonable Adjustments</mark>](#reasonable-adjustments)
    * [<mark style="background-color: #F4D03F;">Communication</mark>](#communication)
    * [<mark style="background-color: #D09DF6;">Disability</mark>](#disability)
    * [<mark style="background-color: #48C9C0;">Health</mark>](#health)
    * [<mark style="background-color: #F086F0;">Housing Conditions</mark>](#housing-conditions)
    * [<mark style="background-color: #EB984E;">Life Events</mark>](#life-events)
    * [<mark style="background-color: #7FB3D5;">Mobility</mark>](#mobility)
    * [<mark style="background-color: #B5B9C2;">Property Level</mark>](#property-level)
    * [<mark style="background-color: #EC7063;">Safety & Risk</mark>](#safety--risk)
* [Edge Cases](#edge-cases)

---

## Overview

### What annotators do

For each housing case note, annotators produce three things:

- **Additional Needs Spans**: Exact text spans / substrings that idenify a vulnerability or risk, labelled with the closest AN category. A single span may carry multiple labels if it implies multiple vulnerabilities.
- **Entity Spans**: Text substrings that identify a person (by name, role or pronoun), so vulnerabilities can be attributed correctly.
- **Links**: Directed relationships *from a vulnerability span* *to the entity span* it belongs to. Leave a vulnerability unlinked when the responsible person is unidentifiable from context. `Property Level` ANs will *never* be linked to an entity.

![image](annotation/label-studio-ui-screenshot.png)

---

## Entity Types

### `Person_Name`
A person's name or pseudonym/

**Examples:** `"Eva Smith", "Jane Doe", "John Doe"`

### `Person_Role`

A descriptive role, relationship, used to refer to a person.

**Examples:** `"tenant", "his mother", "the children", "support worker", "her teenage son", "flat no. 6"`

### `Person_Pronoun`

A pronoun that is the primary reference to a person where no nearby role span exists to link to.

**When to use:** Tag "she" in "she has asthma" only if no role (e.g. "tenant") or name has been established nearby. Do **not** tag every pronoun. Only use when the pronoun is doing the attributional work.

*Note: Tag entities even if they have no ANs of their own. E.g. a mentioned family member.*

---

## Additional Needs Types

### <mark style="background-color: #58D68D;">Care</mark>

| Label | Value hints | Notes |
|---|---|---|
| `care_care_experienced` | Care leaver · Care experienced (under 25) | References to leaving care, being a looked-after child, or having been in the care system |
| `care_care_setting` | Fostered · Social care | Use `health_care_setting` for current medical/residential placements. This label is for background life context only (was fostered, grew up in social care). |
| `care_has_caring_responsibility` | Formal · Informal | Providing personal care, cooking/cleaning, daily check-ins, unpaid caring. Link to the carer, not the person being cared for. e.g. "visits daily to help with cooking and personal care" → link to the visitor. |
| `care_social_care_involvement` | Adult Social Care · Children's Social Care | Care packages, social worker involvement, referrals to ASC/CSC, care assessments |

---

### <mark style="background-color: #CD6155;">Cautions</mark>

| Label | Value hints | Notes |
|---|---|---|
| `cautions_asbo_or_injunction_obtained` | ASBO · Injunction | |
| `cautions_dangerous_animals` | Dog · Reptile · Other | Aggressive/uncontrolled pets, exotic animals, animals posing a risk to staff visiting the property |
| `cautions_physical_abuse_or_threat_of` | Physical Abuse (actual) · Physical Abuse (threat) | For incidents not framed as a DA relationship. If DA context is clear, prefer `safety_risk_domestic_abuse` and apply both. |
| `cautions_unclean_unsafe_living_environment` | Unclean and/or unsanitary · Unsafe (e.g. undisposed sharps) | Property state posing health/safety risk to staff or occupants. Undisposed needles/sharps are a classic trigger. |
| `cautions_verbal_abuse_or_threat_of` | Verbal Abuse (actual) · Verbal Abuse (threat) | "Shouting heard from the flat" is ambiguous — leave unlinked if the responsible person cannot be identified from context. |

---
### <mark style="background-color: #7787EF;">Reasonable Adjustments</mark>

| Label | Value hints | Notes |
|---|---|---|
| `reasonable_adjustments_communication_needs` | BSL · Hearing Loop · Braille · Large Print · Easy Read · Translator | |
| `reasonable_adjustments_mental_capacity` | Has advocate · Court appointed advocate · Local Authority advocate · Client Financial Affairs Team · Informal arrangement | Power of attorney, advocate involved, family acting on behalf of person lacking capacity |

### <mark style="background-color: #F4D03F;">Communication</mark>

| Label | Value hints | Notes |
|---|---|---|
| `communication_digital_exclusion` | No internet access · No digital device · Can't use digital communication | No smartphone, no computer, inability to use online portals or email |
| `communication_fluency_in_english` | A Little English · Appropriate for Age · Does not Wish to Reply · Good (Both Written and Spoken) · None · Sign Language or Other | Needs interpreter, communicates via family member, preferred language noted |

---

### <mark style="background-color: #D09DF6;">Disability</mark>

| Label | Value hints | Notes |
|---|---|---|
| `disability_requires_adapted_property` | Minor adaptations · Major adaptation | Wet room, level-access shower, ramp, grab rails, stairlift requested/needed. Use `property_level_property_adapted` for adaptations already in place. |
| `disability_sensory` | Deaf · Hearing impairment · Visually impaired · Partially sighted · Deafblindness · Speech impairment · Poor sense of smell/taste | |

---

### <mark style="background-color: #48C9C0;">Health</mark>

| Label | Value hints | Notes |
|---|---|---|
| `health_substance_misuse` | Alcohol · Drug | Alcohol dependency, drug use, addiction, references to substances affecting tenancy |
| `health_breathing_respiratory_problems` | Asthma · Allergies · COPD | |
| `health_care_setting` | Hospice (respite) · Hospice (longer-term) · Hospital · Care home · Staying with family · Staying with friends · Post-hospital recovery at home | "Staying with his sister" counts if tenant is displaced due to health/vulnerability. "Discharged from hospital" implies a prior hospital setting — annotate the setting referenced, not just the discharge event. |
| `health_cognitive_impairment` | Dementia · Mild Cognitive Impairment · Developmental Condition | |
| `health_neurodiversity_learning_disability` | Autism Spectrum Disorder · ADHD · Dyslexia | |
| `health_medical_condition` | Frailty · Weakened immune system · Chronic illness | Chronic conditions not covered by a more specific label (e.g. diabetes, kidney disease, cancer if not terminal) |
| `health_mental_health` | Anxiety · Depression · OCD · Other mental health condition | "Mental health crisis", "mental health breakdown", named conditions. "Struggling" or "not coping" alone is too vague — include only if mental health is reasonably implied by context. |
| `health_terminally_ill` | Terminally ill | End-of-life care, palliative care, terminal diagnosis. Also consider `health_care_setting` (hospice). |
| `health_medical_life_sustaining` | Nebuliser · Heart/lung/ventilator · Dialysis · Oxygen concentrator · Water dependent · Other medical equipment | Any equipment the person depends on for survival — especially relevant to utilities failures |

---

### <mark style="background-color: #F086F0;">Housing Conditions</mark>

| Label | Value hints | Notes |
|---|---|---|
| `housing_conditions_utilities` | No gas · No electric · No water | Especially serious if `health_medical_life_sustaining` also applies — apply both and note the combination. |
| `housing_conditions_hoarding` | Clutter Image Rating 1–3 · 4–6 · 7–9 | Explicit hoarding references or significant clutter (e.g. "rooms inaccessible", "floor-to-ceiling items").|

---

### <mark style="background-color: #EB984E;">Life Events</mark>

| Label | Value hints | Notes |
|---|---|---|
| `life_events_social_isolation` | Social Isolation | Not leaving the house, no social contacts, housebound due to anxiety or condition, withdrawn from community |
| `life_events_life_events` | Ex service personnel · In prison · Left prison | Narrowly-defined set — do not use for general life disruption. Use `life_events_temporary` for bereavement/pregnancy. |
| `life_events_temporary` | Bereavement · Pregnancy | |

---

### <mark style="background-color: #7FB3D5;">Mobility</mark>

| Label | Value hints | Notes |
|---|---|---|
| `mobility_mobility_physical` | Frail · History of falls · Zimmer frame · Walking stick · Rollator · Cared for in bed (entirely/partially) · Stairlift · Hoist · Wheelchair user | "Limited mobility", "difficulty walking", recent falls, hip/knee surgery. The exact aid does not need to be named — if mobility limitation is clear, apply the label. |
| `mobility_service_need` | Unable to answer door · Allow time to answer door · Prepayment meter — can't access · PEEP | Staff notes to allow extra time, buzzer/intercom issues due to disability, evacuation plan needed |

---

### <mark style="background-color: #B5B9C2;">Property Level</mark>

| Label | Value hints | Notes |
|---|---|---|
| `property_level_property_adapted` | Minor adaptations · Major adaptation | For adaptations already installed. Use `disability_requires_adapted_property` for adaptations needed but not yet in place. |
| `property_level_disrepair_damp_mould` | Disrepair · Damp · Mould · Leaks | |
| `property_level_infestation` | Rats/mice · Daddy long-legs · Silver fish · Cockroaches · Bed bugs · Fleas | |

---

### <mark style="background-color: #EC7063;">Safety & Risk</mark>

| Label | Value hints | Notes |
|---|---|---|
| safety_risk_antisocial_behaviour | Perpetrator of harassment · Victim of harassment · Perpetrator of hate crime · Victim of hate crime | Link to the person the span is describing. If it's unclear who that is, leave unlinked. |
| `safety_risk_domestic_abuse` | Alleged Perpetrator · Abusive person · Survivor | DA disclosures, referrals to MARAC/IDVA, protective orders, historical DA. Link to the correct party for each person mentioned. |
| `safety_risk_firerelated_risks` | Arson · Fire hazards in the house | Hoarding + fire risk combination, unsafe cooking practices, candles/smoking near combustibles |
| `safety_risk_gas_capped` | Resident choice · Financial issues · Access issues | |
| `safety_risk_risk_of_exploitation` | Financial · Sexual · Criminal · Cuckooing · Radicalisation · Modern slavery | Unknown persons in the property, tenant appears coerced, financial exploitation by family/associates, county lines indicators |

---

## Edge Cases

| If | Then |
| --- | --- |
| Note says "needs a wet room" (tenant wants it but doesn't have it) | `disability_requires_adapted_property` |
| Note says "property has a wet room already fitted" | `property_level_property_adapted` |
| Note says "staying with family" — is this care setting or just a location? | `health_care_setting` if the person is displaced from their property due to health/vulnerability. Omit if it is incidental context only. |
| Physical altercation in a domestic context | Apply both `safety_risk_domestic_abuse` and `cautions_physical_abuse_or_threat_of` if both are clearly implied. |
| Note says "Client is fleeing her husband's domestic abuse" | Link `safety_risk_domestic_abuse` to the client entity. If a span describes the husband's behaviour directly, link that to the husband entity. |
| Note mentions a condition of a household member who is not the tenant | Tag the member as a `Person_Role` entity. Link the vulnerability to that entity, not to "Tenant". |
| "Mental health crisis" — which label? | `health_mental_health`. If hospitalisation is mentioned, also `health_care_setting` for the hospitalisation span. |
| Carer mentioned but no vulnerability described for them | Still tag as `Person_Role`. Apply `care_has_caring_responsibility` to the caring activity span and link it to the carer entity. |

> **⚠️ When in doubt between two labels** Apply both. We would prefer over-tagging over under-tagging.

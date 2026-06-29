-- Reshapes MMH Notes (which can be targeted at a person, tenure or asset) to be targeted only to the relevant tenure based on the note's timestamp and the tenure's start/end date.

-- 1. Enhance tenure and notes tables - extract ids, parse dates, etc.
WITH
tenure_enhanced AS (
    SELECT
        t.id                                                                AS tenure_id,
        json_extract_scalar(t.tenuredAsset, '$[0].id')                      AS asset_id,
        TRY(CAST(from_iso8601_timestamp(t.startOfTenureDate) AS TIMESTAMP)) AS tenure_startdate,
        TRY(CAST(from_iso8601_timestamp(t.endOfTenureDate)   AS TIMESTAMP)) AS tenure_enddate,
        t.householdmembers                                                  AS household_members
    FROM "housing-raw-zone"."mtfh_tenureinformation" t
),

notes_enhanced AS (
    SELECT
        id,
        targetId,
        targetType,
        categorisation.category                                           AS note_category,
        TRY(CAST(from_iso8601_timestamp(createdAt) AS TIMESTAMP))         AS note_date,
        TRY(CAST(from_iso8601_timestamp(createdAt) AS TIMESTAMP)) IS NULL AS flag_note_date_parse_failed, -- add flag column instead of silently dropping
        TRIM(CONCAT(
            COALESCE(title, ''), ' ',
            COALESCE(description, '') 
        ))                                                                AS note_content -- concat title & desc (w/ null handling)
    FROM "housing-raw-zone"."mtfh_notes"
),

-- 2. Explode household members (so that each person in a household has a row), flag inactive tenants (people without any active tenancies)
tenure_exploded AS (
    SELECT 
        t.*, 
        member.id AS person_id,
        member.type AS person_type
    FROM tenure_enhanced t
    CROSS JOIN UNNEST(
        CAST(json_parse(t.household_members) AS ARRAY(ROW(id VARCHAR, type VARCHAR)))
    ) AS hm (member) -- https://docs.aws.amazon.com/athena/latest/ug/flattening-arrays.html
),

active_system_humans AS (
    SELECT DISTINCT person_id
    FROM tenure_exploded
    WHERE CURRENT_DATE >= CAST(tenure_startdate AS DATE)
      AND (tenure_enddate IS NULL OR CURRENT_DATE <= CAST(tenure_enddate AS DATE))
),

tenure_inactive_flag AS (
    SELECT 
        t.tenure_id,
        NOT COALESCE(bool_or(a.person_id IS NOT NULL), FALSE) AS flag_inactive -- NOT ANY person on the tenure is active
    FROM tenure_exploded t
    LEFT JOIN active_system_humans a ON t.person_id = a.person_id
    GROUP BY t.tenure_id
),

organisations AS (
    SELECT DISTINCT person_id
    FROM tenure_exploded
    WHERE person_type = 'organisation'
),

-- 3. Join notes on each ID based on targetType, combine back into one
all_notes AS (
    -- Tenure-targeted notes: join on tenure id
    SELECT n.*, t.tenure_id, t.household_members,
           (t.tenure_id IS NULL) AS flag_no_matching_target,
           FALSE AS flag_is_organisation
    FROM notes_enhanced n
    LEFT JOIN tenure_enhanced t ON n.targetId = t.tenure_id
    WHERE n.targetType = 'tenure'

    UNION ALL

    -- Asset-targeted notes: join on asset id & whether note date aligns with tenure start/end dates
    SELECT n.*, t.tenure_id, t.household_members,
           (t.tenure_id IS NULL) AS flag_no_matching_target,
           FALSE AS flag_is_organisation
    FROM notes_enhanced n
    LEFT JOIN tenure_enhanced t 
        ON  n.targetId = t.asset_id
        AND n.note_date >= t.tenure_startdate
        AND (t.tenure_enddate IS NULL OR n.note_date <= t.tenure_enddate)
    WHERE n.targetType = 'asset'

    UNION ALL

    -- Person-targeted, humans only: join on person id & whether note date aligns with tenure start/end dates
    SELECT n.*, t.tenure_id, t.household_members,
        (t.tenure_id IS NULL) AS flag_no_matching_target,
        FALSE AS flag_is_organisation
    FROM notes_enhanced n
    LEFT JOIN tenure_exploded t 
        ON  n.targetId = t.person_id
        AND t.person_type <> 'organisation'
        AND n.note_date >= t.tenure_startdate
        AND (t.tenure_enddate IS NULL OR n.note_date <= t.tenure_enddate)
    WHERE n.targetType = 'person'

    UNION ALL

    -- Person-targeted, organisations: don't join with tenures. Orgs can have many concurrent tenures, so could cause massive row duplication.
    SELECT n.*, NULL, NULL,
        FALSE AS flag_no_matching_target,
        TRUE AS flag_is_organisation
    FROM notes_enhanced n
    WHERE n.targetType = 'person'
      AND n.targetId IN (SELECT person_id FROM organisations)
),

-- 4. Attach inactive flag and import dates
final AS (
    SELECT
        -- data columns
        n.id AS note_id,
        n.targetId AS note_target_id,
        n.targetType AS note_target_type,
        
        n.note_content,
        n.note_date,
        n.note_category,

        n.tenure_id,
        n.household_members,

        -- flags
        n.flag_no_matching_target,
        n.flag_note_date_parse_failed,
        n.flag_is_organisation,
        COALESCE(tif.flag_inactive, FALSE) AS flag_inactive,
        
        -- import dates
        CAST(YEAR(CURRENT_DATE)  AS VARCHAR) AS import_year,
        CAST(MONTH(CURRENT_DATE) AS VARCHAR) AS import_month,
        CAST(DAY(CURRENT_DATE)   AS VARCHAR) AS import_day,
        CAST(CURRENT_DATE        AS VARCHAR) AS import_date
    FROM all_notes n
    LEFT JOIN tenure_inactive_flag tif ON n.tenure_id = tif.tenure_id
)

SELECT * FROM final
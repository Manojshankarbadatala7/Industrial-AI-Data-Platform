-- ============================================================
-- SCANIA AI FAILURE PREDICTION
-- INDUSTRIAL AI DATA PLATFORM
-- SNOWFLAKE ML PIPELINE
-- ============================================================


-- ============================================================
-- 1. DATABASE AND SCHEMA SETUP
-- ============================================================

CREATE DATABASE IF NOT EXISTS SCANIA_AI_PLATFORM;

CREATE SCHEMA IF NOT EXISTS SCANIA_AI_PLATFORM.RAW;
CREATE SCHEMA IF NOT EXISTS SCANIA_AI_PLATFORM.STAGING;
CREATE SCHEMA IF NOT EXISTS SCANIA_AI_PLATFORM.ML_FEATURES;

USE DATABASE SCANIA_AI_PLATFORM;
USE WAREHOUSE COMPUTE_WH;


-- ============================================================
-- 2. STAGING TABLE
-- ============================================================

USE SCHEMA STAGING;

CREATE OR REPLACE TABLE APS_FAILURE_STAGING AS
SELECT *
FROM SCANIA_AI_PLATFORM.RAW.APS_FAILURE_RAW;


-- ============================================================
-- 3. CLEAN NUMERIC FEATURES
--
-- Converts:
--     "na" -> NULL
--     numeric strings -> NUMBER
--
-- ============================================================

DECLARE
    SQL_CMD STRING;
BEGIN

    SELECT
        'CREATE OR REPLACE TABLE ' ||
        'SCANIA_AI_PLATFORM.STAGING.APS_FAILURE_CLEAN AS ' ||
        'SELECT CLASS, ' ||
        LISTAGG(
            'TRY_TO_NUMBER(NULLIF(TRIM(TO_VARCHAR("' ||
            COLUMN_NAME ||
            '")), ''na'')) AS "' ||
            COLUMN_NAME ||
            '"',
            ', '
        ) WITHIN GROUP (ORDER BY ORDINAL_POSITION) ||
        ' FROM SCANIA_AI_PLATFORM.STAGING.APS_FAILURE_STAGING'
    INTO :SQL_CMD
    FROM SCANIA_AI_PLATFORM.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'STAGING'
      AND TABLE_NAME = 'APS_FAILURE_STAGING'
      AND COLUMN_NAME <> 'CLASS';

    EXECUTE IMMEDIATE :SQL_CMD;

END;


-- ============================================================
-- 4. CREATE ML FEATURE TABLE
-- ============================================================

USE SCHEMA ML_FEATURES;

CREATE OR REPLACE TABLE APS_FAILURE_FEATURES AS
SELECT *
FROM SCANIA_AI_PLATFORM.STAGING.APS_FAILURE_CLEAN;


-- ============================================================
-- 5. REMOVE HIGH-MISSING FEATURES
--
-- These features were identified during preprocessing
-- as having excessive missing values.
--
-- Python preprocessing removed:
--     AB_000
--     BM_000
--     BN_000
--     BO_000
--     BP_000
--     BQ_000
--     BR_000
--     CR_000
--
-- ============================================================

CREATE OR REPLACE TABLE APS_FAILURE_FILTERED AS
SELECT
    * EXCLUDE (
        AB_000,
        BM_000,
        BN_000,
        BO_000,
        BP_000,
        BQ_000,
        BR_000,
        CR_000
    )
FROM SCANIA_AI_PLATFORM.STAGING.APS_FAILURE_CLEAN;


-- ============================================================
-- 6. HANDLE REMAINING MISSING VALUES
--
-- Remaining NULL values are replaced with 0.
--
-- ============================================================

SET SQL_CMD = (
    SELECT
        'CREATE OR REPLACE TABLE ' ||
        'SCANIA_AI_PLATFORM.ML_FEATURES.APS_FAILURE_FINAL AS ' ||
        'SELECT CLASS, ' ||
        LISTAGG(
            'COALESCE("' ||
            COLUMN_NAME ||
            '", 0) AS "' ||
            COLUMN_NAME ||
            '"',
            ', '
        ) WITHIN GROUP (ORDER BY ORDINAL_POSITION) ||
        ' FROM SCANIA_AI_PLATFORM.ML_FEATURES.APS_FAILURE_FILTERED'
    FROM SCANIA_AI_PLATFORM.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'ML_FEATURES'
      AND TABLE_NAME = 'APS_FAILURE_FILTERED'
      AND COLUMN_NAME <> 'CLASS'
);

EXECUTE IMMEDIATE $SQL_CMD;


-- ============================================================
-- 7. CREATE BALANCED DATASET
--
-- Original data:
--     59,000 non-failures
--      1,000 failures
--
-- We keep:
--      1,000 failures
--      1,000 non-failures
--
-- Total:
--      2,000 samples
--
-- ============================================================

CREATE OR REPLACE TABLE APS_FAILURE_ML_BALANCED AS

SELECT *
FROM APS_FAILURE_FINAL
WHERE CLASS = 'pos'

UNION ALL

SELECT *
FROM APS_FAILURE_FINAL
WHERE CLASS = 'neg'
QUALIFY ROW_NUMBER() OVER (
    ORDER BY RANDOM()
) <= 1000;


-- ============================================================
-- 8. SELECT FINAL TOP 20 FEATURES
--
-- These are the features used by the final XGBoost model.
--
-- ============================================================

CREATE OR REPLACE TABLE APS_FAILURE_ML_TOP20 AS

SELECT
    CLASS,
    CI_000,
    AA_000,
    AH_000,
    BB_000,
    BG_000,
    BU_000,
    BV_000,
    CQ_000,
    BT_000,
    AN_000,
    AO_000,
    BH_000,
    DN_000,
    CC_000,
    BX_000,
    AQ_000,
    AP_000,
    CK_000,
    BY_000,
    BJ_000

FROM APS_FAILURE_ML_BALANCED;


-- ============================================================
-- 9. CREATE TRAINING DATA
--
-- 800 failures
-- 800 non-failures
--
-- Total = 1,600
--
-- ============================================================

CREATE OR REPLACE TABLE APS_FAILURE_ML_TRAIN AS

SELECT *
FROM APS_FAILURE_ML_TOP20

QUALIFY ROW_NUMBER() OVER (
    PARTITION BY CLASS
    ORDER BY RANDOM()
) <= 800;


-- ============================================================
-- 10. CREATE TEST DATA
--
-- 200 failures
-- 200 non-failures
--
-- Total = 400
--
-- ============================================================

CREATE OR REPLACE TABLE APS_FAILURE_ML_TEST AS

SELECT *
FROM APS_FAILURE_ML_TOP20

QUALIFY ROW_NUMBER() OVER (
    PARTITION BY CLASS
    ORDER BY RANDOM()
) > 800;


-- ============================================================
-- 11. PREDICTION TABLE
-- ============================================================

CREATE OR REPLACE TABLE APS_FAILURE_PREDICTIONS (

    PREDICTION_ID INTEGER AUTOINCREMENT,

    PREDICTION INTEGER,

    PREDICTION_LABEL VARCHAR(20),

    FAILURE_PROBABILITY FLOAT,

    RISK_LEVEL VARCHAR(10),

    CREATED_AT TIMESTAMP_NTZ
        DEFAULT CURRENT_TIMESTAMP()

);


-- ============================================================
-- 12. MONITORING SUMMARY VIEW
-- ============================================================

CREATE OR REPLACE VIEW APS_FAILURE_MONITORING_SUMMARY AS

SELECT

    COUNT(*) AS TOTAL_PREDICTIONS,

    SUM(
        CASE
            WHEN PREDICTION = 1
                THEN 1
            ELSE 0
        END
    ) AS PREDICTED_FAILURES,

    SUM(
        CASE
            WHEN PREDICTION = 0
                THEN 1
            ELSE 0
        END
    ) AS PREDICTED_NON_FAILURES,

    ROUND(
        AVG(FAILURE_PROBABILITY),
        4
    ) AS AVG_FAILURE_PROBABILITY,

    SUM(
        CASE
            WHEN RISK_LEVEL = 'HIGH'
                THEN 1
            ELSE 0
        END
    ) AS HIGH_RISK_COUNT,

    SUM(
        CASE
            WHEN RISK_LEVEL = 'MEDIUM'
                THEN 1
            ELSE 0
        END
    ) AS MEDIUM_RISK_COUNT,

    SUM(
        CASE
            WHEN RISK_LEVEL = 'LOW'
                THEN 1
            ELSE 0
        END
    ) AS LOW_RISK_COUNT,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN RISK_LEVEL = 'HIGH'
                    THEN 1
                ELSE 0
            END
        )
        /
        NULLIF(COUNT(*), 0),
        2
    ) AS HIGH_RISK_PERCENTAGE

FROM APS_FAILURE_PREDICTIONS;


-- ============================================================
-- 13. RISK DISTRIBUTION VIEW
-- ============================================================

CREATE OR REPLACE VIEW APS_FAILURE_RISK_DISTRIBUTION AS

SELECT

    RISK_LEVEL,

    COUNT(*) AS PREDICTION_COUNT,

    ROUND(
        100.0 * COUNT(*)
        /
        NULLIF(
            SUM(COUNT(*)) OVER (),
            0
        ),
        2
    ) AS PERCENTAGE

FROM APS_FAILURE_PREDICTIONS

GROUP BY RISK_LEVEL

ORDER BY
    CASE RISK_LEVEL
        WHEN 'HIGH' THEN 1
        WHEN 'MEDIUM' THEN 2
        WHEN 'LOW' THEN 3
        ELSE 4
    END;


-- ============================================================
-- 14. HIGH-RISK PREDICTIONS VIEW
-- ============================================================

CREATE OR REPLACE VIEW APS_FAILURE_HIGH_RISK AS

SELECT

    PREDICTION_ID,

    PREDICTION_LABEL,

    FAILURE_PROBABILITY,

    RISK_LEVEL,

    CREATED_AT

FROM APS_FAILURE_PREDICTIONS

WHERE RISK_LEVEL = 'HIGH';


-- ============================================================
-- 15. MAINTENANCE PRIORITY VIEW
--
-- >= 0.95  -> CRITICAL
-- >= 0.80  -> URGENT
-- >= 0.50  -> MONITOR
-- <  0.50  -> NORMAL
--
-- ============================================================

CREATE OR REPLACE VIEW APS_FAILURE_MAINTENANCE_PRIORITY AS

SELECT

    PREDICTION_ID,

    PREDICTION_LABEL,

    FAILURE_PROBABILITY,

    RISK_LEVEL,

    CASE

        WHEN FAILURE_PROBABILITY >= 0.95
            THEN 'CRITICAL'

        WHEN FAILURE_PROBABILITY >= 0.80
            THEN 'URGENT'

        WHEN FAILURE_PROBABILITY >= 0.50
            THEN 'MONITOR'

        ELSE 'NORMAL'

    END AS MAINTENANCE_PRIORITY,

    CREATED_AT

FROM APS_FAILURE_PREDICTIONS;


-- ============================================================
-- 16. MAINTENANCE ACTIONS VIEW
-- ============================================================

CREATE OR REPLACE VIEW APS_FAILURE_MAINTENANCE_ACTIONS AS

SELECT

    PREDICTION_ID,

    PREDICTION_LABEL,

    FAILURE_PROBABILITY,

    RISK_LEVEL,

    MAINTENANCE_PRIORITY,

    CASE

        WHEN MAINTENANCE_PRIORITY = 'CRITICAL'
            THEN
                'Immediate inspection and preventive maintenance'

        WHEN MAINTENANCE_PRIORITY = 'URGENT'
            THEN
                'Schedule maintenance as soon as possible'

        WHEN MAINTENANCE_PRIORITY = 'MONITOR'
            THEN
                'Increase monitoring frequency'

        ELSE
                'Continue normal operation and routine monitoring'

    END AS RECOMMENDED_ACTION,

    CREATED_AT

FROM APS_FAILURE_MAINTENANCE_PRIORITY;


-- ============================================================
-- 17. DASHBOARD SUMMARY VIEW
-- ============================================================

CREATE OR REPLACE VIEW APS_FAILURE_DASHBOARD AS

SELECT

    COUNT(*) AS TOTAL_PREDICTIONS,

    SUM(
        CASE
            WHEN PREDICTION_LABEL = 'FAILURE'
                THEN 1
            ELSE 0
        END
    ) AS PREDICTED_FAILURES,

    SUM(
        CASE
            WHEN PREDICTION_LABEL = 'NO_FAILURE'
                THEN 1
            ELSE 0
        END
    ) AS PREDICTED_NON_FAILURES,

    ROUND(
        AVG(FAILURE_PROBABILITY),
        4
    ) AS AVG_FAILURE_PROBABILITY,

    SUM(
        CASE
            WHEN MAINTENANCE_PRIORITY = 'CRITICAL'
                THEN 1
            ELSE 0
        END
    ) AS CRITICAL_COUNT,

    SUM(
        CASE
            WHEN MAINTENANCE_PRIORITY = 'URGENT'
                THEN 1
            ELSE 0
        END
    ) AS URGENT_COUNT,

    SUM(
        CASE
            WHEN MAINTENANCE_PRIORITY = 'MONITOR'
                THEN 1
            ELSE 0
        END
    ) AS MONITOR_COUNT,

    SUM(
        CASE
            WHEN MAINTENANCE_PRIORITY = 'NORMAL'
                THEN 1
            ELSE 0
        END
    ) AS NORMAL_COUNT,

    ROUND(
        SUM(
            CASE
                WHEN MAINTENANCE_PRIORITY
                     IN ('CRITICAL', 'URGENT')
                    THEN 1
                ELSE 0
            END
        )
        * 100.0
        /
        NULLIF(COUNT(*), 0),
        2
    ) AS ACTION_REQUIRED_PERCENTAGE

FROM APS_FAILURE_MAINTENANCE_ACTIONS;


-- ============================================================
-- 18. FINAL OBJECT SUMMARY
-- ============================================================

SHOW TABLES IN SCHEMA SCANIA_AI_PLATFORM.ML_FEATURES;

SHOW VIEWS IN SCHEMA SCANIA_AI_PLATFORM.ML_FEATURES;
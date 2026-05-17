-- ======================================================================
-- DEFI CREDIT SCORING: WALLET BEHAVIOR & LIQUIDATION RISK (V2 ARCHITECTURE)
-- Description: Extracting Aave borrowers, their on-chain behavior, 
--              and assigning a default risk label using Dune Spellbook.
-- ======================================================================

WITH borrowers AS (
    -- 1. Identify wallets that have borrowed assets on Aave using standard lending tables
    SELECT DISTINCT borrower
    FROM lending.borrow
    WHERE project = 'aave' AND blockchain = 'ethereum'
    LIMIT 10000 -- Limiting to 10k wallets to ensure fast querying
),

liquidations AS (
    -- 2. Identify wallets that failed to maintain collateral and got liquidated
    -- Using the correct Dune v2 table: aave_ethereum.LendingPool_evt_LiquidationCall
    SELECT DISTINCT user AS borrower, 1 AS is_liquidated
    FROM aave_v2_ethereum.LendingPool_evt_LiquidationCall
    
    UNION
    
    -- Also include Aave V3 liquidations
    SELECT DISTINCT user AS borrower, 1 AS is_liquidated
    FROM aave_v3_ethereum.Pool_evt_LiquidationCall
),

wallet_activity AS (
    -- 3. Calculate on-chain footprint for each borrower (Feature Engineering)
    SELECT 
        "from" AS wallet_address,
        COUNT(hash) AS total_transactions,
        DATE_DIFF('day', MIN(block_time), CURRENT_TIMESTAMP) AS wallet_age_days
    FROM ethereum.transactions
    WHERE "from" IN (SELECT borrower FROM borrowers)
    GROUP BY 1
)

-- 4. Combine features and target labels for our Machine Learning model
SELECT 
    w.wallet_address,
    w.total_transactions,
    w.wallet_age_days,
    -- If the wallet exists in the liquidation table, label as 1 (Default/Bad Risk). Else 0.
    COALESCE(l.is_liquidated, 0) AS target_default
FROM wallet_activity w
LEFT JOIN liquidations l ON w.wallet_address = l.borrower
ORDER BY w.total_transactions DESC;
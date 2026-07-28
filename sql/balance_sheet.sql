-- The balance sheet, read back out of the ERP as a running balance.
--
-- The BS journal holds *movements*, not positions: one entry per quarter booking
-- the change in every account since the prior quarter end. So the balance at a
-- given date is the cumulative sum of every posted line up to and including it,
-- which is what the self-join below computes. Storing movements rather than
-- restated positions is what makes the ledger auditable — you can see what
-- changed, not just what it ended at.
--
-- The natural balance convention, and it is worth being explicit because the sign
-- confuses everyone at least once:
--   * assets     -> debit balance  -> SUM(debit - credit) is POSITIVE
--   * liabilities and equity -> credit balance -> NEGATIVE
--   * treasury stock -> contra-equity -> POSITIVE, and correctly so
--
-- ``fpa.extract.odoo_sql.reconcile_balance_sheet`` flips the credit side back to
-- presentation sign before comparing against the filed statement.
--
-- Odoo 18 schema notes: account codes are company-dependent JSONB (code_store)
-- and names are translatable JSONB. Neither is a plain column.

WITH quarter_ends AS (
    SELECT DISTINCT am.date AS period
    FROM account_move am
    WHERE am.state = 'posted'
      AND am.ref LIKE 'BS-%'
)
SELECT
    q.period,
    acc.code_store ->> am.company_id::text  AS account_code,
    acc.name ->> 'en_US'                    AS account_name,
    acc.account_type                        AS account_type,
    SUM(aml.debit - aml.credit)             AS balance
FROM quarter_ends q
JOIN account_move      am  ON am.state = 'posted'
                          AND am.ref LIKE 'BS-%'
                          AND am.date <= q.period
JOIN account_move_line aml ON aml.move_id = am.id
JOIN account_account   acc ON acc.id = aml.account_id
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2;

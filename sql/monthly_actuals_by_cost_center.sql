-- Monthly actuals by cost center, straight out of the ERP.
--
-- Reads account_analytic_line rather than account_move_line: Odoo generates one
-- analytic line per posted journal line that carries an analytic distribution,
-- already resolved to a single cost center. Going through account_move_line
-- instead would mean parsing the JSONB analytic_distribution column and
-- apportioning by percentage, which is the same answer with more ways to be wrong.
--
-- Sign: Odoo carries costs as negative analytic amounts. Negated here so the
-- extract reports expenses as positive, matching the filed income statement.
--
-- Only posted moves count. A draft entry is not in the books.
--
-- Odoo 18 schema notes, both of which bite a naive query:
--   * account_account.code is *company-dependent* and stored as JSONB keyed by
--     company id (code_store), not as a plain column.
--   * Translatable fields (account and analytic-account names) are JSONB keyed by
--     language code.

SELECT
    -- Month-end. Written as two separate INTERVAL literals rather than the
    -- compound '1 month - 1 day' form so the file runs unchanged under both
    -- psql and DuckDB's postgres scanner, which the pipeline reads it through.
    (date_trunc('month', aal.date) + INTERVAL '1 month' - INTERVAL '1 day')::date AS period,
    split_part(aa.name ->> 'en_US', ' / ', 1)                           AS function,
    split_part(aa.name ->> 'en_US', ' / ', 2)                           AS sub_center,
    acc.code_store ->> am.company_id::text                              AS account_code,
    acc.name ->> 'en_US'                                                AS account_name,
    SUM(-aal.amount)                                                    AS amount
FROM account_analytic_line aal
JOIN account_analytic_account aa  ON aa.id  = aal.account_id
JOIN account_account         acc ON acc.id = aal.general_account_id
JOIN account_move_line       aml ON aml.id = aal.move_line_id
JOIN account_move            am  ON am.id  = aml.move_id
WHERE am.state = 'posted'
  AND am.ref LIKE 'FPA-%'
GROUP BY 1, 2, 3, 4, 5
ORDER BY 1, 2, 3, 4;

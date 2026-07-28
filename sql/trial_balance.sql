-- Trial balance by account and fiscal year.
--
-- The first thing any accountant asks an ERP for, and the cheapest proof the
-- ledger is internally consistent: total debits must equal total credits for
-- every period. If this does not net to zero, nothing downstream is worth reading.
--
-- Covers both seeded journals, because a trial balance that excludes a journal is
-- not a trial balance:
--   * FPA- : the cost-center allocation overlay, offset to 990000.
--   * BS-  : the filed balance sheet, self-balancing with no offset at all.

-- Odoo 18: account codes are company-dependent JSONB (code_store), and names are
-- translatable JSONB. Neither is a plain column.

SELECT
    EXTRACT(YEAR FROM am.date)::int         AS fiscal_year,
    acc.code_store ->> am.company_id::text  AS account_code,
    acc.name ->> 'en_US'                    AS account_name,
    acc.account_type                        AS account_type,
    SUM(aml.debit)                   AS total_debit,
    SUM(aml.credit)                  AS total_credit,
    SUM(aml.debit - aml.credit)      AS balance
FROM account_move_line aml
JOIN account_move    am  ON am.id  = aml.move_id
JOIN account_account acc ON acc.id = aml.account_id
WHERE am.state = 'posted'
  AND (am.ref LIKE 'FPA-%' OR am.ref LIKE 'BS-%')
GROUP BY 1, 2, 3, 4
ORDER BY 1, 2;

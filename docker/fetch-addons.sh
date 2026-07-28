#!/usr/bin/env bash
# Fetch the OCA modules that give stock Odoo the reporting half of EPM.
#
#   mis-builder       KPI expressions over account balances, with budget and
#                     variance columns — the closest thing Odoo has to an EPM
#                     report writer.
#   account-budgeting Budgets against analytic accounts (our cost centers).
#
# Pinned to the 18.0 branch to match the Odoo image tag. Shallow clones: we want
# the modules, not the history.
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p addons

ODOO_SERIES="18.0"
REPOS=(
  "mis-builder"        # mis_builder, mis_builder_budget
  "account-budgeting"  # account_budget_oca (analytic budgets)
  # mis_builder's own dependencies, discovered from its manifest rather than
  # assumed: it depends on report_xlsx (reporting-engine) and date_range
  # (server-ux). Odoo will refuse to install without them.
  "reporting-engine"   # report_xlsx
  "server-ux"          # date_range
)

for repo in "${REPOS[@]}"; do
  target="addons/${repo}"
  if [ -d "${target}/.git" ]; then
    echo "==> ${repo}: updating ${ODOO_SERIES}"
    git -C "${target}" fetch --depth 1 origin "${ODOO_SERIES}"
    git -C "${target}" reset --hard FETCH_HEAD
  else
    echo "==> ${repo}: cloning ${ODOO_SERIES}"
    git clone --depth 1 --branch "${ODOO_SERIES}" \
      "https://github.com/OCA/${repo}.git" "${target}"
  fi
  echo "    $(find "${target}" -maxdepth 1 -type d -name '[a-z]*' | wc -l) modules"
done

echo
echo "Done. Modules are mounted read-only at /mnt/extra-addons inside the container."
echo "Bring the stack up with:  docker compose -f docker/docker-compose.yml up -d"

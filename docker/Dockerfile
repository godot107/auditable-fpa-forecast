# Odoo plus the one Python package the OCA modules need.
#
# mis_builder declares external_dependencies {'python': ['openupgradelib']}, and
# Odoo refuses to install a module whose external dependencies are missing. The
# stock image has no way to add it, so this is the minimal extension.
FROM odoo:18.0-20260723

USER root
# --break-system-packages: Debian marks the system Python as externally managed
# (PEP 668). Safe here — this image exists only to run Odoo.
RUN pip3 install --no-cache-dir --break-system-packages openupgradelib
USER odoo

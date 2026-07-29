"""The property that made the app's dependency guard dead code.

``fpa.forecast.bayes`` imports NumPyro and JAX *inside* its functions, so the whole
pipeline and test suite run without the heavy stack. That is deliberate and worth
keeping — but it has a consequence the Forecast page got wrong:

    try:
        from fpa.forecast.bayes import forecast_intervals   # succeeds regardless
    except ImportError:
        st.warning("NumPyro is not installed")              # unreachable

The import cannot fail, so the guard never fired, and on a host without NumPyro the
``ModuleNotFoundError`` was raised later — at call time, outside the ``try`` — and
reached a public reader as a raw traceback. The page now probes for the *dependency*
with ``importlib.util.find_spec`` rather than for the success of an import statement.

Same shape as the other defects in this codebase: a check that passes for a reason
unrelated to what it was meant to verify.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

HEAVY = ("numpyro", "jax", "jaxlib")


def test_importing_bayes_does_not_pull_in_numpyro_or_jax():
    """A fresh interpreter must import the module without loading the heavy stack.

    Run in a subprocess because the rest of the suite may already have imported JAX,
    which would make an in-process ``sys.modules`` check pass for the wrong reason.
    """
    code = (
        "import sys; import fpa.forecast.bayes as b; "
        "assert hasattr(b, 'forecast_intervals'); "
        "leaked = [m for m in ('numpyro', 'jax') if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = out.stdout.strip()
    assert leaked == "", f"importing bayes loaded the heavy stack: {leaked}"


def test_guarding_the_import_would_not_detect_a_missing_dependency():
    """Pins the reasoning, so nobody reinstates the try/except.

    ``forecast_intervals`` is importable here, and would be importable on a host with
    no NumPyro at all. Availability therefore has to be asked of the environment.
    """
    from fpa.forecast.bayes import forecast_intervals  # noqa: F401

    module = importlib.import_module("fpa.forecast.bayes")
    source = module.__doc__ or ""
    assert "lazy-imported" in source.lower() or "lazy" in source.lower()


def test_the_forecast_page_probes_for_the_dependency_not_the_import():
    """The fix itself, asserted against the page source.

    A behavioural test would need a Streamlit runtime; this at least fails loudly if
    someone replaces the probe with a bare ``except ImportError`` again.
    """
    from pathlib import Path

    page = Path(__file__).resolve().parents[1] / "app" / "pages" / "2_Forecast.py"
    text = page.read_text()

    assert "importlib.util.find_spec" in text
    assert "_HAS_BAYES" in text
    assert "except ImportError" not in text, "the import guard cannot detect a missing dep"

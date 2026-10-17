import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_importing_the_library_never_changes_the_matplotlib_backend():
    """Regression: src.plotting once forced the Agg backend, which silently removed inline charts from notebooks."""
    code = (
        "import matplotlib; before = matplotlib.get_backend(); "
        "import src.plotting, src.funnel, src.pipeline, src.results_table; "
        "assert matplotlib.get_backend() == before, (before, matplotlib.get_backend())"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

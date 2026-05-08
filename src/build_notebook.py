from __future__ import annotations

import json
from pathlib import Path

from config import NOTEBOOKS_DIR, ensure_directories


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    ensure_directories()
    cells = [
        markdown_cell(
            "# Hotel Booking Cancellation Analysis\n"
            "\n"
            "This notebook is generated from the reusable pipeline in `src/`. "
            "Run `python src/run_analysis.py` first to refresh all outputs."
        ),
        code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "ROOT = Path('..').resolve()\n"
            "summary = json.loads((ROOT / 'outputs' / 'analysis_summary.json').read_text(encoding='utf-8'))\n"
            "summary.keys()"
        ),
        markdown_cell("## Dataset profile"),
        code_cell("pd.Series(summary['raw_profile']).drop('missing_by_column')"),
        markdown_cell("## Cancellation distribution"),
        code_cell("pd.DataFrame(summary['target_distribution']).T"),
        markdown_cell("## Prediction metrics"),
        code_cell("pd.Series({k: summary['model'][k] for k in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']})"),
        markdown_cell("## EDA figures"),
        code_cell(
            "from IPython.display import Image, display\n"
            "for name in ['target_distribution', 'numeric_distributions', 'correlation_heatmap', 'cancellation_by_categories', 'monthly_cancellation_rate']:\n"
            "    display(Image(filename=summary['figures'][name]))"
        ),
        markdown_cell("## Dashboard"),
        code_cell("summary['dashboard']"),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOKS_DIR / "hotel_booking_analysis.ipynb"
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Notebook written: {path}")


if __name__ == "__main__":
    main()

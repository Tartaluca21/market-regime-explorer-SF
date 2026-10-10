"""IT: Grafici e file di presentazione. EN: Charts and presentation files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def build_charts(prices: pd.DataFrame, results: dict[str, Any],
                 settings: dict[str, Any]) -> dict[str, Any]:
    """IT: Creare grafici leggibili di stati, probabilità, asset e confronti.
    Mostrare date, unità e tempi del modello; restituire figure con nomi stabili.

    EN: Create readable charts of states, probabilities, assets and comparisons.
    Show dates, units and model timing; return figures with stable names.
    """
    raise NotImplementedError("Grafici da implementare / Charts not implemented.")


def export_figures(charts: dict[str, Any], output_dir: Path) -> None:
    """IT: Salvare figure PNG/PDF leggibili in results/figures/.
    Conservare date e unità; nessun training nascosto durante l'esportazione.

    EN: Save readable PNG/PDF figures in results/figures/.
    Preserve dates and units; no hidden model fitting during export.
    """
    raise NotImplementedError("Export da implementare / Figure export not implemented.")

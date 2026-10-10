"""IT: Lettura e preparazione dei dati. EN: Data loading and preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def load_data(csv_path: Path) -> pd.DataFrame:
    """IT: Leggere l'unico CSV con date e prezzi di equity, bond e gold.
    Restituire i dati sorgente senza pulizia automatica.

    EN: Read the single CSV containing dates and equity, bond and gold prices.
    Return the source data without automatic cleaning.
    """
    raise NotImplementedError("Lettura da implementare / Data loading not implemented.")


def prepare_prices(data: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """IT: Preparare prezzi equity/bond/gold con indice date unico e crescente.
    Gestire esplicitamente valori mancanti, anomalie e date.

    EN: Prepare equity/bond/gold prices with a unique, increasing date index.
    Handle missing values, anomalies and dates explicitly.
    """
    raise NotImplementedError("Preparazione da implementare / Price preparation not implemented.")

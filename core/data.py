"""Lettura e preparazione dei dati."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# Le 19 colonne attese: la data + 6 campi (OHLC, adj close, volume) per ognuno dei 3 ETF
ASSETS = ["spy", "tlt", "gld"]
CAMPI = ["open", "high", "low", "close", "adj_close", "volume"]
COLONNE_ATTESE = ["date"] + [f"{a}_{c}" for a in ASSETS for c in CAMPI]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Legge il CSV e lo restituisce così com'è, senza pulizia automatica.
    Controlla solo che ci siano tutte le colonne e che i tipi siano corretti."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File non trovato: {csv_path}")

    df = pd.read_csv(csv_path)

    # controllo dello schema: devono esserci tutte le 19 colonne
    mancanti = [c for c in COLONNE_ATTESE if c not in df.columns]
    if mancanti:
        raise ValueError(f"Colonne mancanti nel CSV: {mancanti}")

    # date come datetime, tutto il resto come numeri
    # (se un valore non è convertibile pandas dà errore invece di inventare qualcosa)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    for col in COLONNE_ATTESE[1:]:
        df[col] = pd.to_numeric(df[col])

    return df[COLONNE_ATTESE]


def prepare_prices(data: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Da implementare al passaggio 2."""
    raise NotImplementedError("Da implementare")
    EN: Prepare equity/bond/gold prices with a unique, increasing date index.
    Handle missing values, anomalies and dates explicitly.
    """
    raise NotImplementedError("Preparazione da implementare / Price preparation not implemented.")?

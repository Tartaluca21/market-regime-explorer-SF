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
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    for col in COLONNE_ATTESE[1:]:
        df[col] = pd.to_numeric(df[col])

    return df[COLONNE_ATTESE]


def prepare_prices(data: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Tiene i 3 prezzi adjusted, li rinomina equity/bond/gold e mette le date
    come indice crescente. Non riempie niente: se trova problemi dà errore."""
    # mappa colonna originale -> nome nuovo, presa dai settings
    nomi = {
        settings["assets"]["equity"]: "equity",
        settings["assets"]["bond"]: "bond",
        settings["assets"]["gold"]: "gold",
    }

    # controlli sulle date
    if data["date"].isna().any():
        raise ValueError("Ci sono date mancanti")
    if data["date"].duplicated().any():
        doppie = data.loc[data["date"].duplicated(), "date"].dt.date.tolist()
        raise ValueError(f"Date duplicate: {doppie[:5]}")

    # date come indice, solo le 3 colonne di prezzo, rinominate e ordinate
    prices = data.set_index("date")[list(nomi)].rename(columns=nomi).sort_index()

    # controlli sui prezzi: niente mancanti, niente <= 0
    if prices.isna().any().any():
        giorni = prices.index[prices.isna().any(axis=1)].date.tolist()
        raise ValueError(f"Prezzi mancanti nei giorni: {giorni[:5]}")
    if (prices <= 0).any().any():
        giorni = prices.index[(prices <= 0).any(axis=1)].date.tolist()
        raise ValueError(f"Prezzi non positivi nei giorni: {giorni[:5]}")

    return prices.astype(float)

def trova_rendimenti_estremi(prices: pd.DataFrame, soglia: float = 6.0) -> pd.DataFrame:
    """Restituisce i giorni in cui almeno un asset ha un rendimento con |z-score| > soglia.
    Serve solo per controllarli a mano: non toglie niente dai dati."""
    returns = prices.pct_change(fill_method=None)
    z = (returns - returns.mean()) / returns.std(ddof=1)
    estremi = (z.abs() > soglia).any(axis=1)

    tabella = (returns[estremi] * 100).round(2)
    tabella.columns = [f"{c} (%)" for c in tabella.columns]
    tabella["z max"] = z[estremi].abs().max(axis=1).round(1)
    return tabella

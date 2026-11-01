"""Costruzione delle feature candidate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ASSET = ["equity", "bond", "gold"]


def rendimenti_e_volatilita(prices: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Per ogni asset: rendimento semplice del giorno e volatilità annualizzata
    sugli ultimi N rendimenti (N = settings["windows"]["volatility"], ddof=1).
    Ogni valore in t usa solo prezzi fino a t."""
    finestra = settings["windows"]["volatility"]
    giorni_anno = settings["annualization_days"]

    returns = prices[ASSET].pct_change(fill_method=None)
    out = pd.DataFrame(index=prices.index)
    for a in ASSET:
        out[f"{a}_return"] = returns[a]
    for a in ASSET:
        out[f"{a}_volatility"] = (returns[a].rolling(finestra, min_periods=finestra).std(ddof=1)
                                  * np.sqrt(giorni_anno))
    return out

def momentum_e_drawdown(prices: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Per ogni asset: momentum = P_t / P_(t-60) - 1 e drawdown = P_t / max(ultimi 252 prezzi) - 1.
    Il massimo include il prezzo di oggi, quindi il drawdown è sempre <= 0."""
    w_mom = settings["windows"]["momentum"]
    w_dd = settings["windows"]["drawdown"]

    out = pd.DataFrame(index=prices.index)
    for a in ASSET:
        out[f"{a}_momentum"] = prices[a] / prices[a].shift(w_mom) - 1
    for a in ASSET:
        massimo = prices[a].rolling(w_dd, min_periods=w_dd).max()
        out[f"{a}_drawdown"] = prices[a] / massimo - 1
    return out

def correlazioni_mobili(prices: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Correlazione di Pearson sugli ultimi N rendimenti (N = settings["windows"]["correlation"])
    per le coppie equity-bond, equity-gold e bond-gold.
    Se una finestra è costante la correlazione non è definita e resta NaN: non la sostituiamo."""
    finestra = settings["windows"]["correlation"]
    returns = prices[ASSET].pct_change(fill_method=None)

    out = pd.DataFrame(index=prices.index)
    for a, b in [("equity", "bond"), ("equity", "gold"), ("bond", "gold")]:
        out[f"{a}_{b}_correlation"] = (returns[a].rolling(finestra, min_periods=finestra)
                                       .corr(returns[b]))
    return out

def build_features(prices: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Le 15 feature candidate (rendimento, volatilità, momentum, drawdown per asset
    + 3 correlazioni) dopo il warm-up comune. Ogni valore in t usa solo prezzi fino a t.
    Se dopo il warm-up resta un valore non finito, dà errore invece di eliminarlo."""
    tutte = pd.concat([
        rendimenti_e_volatilita(prices, settings),
        momentum_e_drawdown(prices, settings),
        correlazioni_mobili(prices, settings),
    ], axis=1)

    # warm-up comune: la finestra più lunga è il drawdown (252 prezzi -> 251 righe da scartare)
    w = settings["windows"]
    warmup = max(w["volatility"], w["momentum"], w["drawdown"] - 1, w["correlation"])
    if len(tutte) <= warmup:
        raise ValueError(f"Servono più di {warmup} prezzi per costruire le feature")
    features = tutte.iloc[warmup:]

    # dopo il warm-up non devono esserci NaN o infiniti
    non_finiti = ~np.isfinite(features)
    if non_finiti.any().any():
        colonne = features.columns[non_finiti.any()].tolist()
        giorni = features.index[non_finiti.any(axis=1)].date.tolist()
        raise ValueError(f"Valori non finiti dopo il warm-up in {colonne}, giorni {giorni[:5]}")

    features.index.name = "date"
    return features


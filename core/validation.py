"""IT: Valutazione e profili degli stati. EN: Evaluation and state profiles."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def evaluate_model(results: dict[str, Any], settings: dict[str, Any]) -> pd.DataFrame:
    """IT: Costruire confronti e controlli di stabilità con date e criteri espliciti.
    Usare solo lo sviluppo per cambiare scelte; distinguere replay e valutazione.

    EN: Build comparisons and stability checks with explicit dates and criteria.
    Use development data for changing choices; distinguish replay from evaluation.
    """
    raise NotImplementedError("Valutazione da implementare / Evaluation not implemented.")


def build_regime_profiles(prices: pd.DataFrame, results: dict[str, Any],
                          settings: dict[str, Any]) -> pd.DataFrame:
    """IT: Descrivere tutti gli asset per stato con conteggi, quote e unità.
    Conservare prezzi e rendimenti esclusi dal modello; i profili non sono trading.

    EN: Describe all assets by state with counts, shares and units.
    Retain prices and returns excluded from the model; profiles are not trading results.
    """
    raise NotImplementedError("Profili da implementare / State profiles not implemented.")

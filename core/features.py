"""IT: Costruzione delle feature candidate. EN: Candidate feature construction."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def build_features(prices: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """IT: Costruire 15 candidate: return/volatility/momentum/drawdown per asset
    e tre correlazioni. Usare finestre configurate e date comuni, senza futuro.

    EN: Build 15 candidates: return/volatility/momentum/drawdown per asset
    and three correlations. Use configured windows and common dates, without future data.
    """
    raise NotImplementedError("Feature da implementare / Feature construction not implemented.")

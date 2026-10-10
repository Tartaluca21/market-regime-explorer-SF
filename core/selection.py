"""IT: Selezione e preprocessing. EN: Feature selection and preprocessing."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def select_features(feature_bank: pd.DataFrame, training_end: str,
                    settings: dict[str, Any]) -> list[str]:
    """IT: Restituire la lista ordinata delle feature scelte usando solo il training.
    La data finale è inclusa; valutare ridondanza e variazione rispettando le unità.

    EN: Return the ordered feature list selected using training observations only.
    The end date is inclusive; assess redundancy and variation with attention to units.
    """
    raise NotImplementedError("Selezione da implementare / Feature selection not implemented.")


def prepare_model_data(feature_bank: pd.DataFrame, selected_features: list[str],
                       settings: dict[str, Any]) -> dict[str, Any]:
    """IT: Restituire feature_names, scaler, boundaries e scaled_blocks con date.
    Dividere dopo il warm-up; selezione e scaler restano fissi dopo il training.

    EN: Return feature_names, scaler, boundaries and dated scaled_blocks.
    Split after warm-up; selection and scaling remain fixed after training.
    """
    raise NotImplementedError("Preprocessing da implementare / Preprocessing not implemented.")

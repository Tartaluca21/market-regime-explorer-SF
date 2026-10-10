"""IT: Modelli, selezione e inferenza. EN: Models, selection and inference."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def fit_baselines(prepared: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """IT: Addestrare K-Means e GMM sul training e restituire kmeans e gmm.
    Usare le stesse feature e lo stesso scaler dell'HMM.

    EN: Fit K-Means and GMM on training data and return kmeans and gmm entries.
    Use the same features and scaler as the HMM.
    """
    raise NotImplementedError("Baseline da implementare / Baseline fitting not implemented.")


def fit_hmm_candidates(prepared: dict[str, Any], settings: dict[str, Any]) -> tuple[list[Any], pd.DataFrame]:
    """IT: Addestrare HMM Gaussiani diagonali sul training di un solo subset.
    Restituire modelli e diagnostica, rispettando il budget globale dei tentativi.

    EN: Fit diagonal Gaussian HMMs on the training block of one feature subset.
    Return models and diagnostics, respecting the global attempted-fit budget.
    """
    raise NotImplementedError("Training da implementare / HMM fitting not implemented.")


def select_hmm(candidates: list[Any], diagnostics: pd.DataFrame,
               prepared: dict[str, Any], settings: dict[str, Any]) -> Any:
    """IT: Scegliere un HMM con criteri dichiarati su training e validation.
    Non usare il test; le densità non confrontano feature o scaler diversi.

    EN: Select an HMM using declared training and validation criteria.
    Do not use test data; densities do not compare different features or scalers.
    """
    raise NotImplementedError("Scelta da implementare / HMM selection not implemented.")


def filter_probabilities(model: Any, observations: pd.DataFrame,
                         previous_posterior: Any | None = None) -> pd.DataFrame:
    """IT: Restituire probability_0..K-1 e log_predictive_density alle stesse date.
    Input scalato e ordinato; mantenere continuità, ordine degli stati e inferenza causale.
    previous_posterior è la distribuzione dopo l'osservazione precedente.

    EN: Return probability_0..K-1 and log_predictive_density on the same dates.
    Input is scaled and ordered; preserve continuity, state order and causal inference.
    previous_posterior is the distribution after the preceding observation.
    """
    raise NotImplementedError("Filtro da implementare / Forward filtering not implemented.")

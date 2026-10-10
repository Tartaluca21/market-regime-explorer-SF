"""IT: Proiezione degli stati e piccolo esperimento sul rischio.
EN: State projection and a small risk forecasting experiment.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def predict_next_states(model: Any, probabilities: pd.DataFrame) -> pd.DataFrame:
    """IT: Proiettare ogni posterior filtrato con la matrice di transizione.
    Restituire probabilità a una seduta, mantenendo date e ordine degli stati.

    EN: Project each filtered posterior through the transition matrix.
    Return one-session-ahead probabilities, preserving dates and state order.
    """
    raise NotImplementedError("Proiezione da implementare / State projection not implemented.")


def build_risk_targets(prices: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """IT: Alla data t, calcolare la volatilità dei cinque rendimenti successivi.
    Usare ddof=1 e sqrt(252); lasciare mancanti le ultime cinque etichette.
    La soglia del rischio si stima dopo lo split, solo da target di training validi.

    EN: At date t, calculate volatility over the next five returns.
    Use ddof=1 and sqrt(252); leave the final five targets missing.
    Estimate the risk threshold after splitting, from valid training targets only.
    """
    raise NotImplementedError("Target da implementare / Risk targets not implemented.")


def evaluate_risk_forecast(features: pd.DataFrame, probabilities: pd.DataFrame,
                           target: pd.DataFrame, split_dates: dict[str, Any],
                           settings: dict[str, Any]) -> pd.DataFrame:
    """IT: Confrontare prevalenza di training, logit su volatilità corrente e
    stessa logit con probabilità HMM. Scartare origini con target oltre il blocco;
    scegliere C sulla validation e valutare Brier e average precision sul test.

    EN: Compare training prevalence, a logit on current volatility and the same
    logit with HMM probabilities. Drop origins whose target crosses a block end;
    choose C on validation and evaluate Brier and average precision on test.
    """
    raise NotImplementedError("Forecast da implementare / Risk evaluation not implemented.")

"""Selezione e preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd


def dividi_date(features: pd.DataFrame, settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Divide i giorni dopo il warm-up in training / validation / test (60/20/20),
    in ordine cronologico, senza sovrapposizioni e senza mescolare.
    Restituisce per ogni blocco data di inizio, data di fine (incluse) e numero di giorni."""
    n = len(features)
    n_train = int(n * settings["split"]["train_fraction"])
    n_val = int(n * settings["split"]["validation_fraction"])

    posizioni = {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),     # il resto va al test
    }
    confini = {}
    for nome, (inizio, fine) in posizioni.items():
        if fine - inizio < 1:
            raise ValueError(f"Il blocco {nome} è vuoto: servono più dati")
        date = features.index[inizio:fine]
        confini[nome] = {
            "start": date[0].strftime("%Y-%m-%d"),
            "end": date[-1].strftime("%Y-%m-%d"),
            "n_obs": len(date),
        }
    return confini


def select_features(feature_bank: pd.DataFrame, training_end: str,
                    settings: dict[str, Any]) -> list[str]:
    """Da implementare al passaggio 3."""
    raise NotImplementedError("Da implementare")


def prepare_model_data(feature_bank: pd.DataFrame, selected_features: list[str],
                       settings: dict[str, Any]) -> dict[str, Any]:
    """Da implementare al passaggio 4."""
    raise NotImplementedError("Da implementare")
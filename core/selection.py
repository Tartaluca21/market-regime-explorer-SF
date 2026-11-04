"""Selezione e preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

def controlla_variabilita(features: pd.DataFrame, training_end: str,
                          soglia_persistenza: float = 0.5) -> pd.DataFrame:
    """Statistiche calcolate SOLO sul training (fino a training_end incluso) per decidere
    quali feature escludere. Criteri che non dipendono dalle unità di misura:
    - costante: deviazione standard (quasi) zero;
    - quota di valori ripetuti (es. drawdown = 0 sui massimi);
    - persistenza: autocorrelazione a 1 giorno. Un regime dura settimane o mesi,
      quindi una feature che cambia a caso ogni giorno (|autocorr| < soglia) è solo rumore."""
    train = features.loc[:training_end]
    report = pd.DataFrame({
        "media": train.mean(),
        "dev_std": train.std(ddof=1),
        "quota_valori_ripetuti": train.apply(lambda s: 1 - s.nunique() / len(s)),
        "autocorr_1g": train.apply(lambda s: s.autocorr(lag=1)),
    })
    report["costante"] = report["dev_std"] < 1e-10
    report["persistente"] = report["autocorr_1g"].abs() >= soglia_persistenza
    report["utilizzabile"] = ~report["costante"] & report["persistente"]
    return report

# Ordine di priorità deciso PRIMA di guardare i dati, per motivi economici:
# rischio (volatilità), trend (momentum), legami fra asset (correlazioni), poi drawdown.
PRIORITA = [
    "equity_volatility", "bond_volatility", "gold_volatility",
    "equity_momentum", "bond_momentum", "gold_momentum",
    "equity_bond_correlation", "equity_gold_correlation", "bond_gold_correlation",
    "equity_drawdown", "bond_drawdown", "gold_drawdown",
    "equity_return", "bond_return", "gold_return",
]


def select_features(feature_bank: pd.DataFrame, training_end: str,
                    settings: dict[str, Any], soglia_correlazione: float = 0.7) -> list[str]:
    """Sceglie 5-8 feature usando solo il training (fino a training_end incluso).
    Regola: 1) si tengono solo le feature utilizzabili (controlla_variabilita);
    2) si scorrono in ordine di PRIORITA e se ne tiene una solo se la sua correlazione
    (Spearman, in valore assoluto) con tutte quelle già scelte è <= soglia;
    3) ci si ferma al massimo indicato nei settings."""
    minimo = settings["selection"]["suggested_minimum"]
    massimo = settings["selection"]["suggested_maximum"]

# Se la selezione è già stata conclusa e scritta nei settings, la usiamo così com'è
    # (scelta "congelata"), controllando solo che i nomi esistano.
    congelata = list(settings["selection"].get("selected_features") or [])
    if congelata:
        sconosciute = [f for f in congelata if f not in feature_bank.columns]
        if sconosciute:
            raise ValueError(f"Feature nei settings che non esistono: {sconosciute}")
        return congelata

    report = controlla_variabilita(feature_bank, training_end)
    train = feature_bank.loc[:training_end]
    corr = train.corr(method="spearman").abs()

    scelte = []
    for nome in PRIORITA:
        if nome not in feature_bank.columns or not report.loc[nome, "utilizzabile"]:
            continue
        if all(corr.loc[nome, altra] <= soglia_correlazione for altra in scelte):
            scelte.append(nome)
        if len(scelte) == massimo:
            break

    if len(scelte) < minimo:
        raise ValueError(f"Selezionate solo {len(scelte)} feature, ne servono almeno {minimo}")
    return scelte

def prepare_model_data(feature_bank: pd.DataFrame, selected_features: list[str],
                       settings: dict[str, Any]) -> dict[str, Any]:
    """Prepara i dati per i modelli: tiene solo le feature scelte (nell'ordine dato),
    divide in train / validation / test e standardizza con uno scaler adattato SOLO sul training.
    Restituisce feature_names, scaler, boundaries e scaled_blocks (tabelle con le date)."""
    if not selected_features:
        raise ValueError("La lista delle feature è vuota")
    dati = feature_bank[list(selected_features)]
    if not np.isfinite(dati.to_numpy()).all():
        raise ValueError("Ci sono valori non finiti nelle feature scelte")

    confini = dividi_date(feature_bank, settings)
    blocchi = {nome: dati.loc[b["start"]:b["end"]] for nome, b in confini.items()}

    # media e deviazione standard calcolate solo sul training
    scaler = StandardScaler().fit(blocchi["train"].to_numpy())

    scaled_blocks = {}
    for nome, blocco in blocchi.items():
        scaled_blocks[nome] = pd.DataFrame(scaler.transform(blocco.to_numpy()),
                                           index=blocco.index, columns=list(selected_features))
    return {
        "feature_names": list(selected_features),
        "scaler": scaler,
        "boundaries": confini,
        "scaled_blocks": scaled_blocks,
    }
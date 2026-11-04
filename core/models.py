"""Modelli, selezione e inferenza."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture 

BLOCCHI = ["train", "validation", "test"]


def indice_ordinamento(feature_names: list[str]) -> int:
    """Colonna usata per ordinare gli stati: la volatilità SPY se c'è, altrimenti la prima."""
    return feature_names.index("equity_volatility") if "equity_volatility" in feature_names else 0


def adatta_kmeans(prepared: dict[str, Any], k: int, seed: int):
    """Adatta K-Means con k gruppi sul SOLO training scalato, poi applica il modello
    (fermo, senza riaddestrarlo) a validation e test.
    I gruppi vengono rinumerati: 0 = volatilità SPY media più bassa, k-1 = più alta.
    Restituisce il modello e una Series di label con le date di tutti e tre i blocchi."""
    blocchi = prepared["scaled_blocks"]
    modello = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(blocchi["train"].to_numpy())

    # rinumerazione stabile dei gruppi in base al centro della volatilità SPY
    col = indice_ordinamento(prepared["feature_names"])
    ordine = np.argsort(modello.cluster_centers_[:, col])
    nuovo_numero = np.empty(k, dtype=int)
    nuovo_numero[ordine] = np.arange(k)
    modello.cluster_centers_ = modello.cluster_centers_[ordine]
    modello.labels_ = nuovo_numero[modello.labels_]

    label = pd.concat([pd.Series(modello.predict(blocchi[b].to_numpy()), index=blocchi[b].index)
                       for b in BLOCCHI])
    label.name = "kmeans_state"
    return modello, label

def adatta_gmm(prepared: dict[str, Any], k: int, seed: int, covariance_type: str = "diag"):
    """Adatta una Gaussian Mixture con k componenti (covarianze diagonali) sul SOLO training,
    poi la applica, ferma, a validation e test.
    Restituisce (modello, label, probabilità, info). Se il fit fallisce, modello/label/probabilità
    sono None e info contiene l'errore: così anche i fit falliti restano registrati."""
    blocchi = prepared["scaled_blocks"]
    info = {"k": k, "seed": seed, "stato": "ok", "errore": ""}
    try:
        modello = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                  random_state=seed, max_iter=500).fit(blocchi["train"].to_numpy())
    except Exception as e:
        info["stato"], info["errore"] = "fallito", repr(e)
        return None, None, None, info

    # rinumerazione: 0 = volatilità SPY media più bassa
    col = indice_ordinamento(prepared["feature_names"])
    ordine = np.argsort(modello.means_[:, col])
    for attr in ["weights_", "means_", "covariances_", "precisions_", "precisions_cholesky_"]:
        setattr(modello, attr, getattr(modello, attr)[ordine])

    prob = pd.concat([pd.DataFrame(modello.predict_proba(blocchi[b].to_numpy()), index=blocchi[b].index,
                                   columns=[f"gmm_probability_{i}" for i in range(k)])
                      for b in BLOCCHI])
    label = prob.to_numpy().argmax(axis=1)
    label = pd.Series(label, index=prob.index, name="gmm_state")

    info["convergenza"] = bool(modello.converged_)
    info["iterazioni"] = int(modello.n_iter_)
    info["varianza_minima"] = float(modello.covariances_.min())
    return modello, label, prob, info

def confronta_baseline(prepared: dict[str, Any], settings: dict[str, Any]) -> pd.DataFrame:
    """Prova K-Means e GMM per ogni numero di gruppi e seed dei settings, sugli stessi input
    e le stesse date. Misura tutto su training e validation (MAI sul test).
    Regola di scelta dichiarata:
    - seed: il migliore sul training (K-Means: inerzia minima; GMM: log-likelihood massima);
    - numero di gruppi: il migliore sulla validation (K-Means: silhouette massima;
      GMM: BIC di validation minimo)."""
    cfg = settings["baselines"]
    blocchi = prepared["scaled_blocks"]
    x_val = blocchi["validation"].to_numpy()
    n_feat = x_val.shape[1]
    righe = []

    for k in cfg["cluster_counts"]:
        for seed in cfg["seeds"]:
            # --- K-Means ---
            km, lab = adatta_kmeans(prepared, k, seed)
            riga = {"modello": "kmeans", "k": k, "seed": seed, "stato": "ok", "convergenza": True,
                    "inerzia_train": km.inertia_,
                    "silhouette_validation": silhouette_score(x_val, lab.loc[blocchi["validation"].index])}
            righe.append(_aggiungi_label(riga, lab, blocchi, k))

            # --- GMM ---
            gmm, lab, prob, info = adatta_gmm(prepared, k, seed, cfg["gmm_covariance_type"])
            riga = {"modello": "gmm", "k": k, "seed": seed, "stato": info["stato"], "errore": info["errore"]}
            if gmm is not None:
                n_parametri = (k - 1) + 2 * k * n_feat          # pesi + medie + varianze
                ll_val = gmm.score(x_val) * len(x_val)          # log-likelihood totale sulla validation
                riga.update({
                    "convergenza": info["convergenza"], "varianza_minima": info["varianza_minima"],
                    "loglik_train": gmm.score(blocchi["train"].to_numpy()) * len(blocchi["train"]),
                    "loglik_validation": ll_val,
                    "bic_validation": -2 * ll_val + n_parametri * np.log(len(x_val)),
                })
                riga = _aggiungi_label(riga, lab, blocchi, k)
            righe.append(riga)

    tab = pd.DataFrame(righe)
    tab["miglior_seed"] = False
    tab["scelto"] = False
    for modello, metrica_seed, verso_seed, metrica_k, verso_k in [
        ("kmeans", "inerzia_train", "min", "silhouette_validation", "max"),
        ("gmm", "loglik_train", "max", "bic_validation", "min"),
    ]:
        validi = tab[(tab["modello"] == modello) & (tab["stato"] == "ok") & (tab["convergenza"] == True)]
        migliori = [g[metrica_seed].idxmin() if verso_seed == "min" else g[metrica_seed].idxmax()
                    for _, g in validi.groupby("k")]
        tab.loc[migliori, "miglior_seed"] = True
        finalisti = tab.loc[migliori, metrica_k]
        tab.loc[finalisti.idxmin() if verso_k == "min" else finalisti.idxmax(), "scelto"] = True
    return tab


def _aggiungi_label(riga: dict, label: pd.Series, blocchi: dict, k: int) -> dict:
    """Occupazione minima e numero di cambi di label su training e validation."""
    for b in ["train", "validation"]:
        lab = label.loc[blocchi[b].index].to_numpy()
        quote = np.bincount(lab, minlength=k) / len(lab)
        riga[f"quota_min_{b}"] = quote.min()
        riga[f"stati_vuoti_{b}"] = int((quote == 0).sum())
        riga[f"cambi_{b}"] = int((lab[1:] != lab[:-1]).sum())
    return riga

def fit_baselines(prepared: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Confronta tutte le configurazioni (confronta_baseline), poi riaddestra sul training
    il K-Means e la GMM scelti. Stessi input e stesso scaler che userà l'HMM.
    Restituisce i modelli ("kmeans", "gmm"), le label e le probabilità con le date
    e la tabella di confronto."""
    confronto = confronta_baseline(prepared, settings)
    scelti = confronto[confronto["scelto"]].set_index("modello")

    k_km, seed_km = int(scelti.loc["kmeans", "k"]), int(scelti.loc["kmeans", "seed"])
    k_gmm, seed_gmm = int(scelti.loc["gmm", "k"]), int(scelti.loc["gmm", "seed"])
    kmeans, label_km = adatta_kmeans(prepared, k_km, seed_km)
    gmm, label_gmm, prob_gmm, _ = adatta_gmm(prepared, k_gmm, seed_gmm,
                                            settings["baselines"]["gmm_covariance_type"])
    return {
        "kmeans": kmeans,
        "gmm": gmm,
        "kmeans_labels": label_km,
        "gmm_labels": label_gmm,
        "gmm_probabilities": prob_gmm,
        "comparison": confronto,
        "feature_names": prepared["feature_names"],
    }


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
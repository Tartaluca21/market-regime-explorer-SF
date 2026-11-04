"""Grafici e file di presentazione."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

NOMI = {"equity": "SPY (azioni USA)", "bond": "TLT (Treasury lunghi)", "gold": "GLD (oro)"}


# --- Grafici esplorativi (notebook 01) ---

def grafico_prezzi_base100(prices: pd.DataFrame):
    """Prezzi adjusted ribasati a 100 alla prima data, in scala logaritmica."""
    base100 = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    for col in base100.columns:
        ax.plot(base100.index, base100[col], linewidth=1.3, label=NOMI.get(col, col))
    ax.set_yscale("log")
    ax.set_ylabel("Prezzo adjusted, base 100 (scala log)")
    ax.set_title(f"Prezzi ribasati a 100 · {prices.index[0]:%d/%m/%Y} – {prices.index[-1]:%d/%m/%Y}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def grafico_distribuzioni(returns: pd.DataFrame, limite: float = 8):
    """Istogrammi dei rendimenti giornalieri (%) per ogni asset."""
    fig, axes = plt.subplots(1, len(returns.columns), figsize=(12, 3.5), sharey=True)
    for ax, col in zip(axes, returns.columns):
        r = returns[col].dropna() * 100
        ax.hist(r, bins=100)
        ax.set_title(NOMI.get(col, col))
        ax.set_xlabel("Rendimento giornaliero (%)")
        ax.set_xlim(-limite, limite)
    axes[0].set_ylabel("Numero di giorni")
    fig.suptitle(f"Distribuzione dei rendimenti giornalieri (asse x tagliato a ±{limite}%)")
    fig.tight_layout()
    return fig


def grafico_correlazioni(corr: pd.DataFrame, titolo: str):
    """Heatmap di una matrice di correlazione con i valori scritti nelle celle."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * n, 1 + 1.0 * n))
    im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), corr.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title(titolo)
    fig.tight_layout()
    return fig

# --- Grafici dei gruppi (notebook 02) ---

COLORI_GRUPPI = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def grafico_timeline_gruppi(indice: pd.Series, label: pd.Series, titolo: str,
                            confini: dict[str, Any] | None = None):
    """Linea dell'indice (es. SPY base 100, scala log) con lo sfondo colorato
    secondo il gruppo assegnato ogni giorno, e linee verticali all'inizio di validation e test."""
    label = label.loc[indice.index]
    k = int(label.max()) + 1
    fig, ax = plt.subplots(figsize=(11, 4))
    # sfondo: un rettangolo colorato per ogni tratto consecutivo con lo stesso gruppo
    valori = label.to_numpy()
    date = label.index
    inizio = 0
    for i in range(1, len(valori) + 1):
        if i == len(valori) or valori[i] != valori[inizio]:
            fine = date[i] if i < len(valori) else date[-1]
            ax.axvspan(date[inizio], fine, color=COLORI_GRUPPI[valori[inizio] % 4], alpha=0.25, linewidth=0)
            inizio = i
    ax.plot(indice.index, indice, color="black", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("SPY, base 100 (scala log)")
    if confini:
        for nome in ["validation", "test"]:
            data = pd.Timestamp(confini[nome]["start"])
            ax.axvline(data, color="grey", linestyle="--", linewidth=1)
            ax.text(data, ax.get_ylim()[1], f" inizio {nome}", va="top", fontsize=8, color="grey")
    for g in range(k):
        ax.fill_between([], [], color=COLORI_GRUPPI[g % 4], alpha=0.5, label=f"gruppo {g}")
    ax.legend(loc="upper left", ncol=k, fontsize=8)
    ax.set_title(titolo)
    fig.tight_layout()
    return fig
# --- Da implementare nei notebook successivi ---

def build_charts(prices: pd.DataFrame, results: dict[str, Any],
                 settings: dict[str, Any]) -> dict[str, Any]:
    """IT: Creare grafici leggibili di stati, probabilità, asset e confronti.
    Mostrare date, unità e tempi del modello; restituire figure con nomi stabili.

    EN: Create readable charts of states, probabilities, assets and comparisons.
    Show dates, units and model timing; return figures with stable names.
    """
    raise NotImplementedError("Grafici da implementare / Charts not implemented.")


def export_figures(charts: dict[str, Any], output_dir: Path) -> None:
    """IT: Salvare figure PNG/PDF leggibili in results/figures/.
    Conservare date e unità; nessun training nascosto durante l'esportazione.

    EN: Save readable PNG/PDF figures in results/figures/.
    Preserve dates and units; no hidden model fitting during export.
    """
    raise NotImplementedError("Export da implementare / Figure export not implemented.")
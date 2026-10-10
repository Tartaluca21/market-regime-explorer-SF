"""IT: Integrazione del progetto. EN: Project integration."""

from pathlib import Path


def run_pipeline(project_root: Path) -> None:
    """IT: Collegare i moduli usando settings.json e data/market_data.csv.
    Scrivere solo in results/: feature, bundle completo, risultati, confronti e figure.
    Includere la proiezione degli stati; il confronto del rischio è opzionale.
    Modelli, scaler e metadati restano insieme in model_bundle.joblib.

    EN: Connect the modules using settings.json and data/market_data.csv.
    Write only to results/: features, complete bundle, results, comparisons and figures.
    Include state projections; the risk comparison is optional.
    Models, scaler and metadata stay together in model_bundle.joblib.
    """
    raise NotImplementedError("Pipeline da implementare / Pipeline not implemented.")

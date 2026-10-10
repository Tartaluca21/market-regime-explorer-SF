"""IT: Avvio della pipeline ancora da implementare.
EN: Launcher for the pipeline that remains to be implemented.
"""

from pathlib import Path

from core.pipeline import run_pipeline


def main() -> None:
    """IT: Individuare la cartella del progetto e chiamare la pipeline.
    La funzione chiamata non contiene ancora l'implementazione.

    EN: Locate the project directory and call the pipeline.
    The called function does not contain an implementation yet.
    """
    run_pipeline(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()

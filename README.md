# Market Regime Explorer

**IT** Costruire una pipeline di machine learning e una dashboard per leggere i regimi di azioni USA, Treasury e oro. Il team implementa dati, feature, modelli, test e app in 8-12 settimane.

**EN** Build a machine learning pipeline and a dashboard for US equity, Treasury and gold market regimes. The team implements data handling, features, models, tests and the app over 8-12 weeks.

## Inizia qui / Start here

**IT** Leggi `Project_Brief.pdf`, poi apri il notebook 01. Le celle di codice sono vuote; i file in `core/` contengono soltanto le interfacce da implementare. Non sono inclusi modelli allenati o risultati finali.

**EN** Read `Project_Brief.pdf`, then open notebook 01. Code cells are empty; `core/` contains only interfaces to implement. No trained models or final results are included.

| Ordine / Order | Notebook | Risultato / Output |
|---|---|---|
| 01 | `notebooks/01_data_and_features.ipynb` | Controlli, EDA e 15 feature / Checks, EDA and 15 features |
| 02 | `notebooks/02_selection_and_baselines.ipynb` | Selezione, scaler, K-Means e GMM / Selection, scaling, K-Means and GMM |
| 03 | `notebooks/03_hmm_and_validation.ipynb` | HMM, probabilità, profili e valutazione / HMM, probabilities, profiles and evaluation |
| 04 | `notebooks/04_pipeline_and_app.ipynb` | Pipeline riproducibile, test e dashboard / Reproducible pipeline, tests and dashboard |

## Avvio / Setup

**IT** Usa Python 3.12 e apri il terminale nella cartella del progetto.

**EN** Use Python 3.12 and open a terminal in the project folder.

```bash
python -m venv .venv
```

| Sistema / System | Attivazione / Activation |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |

```bash
python -m pip install -r requirements.txt
jupyter lab
```

**IT** Seleziona il kernel dell'ambiente. Il CSV è già nel pacchetto locale; serve internet solo per installare le librerie.

**EN** Select the environment's kernel. The CSV is already in the local package; internet is needed only to install libraries.

**IT** La versione per la repository pubblica non include il CSV: per lavorare, copia il tuo file locale in `data/market_data.csv`, senza aggiungerlo a Git.

**EN** The public repository version excludes the CSV: to work locally, copy your local file to `data/market_data.csv`, without adding it to Git.

## Un solo CSV / One CSV

`data/market_data.csv` - **3.868 righe / 3,868 rows**, **19 colonne / 19 columns**, **2004-11-18 → 2020-04-01**.

| Asset | Colonna del prezzo / Price column | Significato / Meaning |
|---|---|---|
| SPY | `spy_adj_close` | Azioni USA, S&P 500 / US equities, S&P 500 |
| TLT | `tlt_adj_close` | Treasury USA a lunga scadenza / Long-term US Treasuries |
| GLD | `gld_adj_close` | Oro tramite ETF / Gold through an ETF |

**IT** Una riga per data comune. Le altre colonne sono OHLC e volume per ciascun asset. I prezzi sono in USD: questo progetto riguarda il mercato USA e l'oro. I tre file originali sono stati uniti senza riempimenti o modifiche ai valori. Nessuna feature è precalcolata.

**EN** One row per common date. Other columns are OHLC and volume for each asset. Prices are in USD: this project covers US markets and gold. The three source files were joined without filling or changing values. No features are precomputed.

**IT** Usa i prezzi aggiustati per i rendimenti, come proxy dei rendimenti aggiustati; non mescolarli con OHLC non aggiustati. TLT rappresenta obbligazioni governative a lunga durata, non tutto il mercato obbligazionario. Il campione include il 2008 e l'inizio del COVID, ma termina prima della ripresa e non contiene il 2022.

**EN** Use adjusted prices as adjusted-return proxies; do not mix them with unadjusted OHLC. TLT represents long-duration government bonds, not the whole bond market. The sample includes 2008 and early COVID, but ends before the recovery and excludes 2022.

## Dove lavorare / Where to work

```text
market-regime-explorer/
├── README.md             avvio e mappa / setup and map
├── Project_Brief.pdf     guida bilingue / bilingual guide
├── settings.json         configurazione / configuration
├── requirements.txt      librerie / libraries
├── data/market_data.csv  unico input locale / single local input
├── notebooks/            quattro fasi / four stages
├── core/                 funzioni condivise / shared functions
├── tests/                test da scrivere / tests to write
├── results/              output da generare / outputs to generate
├── run.py                avvio pipeline / pipeline launcher
└── app.py                dashboard da costruire / dashboard to build
```

**IT** Sviluppa nel notebook, trasferisci le funzioni in `core/` e richiamale dal notebook. I nomi e le interfacce sono già indicati. Scrivi test sui dati, sui tempi delle feature e sulle probabilità; assegna responsabili e revisori in base alle disponibilità.

**EN** Develop in the notebook, move functions into `core/` and call them from the notebook. Names and interfaces are provided. Write tests for data, feature timing and probabilities; assign owners and reviewers according to availability.

**IT** Tutti i file generati vanno in `results/`: `features.csv`, `baseline_comparison.csv`, `hmm_diagnostics.csv`, `regime_results.csv`, `regime_profiles.csv`, `model_comparison.csv`, `model_bundle.joblib` e `figures/`. Il confronto del rischio opzionale aggiunge `forecast_comparison.csv`.

**EN** All generated files belong in `results/`: `features.csv`, `baseline_comparison.csv`, `hmm_diagnostics.csv`, `regime_results.csv`, `regime_profiles.csv`, `model_comparison.csv`, `model_bundle.joblib` and `figures/`. The optional risk comparison adds `forecast_comparison.csv`.

## Scelte iniziali / Starting choices

**IT** Costruisci 15 feature: rendimento, volatilità, momentum e drawdown per asset, più tre correlazioni. Seleziona circa 5-8 input sul training. Confronta K-Means, GMM e HMM diagonali con 2-4 stati e tre seed; massimo 40 tentativi HMM complessivi. Non forzare tre regimi o nomi come Risk-On/Risk-Off prima di leggere i profili.

**EN** Build 15 features: return, volatility, momentum and drawdown per asset, plus three correlations. Select about 5-8 inputs on training data. Compare K-Means, GMM and diagonal HMMs with 2-4 states and three seeds; at most 40 total HMM fits. Do not force three regimes or names such as Risk-On/Risk-Off before inspecting their profiles.

**IT** Split cronologico 60/20/20 dopo il warm-up. Selezione e scaler sul training; parametri sulla validation; valutazione finale con scelte congelate. Il filtro HMM continua tra i blocchi senza usare osservazioni future; lo smoothing descrive il passato. Confronta likelihood soltanto con gli stessi input e scaler.

**EN** Use a chronological 60/20/20 split after warm-up. Fit selection and scaling on training, choose parameters on validation, then freeze choices for final evaluation. HMM filtering continues across blocks without future observations; smoothing describes the past. Compare likelihood only on identical inputs and scaling.

**IT** La proiezione `pi_t @ A` stima la distribuzione degli stati alla prossima seduta. L'estensione opzionale confronta il rischio a cinque sedute con una baseline, usando etichette separate tra i blocchi. Un risultato nullo è valido; nessuna previsione di crisi o rendimento positivo è garantita.

**EN** The `pi_t @ A` projection estimates the next-session state distribution. The optional extension compares five-session risk against a baseline, keeping labels within their blocks. A null result is valid; crisis prediction or positive returns are not guaranteed.

## Quando è pronto / When it is ready

**IT** Dopo l'implementazione, questi comandi devono funzionare su un ambiente pulito. Lo starter non esegue ancora la pipeline; i test sono da scrivere.

**EN** After implementation, these commands must work in a clean environment. The starter does not yet run the pipeline; tests remain to be written.

```bash
python -m pytest
python run.py
streamlit run app.py
```

**IT** La dashboard legge gli output salvati senza rifare il training al cambio dei controlli. Mostra timeline dei regimi, probabilità, incertezza, profili dei tre asset e confronto dei modelli. Verifica la consegna su un altro laptop ed esporta tre o quattro figure leggibili.

**EN** The dashboard reads saved outputs without retraining when controls change. Show regime timelines, probabilities, uncertainty, three-asset profiles and model comparisons. Check the release on another laptop and export three or four readable figures.

## Fonte e pubblicazione / Source and publication

**IT** Fonte: [Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset), Oleh Onyshchak (jacksoncrow), v2. Il dataset è incluso in data/market_data.csv, disponibile su Kaggle, con licenza dichiarata CC0. Le serie SPY, TLT e GLD sono state unite sulle date comuni.

**EN** Source: [Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset), Oleh Onyshchak (jacksoncrow), v2.  Datasets available on Kaggle, with a declared CC0 license. The SPY, TLT and GLD series have been merged on their common dates.
**IT** Questi dati sono già stati esaminati nella preparazione dello starter. Il blocco finale è una valutazione storica didattica, non un test nuovo completamente indipendente.

**EN** These data have already been inspected while preparing the starter. The final block is an educational historical evaluation, not a fully independent new test.

SHA-256 `market_data.csv`: `a219614b65a51a34daa187d65a7aab0135e89406e141c263d38f00a98f6c0aee`.

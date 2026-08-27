# Deep Learning Applications — Laboratorio 1

Il Laboratorio 1 studia il **riuso e l'adattamento di modelli pre-addestrati** per il riconoscimento di segnali stradali, seguendo un percorso progressivo dalla classificazione di immagini già ritagliate fino all'**object detection in scene stradali complete**.

## Obiettivi del laboratorio

Il lavoro affronta tre domande principali:

* quanto sono informative le feature ImageNet senza aggiornare il backbone;
* quanto migliora la classificazione adattando progressivamente una ResNet al dominio GTSRB;
* se un backbone ResNet-50 fine-tuned per la classificazione GTSRB trasferisce efficacemente a un detector Faster R-CNN.

Il protocollo sperimentale mantiene separati training, validation e test, usa seed e configurazioni esplicite e conserva gli artifact necessari all'analisi senza versionare dataset, checkpoint o output pesanti.

## Struttura del repository

```text
DLA_LAB1/
├── README.md
├── environment.yml
│
├── Exercise1/
│   ├── README.md
│   ├── run_all_exercise1.sh
│   ├── main.py
│   ├── data.py
│   ├── eda.py
│   ├── feature_extraction.py
│   ├── classical_baseline.py
│   ├── fine_tuning.py
│   └── assets/
│
├── Exercise2/
│   ├── README.md
│   ├── configuration.py
│   ├── data.py
│   ├── models.py
│   ├── training.py
│   ├── main.py
│   ├── evaluate_test.py
│   ├── configs/
│   └── assets/
│
└── Exercise3/
    ├── README.md
    ├── main.py
    ├── analysis/
    ├── backbone/
    ├── configs/
    ├── data_pipeline/
    ├── evaluation/
    ├── experiments/
    ├── models/
    ├── training/
    ├── visualization/
    └── assets/
```

## Esercizi

### Exercise 1 — Feature extraction e fine-tuning

L'[Exercise 1](Exercise1/README.md) utilizza il **German Traffic Sign Recognition Benchmark (GTSRB)** e comprende tre fasi:

1. EDA del dataset;
2. estrazione di feature con ResNet-18 e ResNet-50 pre-addestrate e classificazione con LinearSVC, KNN e LDA;
3. fine-tuning con tre strategie (`classifier`, `last_block`, `full`) e due teste (`linear`, `mlp`).

La campagna principale comprende **6 baseline classiche** e **12 configurazioni di fine-tuning da 5 epoche**.

### Exercise 2 — Consolidamento della pipeline

L'[Exercise 2](Exercise2/README.md) non introduce un nuovo problema di classificazione: riorganizza la procedura dell'Exercise 1.3 in una pipeline più compatta e riproducibile.

La versione finale separa:

* configurazione con YAML e OmegaConf;
* preparazione dei dati;
* costruzione del modello;
* training e validation;
* checkpoint;
* valutazione finale del test.

Il test ufficiale viene valutato tramite uno script separato, evitando di usarlo durante la selezione del modello.

### Exercise 3 — Object detection

L'[Exercise 3](Exercise3/README.md) estende il progetto dalla classificazione di crop GTSRB alla **detection di cartelli in immagini stradali complete** con Faster R-CNN ResNet-50-FPN.

Lo studio confronta quattro configurazioni:

| Run | Inizializzazione | Componenti del backbone addestrabili |
| --- | ---------------- | ------------------------------------ |
| A   | COCO             | nessuno                              |
| B   | GTSRB            | nessuno                              |
| C   | GTSRB            | `layer4` + FPN                       |
| D   | GTSRB            | `layer3` + `layer4` + FPN            |

Il dataset effettivamente usato dal detector contiene **545 immagini e 851 oggetti**, dopo la rimozione di una duplicazione esatta nelle annotazioni.

## Ambiente e riproducibilità

L'ambiente Conda di riferimento è definito in [`environment.yml`](environment.yml).

Dalla root del repository:

```bash
conda env create -f DLA_LAB1/environment.yml
conda activate DLA2026_clean
cd DLA_LAB1
```

Versioni principali registrate nell'ambiente:

* Python 3.12.13
* PyTorch 2.13.0 + CUDA 12.6
* Torchvision 0.28.0
* NumPy 2.5.1
* Pandas 3.0.3
* Matplotlib 3.11.0
* Scikit-learn 1.9.0
* Pillow 12.2.0
* OmegaConf 2.3.1
* datasets 3.6.0
* TorchMetrics 1.9.0
* faster-coco-eval 1.7.2
* Weights & Biases 0.28.1

Il seed di riferimento è `42`. La riproducibilità è gestita attraverso split controllati, configurazioni salvate e checkpoint; quando l'esecuzione GPU non è forzata in modalità deterministica, non viene assunto determinismo bit-a-bit.

## Entry point principali

I comandi seguenti vanno eseguiti dalla directory `DLA_LAB1`.

### Exercise 1

EDA:

```bash
python Exercise1/main.py eda
```

Matrice completa delle baseline classiche:

```bash
python Exercise1/main.py baseline \
  --models all \
  --classifiers all \
  --wandb
```

Esempio di fine-tuning:

```bash
python Exercise1/main.py finetune \
  --model resnet18 \
  --strategy full \
  --classifier linear \
  --epochs 5 \
  --wandb
```

Campagna completa dell'Exercise 1:

```bash
bash Exercise1/run_all_exercise1.sh
```

Lo script esegue le 6 baseline classiche e le 12 configurazioni di fine-tuning e abilita il logging W&B.

### Exercise 2

Training con la configurazione di default:

```bash
python Exercise2/main.py
```

Smoke test:

```bash
python Exercise2/main.py experiment.smoke_test_batches=2
```

Valutazione separata sul test:

```bash
python Exercise2/evaluate_test.py \
  --checkpoint Exercise2/outputs/runs/<run_id>/best_model.pt
```

### Exercise 3

La CLI unificata espone i comandi pubblici dell'esercizio. Per consultare le opzioni:

```bash
python -m Exercise3.main --help
```

Per la matrice sperimentale A–D:

```bash
python -m Exercise3.main matrix --help
```

I dettagli delle configurazioni, del trasferimento del backbone e della valutazione COCO-style sono documentati nel README specifico dell'esercizio.

## Tracking degli esperimenti

L'Exercise 1 supporta logging locale e **Weights & Biases** per confrontare configurazioni, curve e metriche.

L'Exercise 2 utilizza output locali e checkpoint, senza integrazione W&B nella versione finale.

L'Exercise 3 supporta tracking sperimentale per le run di detection e conserva separatamente metriche, history, checkpoint e report di valutazione.

## Politica del repository

Per mantenere il repository leggero e riproducibile non vengono versionati:

* dataset;
* checkpoint e pesi addestrati;
* feature `.npz`;
* output sperimentali completi;
* directory W&B locali;
* cache;
* file temporanei;
* materiale di sviluppo non necessario alla consegna.

La documentazione fa quindi riferimento al **codice e alle configurazioni versionate**, mentre i risultati numerici derivano dagli artifact delle run effettivamente eseguite.

## Navigazione

* [Exercise 1 — Transfer learning per la classificazione](Exercise1/README.md)
* [Exercise 2 — Pipeline configurabile e riproducibile](Exercise2/README.md)
* [Exercise 3 — Object detection con Faster R-CNN](Exercise3/README.md)

## Riferimenti e assistenza AI

Il progetto utilizza principalmente PyTorch, Torchvision, Scikit-learn, OmegaConf, Hugging Face Datasets, TorchMetrics e Weights & Biases, insieme ai dataset GTSRB e German Traffic Sign Detection.

ChatGPT è stato utilizzato come supporto per chiarimenti teorici, organizzazione del lavoro, revisione del codice, debugging, analisi degli artifact e documentazione. Le scelte implementative e i risultati riportati sono stati verificati sul codice e sugli output effettivi del progetto.

# Deep Learning Applications — Laboratorio 2

Il Laboratorio 2 studia l'uso di **Transformer pre-addestrati** per task di classificazione testuale e retrieval multimodale, seguendo un percorso progressivo dal riuso di rappresentazioni congelate al **fine-tuning end-to-end di DistilBERT**, fino alla costruzione di un sistema **text-to-image retrieval** basato su CLIP.

## Obiettivi del laboratorio

Il lavoro affronta tre domande principali:

* quanto sono informative le rappresentazioni di DistilBERT senza aggiornare il Transformer;
* quanto migliora la sentiment analysis adattando end-to-end DistilBERT al dominio Rotten Tomatoes;
* se uno spazio multimodale pre-addestrato come CLIP permette di costruire un motore di ricerca semantica immagini-testo senza addestramento aggiuntivo su Flickr8k.

Il protocollo sperimentale mantiene separati training, validation e test, usa configurazioni esplicite e conserva gli artifact necessari all'analisi senza versionare dataset, checkpoint, embedding o output pesanti.

## Struttura del repository

```text
DLA_LAB2/
├── README.md
├── DLA-Lab2.ipynb
│
├── Exercise1/
│   ├── README.md
│   ├── main.py
│   ├── data.py
│   ├── eda.py
│   ├── transformer_inspection.py
│   ├── feature_extraction.py
│   ├── baseline_classifier.py
│   └── assets/
│
├── Exercise2/
│   ├── README.md
│   ├── main.py
│   ├── data.py
│   ├── model.py
│   ├── training.py
│   └── assets/
│
└── Exercise3/
    ├── README.md
    ├── main.py
    ├── data.py
    ├── model.py
    ├── indexing.py
    ├── retrieval.py
    ├── evaluation.py
    ├── app.py
    ├── io_utils.py
    └── assets/
```

Il notebook `DLA-Lab2.ipynb` contiene la consegna originale del laboratorio; il codice definitivo è organizzato nei tre esercizi e documentato nei rispettivi README.

## Esercizi

### Exercise 1 — DistilBERT come feature extractor

L'[Exercise 1](Exercise1/README.md) utilizza il dataset **Cornell Rotten Tomatoes** e comprende tre fasi:

1. analisi esplorativa del dataset;
2. ispezione di tokenizer, padding e rappresentazioni di DistilBERT;
3. estrazione di feature con Transformer congelato e classificazione tramite `StandardScaler + LinearSVC`.

Il modello usa la rappresentazione del primo token dell'ultimo hidden state di DistilBERT, ottenendo vettori da 768 componenti.

La selezione del parametro `C` viene eseguita esclusivamente sulla validation; con `C=0.01` la baseline finale raggiunge sul test:

```text
Accuracy = 0.800188
Macro-F1 = 0.800148
```

### Exercise 2 — Full fine-tuning di DistilBERT

L'[Exercise 2](Exercise2/README.md) mantiene lo stesso task di sentiment analysis, ma sostituisce la rappresentazione congelata con il **fine-tuning completo di DistilBERT**.

La pipeline utilizza:

* `Dataset.map()` per il preprocessing;
* `DataCollatorWithPadding` per il padding dinamico;
* `AutoModelForSequenceClassification`;
* Hugging Face `Trainer`;
* accuracy e macro-F1 come metriche;
* macro-F1 di validation per la selezione del checkpoint.

La run finale seleziona `checkpoint-1602`, corrispondente alla terza epoca, e ottiene sul test:

```text
Accuracy = 0.847092
Macro-F1 = 0.847085
```

Rispetto alla baseline congelata dell'Exercise 1, il miglioramento è di circa **4,69 punti percentuali**.

### Exercise 3 — Text-to-image retrieval con CLIP

L'[Exercise 3](Exercise3/README.md) implementa la variante **3.3 — Text-to-Image Retrieval** utilizzando **Flickr8k** e il checkpoint:

```text
openai/clip-vit-base-patch32
```

Il sistema separa indicizzazione offline e ricerca online:

```text
immagini Flickr8k
        ↓
CLIP image encoder
        ↓
embedding normalizzati
        ↓
indice persistente

query testuale
        ↓
CLIP text encoder
        ↓
embedding normalizzato
        ↓
similarità coseno
        ↓
ranking
        ↓
top-k immagini
```

L'applicazione finale indicizza le **8.000 immagini** di Flickr8k e mette a disposizione una demo interattiva tramite Gradio.

La valutazione quantitativa viene mantenuta separata dalla galleria completa dell'applicazione: usa soltanto le **1.000 immagini del test** come candidate e le relative **5.000 didascalie** come query.

I risultati registrati sono:

| Metrica | Valore |
|---|---:|
| Recall@1 | 0,5378 |
| Recall@5 | 0,8078 |
| Recall@10 | 0,8882 |
| Rango mediano | 1 |
| Rango medio | 5,3872 |

## Ambiente e riproducibilità

Gli esperimenti del Laboratorio 2 sono stati eseguiti nell'ambiente Conda locale:

```bash
conda activate DLA2026-transformers
```

Dalla root del repository:

```bash
cd DLA_LAB2
```

Tra le versioni effettivamente registrate negli artifact finali del laboratorio risultano:

* PyTorch 2.12.0 + CUDA 12.6;
* Transformers 5.14.1;
* NumPy 2.4.4.

Il codice utilizza inoltre Hugging Face Datasets, Scikit-learn, Pandas, Matplotlib, Pillow e Gradio.

Il seed di riferimento per le componenti supervisionate è `42`. Il retrieval CLIP è usato in modalità zero-shot e non introduce training sul dataset Flickr8k.

Un file `environment.yml` dedicato al Lab 2 non è ancora versionato; prima dell'audit finale l'ambiente verrà consolidato distinguendo le versioni effettivamente verificate da eventuali dipendenze compatibili proposte.

## Entry point principali

I comandi seguenti vanno eseguiti dalla directory `DLA_LAB2`.

### Exercise 1

EDA:

```bash
python Exercise1/main.py eda
```

Ispezione del Transformer:

```bash
python Exercise1/main.py inspect-transformer
python Exercise1/main.py inspect-transformer-batch
```

Estrazione delle feature:

```bash
python Exercise1/main.py extract-features
```

Selezione della baseline:

```bash
python Exercise1/main.py select-baseline \
  --c-values 0.01 0.1 1 10
```

Valutazione finale:

```bash
python Exercise1/main.py evaluate-test
```

### Exercise 2

Ispezione della tokenizzazione:

```bash
python Exercise2/main.py inspect-tokenization
```

Ispezione del classificatore:

```bash
python Exercise2/main.py inspect-model
```

Full fine-tuning:

```bash
python Exercise2/main.py train
```

Valutazione finale:

```bash
python Exercise2/main.py evaluate-test
```

### Exercise 3

La CLI unificata espone i comandi pubblici dell'applicazione.

Per consultare le opzioni:

```bash
python Exercise3/main.py --help
```

Ispezione di Flickr8k:

```bash
python Exercise3/main.py inspect-dataset
```

Ispezione di CLIP:

```bash
python Exercise3/main.py inspect-clip
```

Costruzione dell'indice:

```bash
python Exercise3/main.py build-index
```

Ricerca testuale:

```bash
python Exercise3/main.py search \
  --query "a dog playing outside"
```

Valutazione del retrieval:

```bash
python Exercise3/main.py evaluate-retrieval
```

Avvio dell'applicazione Gradio:

```bash
python Exercise3/main.py launch-app
```

I dettagli relativi a dataset, indicizzazione, ranking, valutazione e interfaccia sono documentati nel README specifico dell'esercizio.

## Tracking degli esperimenti

L'Exercise 1 salva localmente feature, modelli classici, prediction e metriche; nel repository vengono mantenuti soltanto gli asset e la documentazione necessari.

L'Exercise 2 utilizza artifact locali di Hugging Face `Trainer` e supporta opzionalmente **Weights & Biases** tramite il flag `--wandb`.

L'Exercise 3 non richiede tracking di training, perché CLIP viene utilizzato zero-shot; conserva localmente indice, metadata, risultati di evaluation e artifact necessari alla demo.

## Politica del repository

Per mantenere il repository leggero e riproducibile non vengono versionati:

* dataset;
* checkpoint e pesi addestrati;
* feature ed embedding di grandi dimensioni;
* indici completi del retrieval;
* output sperimentali completi;
* directory W&B locali;
* cache Hugging Face;
* file temporanei;
* materiale di sviluppo non necessario alla consegna.

La documentazione fa quindi riferimento al **codice e agli asset selezionati**, mentre i risultati numerici derivano dagli artifact delle run effettivamente eseguite.

## Navigazione

* [Exercise 1 — DistilBERT congelato e stable baseline](Exercise1/README.md)
* [Exercise 2 — Full fine-tuning di DistilBERT](Exercise2/README.md)
* [Exercise 3 — Text-to-image retrieval con CLIP](Exercise3/README.md)

## Riferimenti e assistenza AI

Il progetto utilizza principalmente PyTorch, Hugging Face Transformers e Datasets, Scikit-learn, NumPy, Pandas, Matplotlib, Pillow e Gradio, insieme ai dataset **Cornell Rotten Tomatoes** e **Flickr8k** e ai modelli pre-addestrati **DistilBERT** e **CLIP**.

Riferimenti principali:

* V. Sanh, L. Debut, J. Chaumond, T. Wolf, *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*, 2019.
* A. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML, 2021.

ChatGPT è stato utilizzato come supporto per chiarimenti teorici, organizzazione del lavoro, revisione del codice, debugging, analisi degli artifact, costruzione dei grafici e documentazione. Le scelte implementative e i risultati riportati sono stati verificati sul codice e sugli output effettivi del progetto.

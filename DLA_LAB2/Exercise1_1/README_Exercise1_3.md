# Exercise 1.3 — Stable Baseline con DistilBERT e LinearSVC

## Deep Learning Applications — Laboratorio 2

L’Esercizio 1.3 costruisce una baseline stabile per la classificazione binaria del sentiment sul dataset **Cornell Rotten Tomatoes**.

Il modello Transformer non viene addestrato: **DistilBERT viene mantenuto completamente congelato e usato come estrattore di feature**. Per ogni frase viene estratta la rappresentazione del primo token dell’ultimo layer, cioè:

```python
last_hidden_state[:, 0, :]
```

Ogni recensione viene così trasformata in un vettore di `768` componenti. Queste feature vengono poi standardizzate e utilizzate per addestrare un classificatore lineare `LinearSVC` di Scikit-learn.

Il flusso completo è:

```text
testo
  ↓
AutoTokenizer
  ↓
input_ids + attention_mask
  ↓
DistilBERT congelato
  ↓
last_hidden_state[:, 0, :]
  ↓
vettore della frase [768]
  ↓
StandardScaler
  ↓
LinearSVC
  ↓
sentiment negativo o positivo
```

L’ambiente Conda utilizzato localmente è:

```text
DLA2026-transformers
```

Il checkpoint utilizzato è:

```text
distilbert/distilbert-base-uncased
```

---

## Requisiti della consegna

La consegna ufficiale richiede di:

1. usare DistilBERT come feature extractor;
2. estrarre rappresentazioni per i testi dei diversi split;
3. addestrare un classificatore classico;
4. valutare le prestazioni su validation e test.

La consegna suggerisce l’uso della Hugging Face feature-extraction `pipeline`, del token `[CLS]` dell’ultimo Transformer layer e di un classificatore come `LinearSVC`.

Nel progetto è stata scelta un’implementazione diretta con `AutoTokenizer` e `AutoModel`, invece della `pipeline`, perché permette di controllare esplicitamente:

- batching;
- padding dinamico;
- device CPU/GPU;
- `attention_mask`;
- shape degli input e degli output;
- selezione di `last_hidden_state[:, 0, :]`;
- conversione e salvataggio delle feature;
- uso di `torch.inference_mode()`.

Questa scelta soddisfa lo stesso obiettivo della consegna mantenendo il flusso più trasparente e verificabile.

---

## Dataset

Il dataset è caricato tramite Hugging Face Datasets:

```text
cornell-movie-review-data/rotten_tomatoes
```

### Split ufficiali

| Split | Esempi | Negative | Positive |
|---|---:|---:|---:|
| Train | 8.530 | 4.265 | 4.265 |
| Validation | 1.066 | 533 | 533 |
| Test | 1.066 | 533 | 533 |
| **Totale** | **10.662** | **5.331** | **5.331** |

Mapping delle classi:

```text
0 → neg
1 → pos
```

Tutti gli split sono perfettamente bilanciati. Per questo motivo non sono stati utilizzati:

- `class_weight="balanced"`;
- oversampling;
- undersampling;
- sampler bilanciati.

Il protocollo mantiene gli split ufficiali separati:

- **train:** adattamento dello scaler e addestramento di `LinearSVC`;
- **validation:** selezione dell’iperparametro `C`;
- **test:** valutazione finale della configurazione già selezionata.

---

## Struttura dei file

La parte principale dell’Esercizio 1.3 utilizza la seguente organizzazione:

```text
Exercise1/
├── main.py
├── data.py
├── transformer_inspection.py
├── feature_extraction.py
├── baseline_classifier.py
└── outputs/
    └── exercise_1_3/
        ├── features/
        │   ├── train_features.npz
        │   ├── validation_features.npz
        │   └── test_features.npz
        ├── models/
        │   ├── linear_svc_pipeline.joblib
        │   └── selected_linear_svc_pipeline.joblib
        ├── predictions/
        │   ├── validation_predictions.npz
        │   ├── selected_validation_predictions.npz
        │   └── test_predictions.npz
        └── results/
            ├── token_length_summary.csv
            ├── feature_extraction_metadata.json
            ├── validation_metrics.json
            ├── validation_classification_report.json
            ├── validation_model_selection.csv
            ├── selected_baseline.json
            ├── selected_validation_classification_report.json
            ├── test_metrics.json
            └── test_classification_report.json
```

### Responsabilità dei moduli

#### `data.py`

Carica il dataset Rotten Tomatoes e restituisce il `DatasetDict` contenente gli split ufficiali.

#### `transformer_inspection.py`

Contiene il checkpoint condiviso e le funzioni usate nell’Esercizio 1.2 per comprendere tokenizer, input e output di DistilBERT.

#### `feature_extraction.py`

Gestisce:

- preflight delle lunghezze tokenizzate;
- selezione del device;
- caricamento di tokenizer e DistilBERT;
- congelamento del modello;
- smoke test;
- estrazione completa delle feature `[CLS]`;
- controlli su shape, dtype e valori finiti;
- salvataggio degli archivi `.npz` e dei metadati.

#### `baseline_classifier.py`

Gestisce:

- caricamento e validazione degli archivi di feature;
- costruzione della pipeline `StandardScaler + LinearSVC`;
- baseline preliminare con `C=1`;
- selezione di `C` sulla validation;
- salvataggio del modello selezionato;
- valutazione finale sul test senza refit;
- salvataggio di metriche, report e predizioni.

#### `main.py`

Espone i sottocomandi CLI dell’esercizio.

---

# 1. Preflight delle lunghezze tokenizzate

Prima di elaborare l’intero dataset è stata misurata la lunghezza di ogni frase secondo il tokenizer DistilBERT, includendo `[CLS]` e `[SEP]`.

Durante questa analisi:

```text
padding=False
truncation=False
```

## Risultati verificati

| Split | Esempi | Media | Dev. standard | Minimo | Mediana | 95° percentile | Massimo | Oltre 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 8.530 | 27,37 | 11,27 | 3 | 27 | 47 | 78 | 0 |
| Validation | 1.066 | 27,32 | 11,61 | 4 | 27 | 49 | 72 | 0 |
| Test | 1.066 | 27,66 | 11,53 | 5 | 27 | 48 | 67 | 0 |

DistilBERT supporta al massimo `512` posizioni. Il testo più lungo osservato contiene soltanto `78` token.

### Decisione

```text
truncation=False
padding dinamico per batch
nessun max_length artificiale
```

Non viene quindi eliminata alcuna parte delle recensioni.

## Comando

```powershell
python Exercise1/main.py feature-preflight
```

## Artifact

```text
Exercise1/outputs/exercise_1_3/results/token_length_summary.csv
```

---

# 2. Smoke test della feature extraction

Prima dell’estrazione completa è stato eseguito un test su otto esempi del training, con batch size `4`.

## Comando verificato

```powershell
python Exercise1/main.py feature-smoke-test \
  --max-examples 8 \
  --batch-size 4 \
  --device auto
```

In PowerShell il comando può anche essere scritto su una singola riga:

```powershell
python Exercise1/main.py feature-smoke-test --max-examples 8 --batch-size 4 --device auto
```

## Risultati verificati

```text
Resolved device: cuda
Model training mode: False
Total model parameters: 66.362.880
Trainable model parameters: 0
```

Shape del primo batch:

```text
input_ids:          (4, 52)
attention_mask:     (4, 52)
last_hidden_state:  (4, 52, 768)
CLS features:       (4, 768)
```

Output aggregato:

```text
Feature matrix: (8, 768), float32
Labels:         (8,), int64
Requires gradient: False
```

Lo smoke test ha confermato che:

- la GPU viene rilevata correttamente;
- il modello è in modalità valutazione;
- tutti i parametri sono congelati;
- il forward non costruisce gradienti;
- una frase produce una feature di `768` componenti;
- il numero di righe delle feature coincide con il numero di esempi.

Nello smoke test non viene salvato alcun file e non viene addestrato alcun classificatore.

---

# 3. Estrazione completa delle feature

## Configurazione

```text
Model:       DistilBertModel
Device:      cuda
Batch size:  32
Padding:     dinamico nel singolo batch
Truncation:  disabilitata
Feature:     last_hidden_state[:, 0, :]
Dtype:       float32
Gradienti:   disabilitati
```

Il modello viene preparato con:

```python
model.requires_grad_(False)
model.eval()
```

Il forward viene eseguito dentro:

```python
with torch.inference_mode():
    outputs = model(**model_inputs)
```

Per ogni batch:

```python
cls_features = outputs.last_hidden_state[:, 0, :]
```

La shape passa da:

```text
last_hidden_state: [batch_size, sequence_length, 768]
```

a:

```text
cls_features: [batch_size, 768]
```

## Comando verificato

```powershell
python Exercise1/main.py extract-features --batch-size 32 --device auto
```

## Risultati verificati

| Split | Feature shape | Label shape | Tempo di estrazione |
|---|---:|---:|---:|
| Train | `(8530, 768)` | `(8530,)` | 12,60 s |
| Validation | `(1066, 768)` | `(1066,)` | 1,59 s |
| Test | `(1066, 768)` | `(1066,)` | 1,64 s |

Tutte le feature:

- sono `float32`;
- hanno valori finiti;
- mantengono lo stesso ordinamento dello split originale;
- sono associate a label `int64`;
- sono salvate insieme agli indici originali dello split.

## Contenuto degli archivi `.npz`

Ogni archivio contiene:

```text
features → [numero_esempi, 768], float32
labels   → [numero_esempi], int64
indices  → [numero_esempi], int64
```

Gli indici permettono di verificare che le righe siano ancora allineate agli esempi originali.

## Artifact

```text
Exercise1/outputs/exercise_1_3/features/train_features.npz
Exercise1/outputs/exercise_1_3/features/validation_features.npz
Exercise1/outputs/exercise_1_3/features/test_features.npz
Exercise1/outputs/exercise_1_3/results/feature_extraction_metadata.json
```

Il caching delle feature evita di ripetere il forward di DistilBERT per ogni configurazione del classificatore.

---

# 4. Classificatore stabile

Il classificatore scelto è una pipeline Scikit-learn:

```text
StandardScaler
      ↓
LinearSVC
```

## Perché standardizzare

Le `768` componenti delle rappresentazioni DistilBERT possono avere scale e dispersioni differenti. `StandardScaler` trasforma ogni componente utilizzando media e deviazione standard calcolate sul training.

La pipeline viene adattata con:

```python
pipeline.fit(train_features, train_labels)
```

In questo modo:

- lo scaler vede soltanto il train;
- il classificatore vede soltanto il train;
- validation e test vengono solamente trasformati e predetti;
- non viene introdotto leakage.

## Configurazione del classificatore

```text
Classifier:    LinearSVC
random_state:  42
max_iter:      10000
dual:          False
class_weight:  None
```

È stato impostato `dual=False` perché il numero di esempi di training è maggiore della dimensionalità:

```text
8.530 esempi > 768 feature
```

---

# 5. Baseline preliminare con C = 1

Prima della selezione è stata eseguita una baseline iniziale con:

```text
C = 1.0
```

## Comando

```powershell
python Exercise1/main.py train-baseline
```

## Risultati sulla validation

```text
Accuracy:                0,818011
Macro-F1:                0,817934
Fit time:                5,406 s
Prediction time:         0,014 s
Iterazioni LinearSVC:    10 / 10000
Convergenza:             sì
```

Matrice di confusione:

```text
[[447  86]
 [108 425]]
```

Interpretazione:

```text
447 true negative
 86 false positive
108 false negative
425 true positive
```

Il test non è stato caricato durante questa fase.

---

# 6. Selezione dell’iperparametro C

La configurazione finale è stata selezionata esclusivamente sulla validation.

## Griglia valutata

```text
C ∈ {0.01, 0.1, 1.0, 10.0}
```

## Criterio di selezione

1. maggiore validation macro-F1;
2. maggiore validation accuracy in caso di parità;
3. valore di `C` più piccolo in caso di ulteriore parità.

La macro-F1 è stata scelta come metrica principale, pur essendo il dataset bilanciato, perché considera separatamente la qualità sulle due classi.

## Comando verificato

```powershell
python Exercise1/main.py select-baseline --c-values 0.01 0.1 1 10
```

## Risultati reali

| C | Validation accuracy | Validation macro-F1 | Iterazioni | Fit time |
|---:|---:|---:|---:|---:|
| **0,01** | **0,820826** | **0,820742** | 9 | 3,339 s |
| 0,1 | 0,818949 | 0,818865 | 10 | 4,559 s |
| 1 | 0,818011 | 0,817934 | 10 | 4,447 s |
| 10 | 0,818011 | 0,817934 | 10 | 4,411 s |

Tutte le configurazioni hanno raggiunto la convergenza entro `10000` iterazioni.

## Configurazione selezionata

```text
StandardScaler + LinearSVC
C = 0.01
```

Il valore più piccolo di `C` corrisponde a una regolarizzazione più forte e ha generalizzato leggermente meglio sulla validation.

### Matrice di confusione selezionata

```text
[[449  84]
 [107 426]]
```

Interpretazione:

```text
449 true negative
 84 false positive
107 false negative
426 true positive
```

Il test non è stato caricato o valutato durante la selezione.

## Artifact

```text
Exercise1/outputs/exercise_1_3/results/validation_model_selection.csv
Exercise1/outputs/exercise_1_3/results/selected_baseline.json
Exercise1/outputs/exercise_1_3/results/selected_validation_classification_report.json
Exercise1/outputs/exercise_1_3/models/selected_linear_svc_pipeline.joblib
Exercise1/outputs/exercise_1_3/predictions/selected_validation_predictions.npz
```

---

# 7. Valutazione finale sul test

Dopo la selezione, il modello salvato è stato valutato sul test una sola volta.

La pipeline non è stata riaddestrata, modificata o adattata nuovamente:

```text
Pipeline refitted before test: False
Test used for model selection: False
```

## Comando verificato

```powershell
python Exercise1/main.py evaluate-test
```

Il comando non accetta un nuovo valore di `C`, così la configurazione non può essere modificata dopo l’apertura del test.

## Risultati finali

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Validation | 0,820826 | 0,820742 |
| Test | **0,800188** | **0,800148** |

Tempo di predizione sul test:

```text
0,052 s
```

### Gap validation-test

```text
Accuracy:  0,020638 ≈ 2,06 punti percentuali
Macro-F1:  0,020594 ≈ 2,06 punti percentuali
```

Il test è leggermente più difficile della validation, ma non si osserva un collasso delle prestazioni.

## Matrice di confusione sul test

```text
[[434  99]
 [114 419]]
```

Interpretazione:

```text
434 true negative
 99 false positive
114 false negative
419 true positive
```

Errori complessivi:

```text
99 + 114 = 213 errori su 1.066 esempi
```

Recall per classe ricavato dalla matrice:

```text
Negative recall = 434 / 533 ≈ 0,8143
Positive recall = 419 / 533 ≈ 0,7861
```

Il classificatore riconosce quindi leggermente meglio le recensioni negative. La differenza resta contenuta e accuracy e macro-F1 sono quasi identiche, coerentemente con il perfetto bilanciamento del dataset.

## Artifact finali

```text
Exercise1/outputs/exercise_1_3/results/test_metrics.json
Exercise1/outputs/exercise_1_3/results/test_classification_report.json
Exercise1/outputs/exercise_1_3/predictions/test_predictions.npz
```

---

# 8. Riproduzione completa

Dalla root `DLA_LAB2`:

## Attivazione ambiente

```powershell
conda activate DLA2026-transformers
```

## Controllo sintattico

```powershell
python -m compileall -q Exercise1
```

Nessun output indica compilazione riuscita.

## Preflight

```powershell
python Exercise1/main.py feature-preflight
```

## Smoke test

```powershell
python Exercise1/main.py feature-smoke-test --max-examples 8 --batch-size 4 --device auto
```

## Estrazione completa

```powershell
python Exercise1/main.py extract-features --batch-size 32 --device auto
```

## Baseline preliminare opzionale

```powershell
python Exercise1/main.py train-baseline
```

## Selezione sulla validation

```powershell
python Exercise1/main.py select-baseline --c-values 0.01 0.1 1 10
```

## Valutazione finale sul test

```powershell
python Exercise1/main.py evaluate-test
```

L’ordine è importante: la valutazione del test deve essere eseguita soltanto dopo aver selezionato la configurazione sulla validation.

---

# 9. Protezione degli artifact

I comandi principali controllano se gli output esistono già.

Per impostazione predefinita non sovrascrivono:

- feature estratte;
- modelli salvati;
- risultati di validation;
- selezione del modello;
- valutazione del test.

L’opzione:

```text
--overwrite
```

deve essere usata soltanto quando si intende riprodurre consapevolmente la stessa fase.

Per la valutazione del test è preferibile non utilizzarla, salvo la necessità di ripetere esattamente la medesima valutazione con la pipeline già selezionata.

---

# 10. Controlli di correttezza implementati

La pipeline verifica:

- presenza di train, validation e test;
- presenza delle colonne `text` e `label`;
- batch size positivi;
- archivi contenenti `features`, `labels` e `indices`;
- feature bidimensionali;
- label e indici monodimensionali;
- corrispondenza tra numero di feature e label;
- dimensione delle feature coerente tra gli split;
- presenza delle classi `0` e `1` nel train e nella validation;
- assenza di valori `NaN` o infiniti;
- mantenimento dell’ordinamento originale degli split;
- modello DistilBERT in `eval()`;
- zero parametri trainabili;
- assenza di gradienti durante l’estrazione;
- convergenza di `LinearSVC`;
- corrispondenza tra il `C` selezionato e il classificatore salvato;
- nessun refit prima della valutazione finale;
- salvataggio atomico tramite file temporanei.

Questi controlli riducono il rischio di errori silenziosi e rendono l’esperimento riproducibile.

---

# 11. Interpretazione scientifica

La baseline finale raggiunge:

```text
Test accuracy:  0,800188
Test macro-F1:  0,800148
```

Il risultato mostra che le rappresentazioni apprese da DistilBERT durante il pretraining contengono già informazione utile per distinguere sentiment positivo e negativo, anche senza aggiornare alcun parametro del Transformer.

Il contributo appreso sul dataset Rotten Tomatoes è limitato a:

- parametri di `StandardScaler`;
- coefficienti e intercetta di `LinearSVC`.

DistilBERT rimane invariato. La baseline misura quindi la qualità delle rappresentazioni pre-addestrate per questo compito.

Il miglioramento ottenuto passando da `C=1` a `C=0.01` è ridotto:

```text
Validation accuracy: +0,002815
Validation macro-F1: +0,002808
```

La conclusione corretta non è che `C=0.01` sia universalmente migliore, ma che risulta la configurazione migliore nella piccola griglia valutata su questa validation.

---

# 12. Limiti

La baseline presenta alcuni limiti intenzionali:

1. usa soltanto il primo token dell’ultimo layer;
2. non confronta mean pooling, max pooling o combinazioni di layer;
3. considera un solo classificatore, `LinearSVC`;
4. valuta una griglia ridotta di valori di `C`;
5. non esegue fine-tuning di DistilBERT;
6. non ripete la selezione con split o seed differenti;
7. non produce intervalli di confidenza;
8. non analizza qualitativamente i singoli errori;
9. non modifica la soglia decisionale dell’SVM;
10. usa il checkpoint uncased, quindi non conserva informazione sulle maiuscole.

Questi limiti sono coerenti con il ruolo dell’esercizio: costruire un punto di partenza semplice, stabile e riproducibile, non il miglior modello possibile.

---

# 13. Warning non bloccanti

Durante il caricamento può apparire:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

Il warning indica soltanto che non è stato configurato un token Hugging Face. Il caricamento continua normalmente, ma le richieste anonime possono avere limiti inferiori.

Può inoltre comparire un report con parametri `UNEXPECTED`, per esempio:

```text
vocab_transform.*
vocab_layer_norm.*
vocab_projector.*
```

Questi parametri appartengono alla testa di masked language modeling del checkpoint. Il progetto carica `DistilBertModel`, cioè il solo encoder base, quindi la testa linguistica non viene utilizzata. I forward riusciti e le shape corrette confermano il caricamento del componente necessario.

Il tokenizer concreto osservato è:

```text
BertTokenizer
```

Questo comportamento è compatibile con il checkpoint DistilBERT, che utilizza il vocabolario e la tokenizzazione WordPiece di BERT.

---

# 14. Stato finale

```text
Token-length preflight:       completato e verificato
Smoke test:                   completato e verificato
Feature extraction train:     completata e verificata
Feature extraction validation:completata e verificata
Feature extraction test:      completata e verificata
Baseline C=1:                 completata e verificata
Selezione di C:               completata e verificata
Valutazione finale del test:  completata e verificata
```

L’Esercizio 1.3 è quindi **completo** e costituisce la stable baseline da confrontare con il fine-tuning di DistilBERT nell’Esercizio 2.

---

## Risultato finale sintetico

```text
Dataset:              Cornell Rotten Tomatoes
Feature extractor:    distilbert/distilbert-base-uncased
Feature:              last_hidden_state[:, 0, :]
Feature dimension:    768
Transformer training: nessuno
Classifier:           StandardScaler + LinearSVC
C selezionato:        0.01
Validation accuracy:  0.820826
Validation macro-F1:  0.820742
Test accuracy:        0.800188
Test macro-F1:        0.800148
```

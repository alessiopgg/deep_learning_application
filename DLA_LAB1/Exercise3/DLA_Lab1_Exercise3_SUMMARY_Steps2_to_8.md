# DLA Lab 1 — Esercizio 3.3 Object Detection

## Riepilogo completo fino al Passo 8

**Progetto:** Deep Learning Applications — Laboratory 1  
**Approfondimento scelto:** Esercizio 3.3 — Object Detection  
**Dataset detection:** `keremberke/german-traffic-sign-detection`, configurazione `full`  
**Data di consolidamento:** 30 luglio 2026  
**Stato raggiunto:** pipeline dati completa e verificata; Faster R-CNN non ancora costruito

---

## 1. Scopo del documento

Questo documento serve come fonte autosufficiente per riprendere lo sviluppo dell’Esercizio 3.3 senza ricostruire ogni volta le conversazioni precedenti.

Contiene:

- obiettivo scientifico;
- stato effettivo del progetto;
- struttura corrente dei file;
- risultati reali dei Passi 2–8;
- contratti di dati, target e batch;
- decisioni progettuali consolidate;
- comandi di esecuzione;
- output prodotti;
- problemi aperti e rischi;
- punto esatto da cui ripartire.

Il documento distingue i dati realmente osservati dalle scelte ancora da implementare. Non contiene metriche di detection o risultati di training, perché il detector non è ancora stato costruito.

---

## 2. Contesto precedente

Gli Esercizi 1 e 2 hanno già realizzato:

- EDA del dataset GTSRB di classificazione;
- feature extraction con ResNet-18 e ResNet-50;
- classificatori LinearSVC, KNN e LDA;
- fine-tuning ResNet-18/50;
- strategie `classifier`, `last_block`, `full`;
- teste lineare e MLP;
- split train/validation stratificato;
- checkpoint, metriche e logging W&B nell’Esercizio 1;
- pipeline modulare e configurabile con OmegaConf nell’Esercizio 2;
- smoke test, training/validation e valutazione separata del test.

L’Esercizio 3.3 passa dalla classificazione di cartelli già ritagliati alla detection di cartelli all’interno di fotografie complete.

### Domanda scientifica principale

> Il fine-tuning della ResNet-50 sui cartelli ritagliati del GTSRB migliora la rilevazione e la classificazione dei cartelli nelle immagini complete rispetto a un’inizializzazione standard COCO/ImageNet?

Il confronto verrà affrontato dopo aver costruito e valutato una baseline Faster R-CNN.

---

## 3. Ambiente e dipendenze rilevanti

### Ambiente locale

- Windows 11;
- ambiente Conda: `DLA2026_clean`;
- Python 3.12;
- GPU locale: NVIDIA RTX 3050 Ti Laptop, 4 GB VRAM;
- sviluppo tramite PyCharm e file `.py`.

### Server disponibile

Per training più costosi è disponibile IRIS:

- Ubuntu 24.04;
- ambiente `DLA2026_server`;
- 2 × NVIDIA RTX 5090;
- selezione GPU con `CUDA_VISIBLE_DEVICES`;
- uso di `tmux`.

### Vincolo Hugging Face `datasets`

Il dataset detection usa ancora uno script di caricamento legacy. Le versioni recenti di `datasets` non lo eseguono più.

Versione consolidata:

```text
datasets==3.6.0
```

Il caricamento usa inoltre:

```python
revision="a549a284a1fefdc761ad459ee85f50c5ad8138ef"
trust_remote_code=True
```

La revisione è fissata per evitare di eseguire automaticamente una futura versione diversa dello script remoto.

Cache locale:

```text
DLA_LAB1/data/huggingface/
```

La cache non deve essere versionata.

---

## 4. Struttura corrente del progetto

Dopo il refactor effettuato e validato, la struttura Python dell’Esercizio 3 è:

```text
Exercise3/
├── __init__.py
├── main.py
├── paths.py
│
├── analysis/
│   ├── __init__.py
│   ├── eda.py
│   └── class_mapping.py
│
├── data_pipeline/
│   ├── __init__.py
│   ├── loading.py
│   ├── taxonomy.py
│   ├── adapter.py
│   ├── transforms.py
│   └── loaders.py
│
├── checks/
│   ├── __init__.py
│   ├── validate_adapter.py
│   ├── validate_transforms.py
│   ├── validate_loaders.py
│   └── validate_ground_truth.py
│
├── visualization/
│   ├── __init__.py
│   └── ground_truth.py
│
└── outputs/
    ├── step_2/
    ├── step_3/
    ├── step_4/
    ├── step_5/
    ├── step_6/
    ├── step_7/
    └── step_8/
```

### Responsabilità dei moduli

| Modulo | Responsabilità |
|---|---|
| `paths.py` | Percorsi centrali di `Exercise3`, progetto, dati e output |
| `data_pipeline/loading.py` | Download, cache e ispezione strutturale del dataset Hugging Face |
| `data_pipeline/taxonomy.py` | Classi GTSRB canoniche, background e mapping delle label |
| `data_pipeline/adapter.py` | Conversione Hugging Face → contratto PyTorch detection |
| `data_pipeline/transforms.py` | Trasformazioni sincronizzate minime |
| `data_pipeline/loaders.py` | Dataset finali, `collate_fn` e DataLoader |
| `analysis/eda.py` | Analisi statistica di immagini, box e classi |
| `analysis/class_mapping.py` | Verifica e salvataggio della mappatura detection ↔ GTSRB |
| `visualization/ground_truth.py` | Disegno e salvataggio delle ground truth |
| `checks/*` | Validazioni complete dei singoli passi |

### Modalità di esecuzione

Dopo il refactor, i comandi vengono lanciati dalla root `DLA_LAB1` tramite package Python:

```powershell
python -m Exercise3.analysis.eda
python -m Exercise3.analysis.class_mapping
python -m Exercise3.checks.validate_adapter
python -m Exercise3.checks.validate_transforms
python -m Exercise3.checks.validate_loaders
python -m Exercise3.checks.validate_ground_truth
```

Il refactor ha creato un backup locale:

```text
Exercise3_backup_before_refactor_20260730_170550
```

Il backup può essere eliminato soltanto dopo aver consolidato il nuovo assetto nel repository. Non deve essere aggiunto a Git.

---

# 5. Passo 2 — Caricamento e ispezione del dataset

## Dataset

```text
Repository:    keremberke/german-traffic-sign-detection
Configurazione: full
Classi:        43
Formato box:   COCO xywh
```

### Split reali osservati

| Split | Immagini |
|---|---:|
| Train | 383 |
| Validation | 108 |
| Test | 54 |
| **Totale** | **545** |

### Struttura di un campione

```python
sample = {
    "image_id": ...,
    "image": ...,
    "width": ...,
    "height": ...,
    "objects": {
        "id": [...],
        "area": [...],
        "bbox": [[x_min, y_min, width, height], ...],
        "category": [...],
    },
}
```

Ogni immagine può contenere un numero variabile di oggetti, inclusi zero oggetti.

### Output

```text
Exercise3/outputs/step_2/dataset_inspection.json
```

---

# 6. Passo 3 — EDA del dataset detection

## Conteggi generali

| Voce | Valore |
|---|---:|
| Immagini | 545 |
| Annotazioni originali | 852 |
| Immagini senza oggetti | 39 |
| Box valide | 852 |
| Box invalide | 0 |
| Box degeneri | 0 |
| Box fuori immagine | 0 |
| Categorie invalide | 0 |
| Area incoerente | 0 |
| Righe appartenenti a duplicati esatti | 2 |

Le due righe duplicate rappresentano un solo gruppo duplicato, quindi una sola copia ridondante da rimuovere.

## Conteggi per split

| Split | Immagini | Oggetti originali | Immagini vuote | Media oggetti/immagine |
|---|---:|---:|---:|---:|
| Train | 383 | 600 | 29 | 1,57 |
| Validation | 108 | 170 | 6 | 1,57 |
| Test | 54 | 82 | 4 | 1,52 |

Le immagini vuote sono state mantenute: costituiscono esempi di background utili al detector.

## Dimensioni delle immagini

Tutte le immagini hanno la stessa risoluzione:

```text
larghezza: 1360 px
altezza:    800 px
aspect ratio: 1,7
```

## Statistiche delle bounding box

| Statistica | Larghezza | Altezza |
|---|---:|---:|
| Minimo | 16 px | 16 px |
| 5° percentile | 20 px | 20 px |
| Mediana | 38 px | 37 px |
| Media | 43,40 px | 42,75 px |
| 95° percentile | 90,45 px | 89,45 px |
| Massimo | 127 px | 128 px |

Area relativa mediana rispetto all’immagine:

```text
circa 0,126%
```

La detection di oggetti piccoli sarà quindi uno dei problemi centrali.

## Distribuzione COCO delle scale

| Scala | Box | Percentuale approssimativa |
|---|---:|---:|
| Small, area < 32² | 315 | 37% |
| Medium, 32² ≤ area < 96² | 503 | 59% |
| Large, area ≥ 96² | 34 | 4% |

Per split:

```text
Train:       219 small, 355 medium, 26 large
Validation:   62 small, 102 medium,  6 large
Test:         34 small,  46 medium,  2 large
```

## Aspect ratio delle box

Le box sono quasi quadrate:

```text
mediana: 1,00
media:   1,013
5° percentile:  0,921
95° percentile: 1,134
```

Decisione consolidata: mantenere inizialmente le anchor standard di Faster R-CNN. Eventuali modifiche saranno motivate solo da metriche e recall sugli oggetti piccoli.

## Class imbalance

Nel train sono osservate 41 classi su 43.

Classe train più frequente:

```text
no overtaking -trucks-: 50 oggetti
```

Classe train meno frequente tra quelle presenti:

```text
pedestrian crossing: 1 oggetto
```

Rapporto massimo non nullo:

```text
50 : 1
```

### Classi assenti dal training

```text
animals
restriction ends
```

Situazione:

| Classe | Train | Validation | Test |
|---|---:|---:|---:|
| animals | 0 | 0 | 1 |
| restriction ends | 0 | 3 | 0 |

La classe `animals` compare nel test ma non ha esempi positivi nel train. La detection head non può apprenderla normalmente con gli split ufficiali.

Decisione consolidata:

- mantenere gli split ufficiali nella baseline;
- documentare il limite;
- riportare supporto e AP per classe;
- distinguere classi assenti dallo split di valutazione da classi con AP pari a zero;
- considerare eventualmente una metrica secondaria sulle sole classi supportate dal train.

## Duplicato esatto individuato

Il duplicato è nel train:

```text
sample_index: 238
image_id:     378
classe:       stop
bbox xywh:    [827, 543, 24, 24]
annotation IDs: 585 e 587
```

Politica adottata:

```text
stessa immagine + stessa categoria + stessa geometria esatta
→ conserva la prima annotazione
→ rimuovi soltanto le copie successive
```

Non vengono eliminate box solo perché hanno IoU elevata.

## Output EDA

```text
Exercise3/outputs/step_3/
├── eda_summary.json
├── tables/
│   ├── images.csv
│   ├── boxes.csv
│   ├── class_distribution.csv
│   ├── invalid_boxes.csv
│   ├── duplicate_boxes.csv
│   └── empty_images.csv
├── figures/
└── examples/
```

---

# 7. Passo 4 — Mappatura delle classi

Le 43 classi detection corrispondono semanticamente alle 43 classi GTSRB, ma:

- sono ordinate diversamente;
- alcuni nomi sono abbreviazioni;
- Faster R-CNN deve riservare la label 0 al background.

## Politica finale delle label

```text
0      = background
1–43   = classi nell’ordine canonico GTSRB
```

Formula:

```python
detector_label = gtsrb_class_id + 1
```

Costanti consolidate:

```python
BACKGROUND_LABEL = 0
NUM_GTSRB_CLASSES = 43
NUM_DETECTOR_CLASSES = 44
```

Non viene usato uno shift cieco dell’ID sorgente Hugging Face.

### Esempi

```text
animals
source detection ID: 0
GTSRB class ID:       31
detector label:       32
```

```text
restriction ends
source detection ID: 26
GTSRB class ID:       32
detector label:       33
```

La mappatura è una permutazione esplicita e biunivoca. Il codice fallisce se:

- il numero delle classi cambia;
- compare un nome inatteso;
- manca una classe prevista;
- due classi sorgente puntano allo stesso ID GTSRB;
- non vengono prodotte tutte le label foreground 1–43.

## Output

```text
Exercise3/outputs/step_4/
├── class_mapping.json
└── class_mapping.csv
```

---

# 8. Passo 5 — Dataset adapter PyTorch

La classe principale è:

```python
GermanTrafficSignDetectionDataset
```

Trasforma ogni campione Hugging Face in:

```python
image, target
```

con:

```python
target = {
    "boxes": boxes,
    "labels": labels,
    "image_id": image_id,
    "area": area,
    "iscrowd": iscrowd,
}
```

## Contratto prima delle trasformazioni

```text
image:
    tv_tensors.Image
    shape = [3, H, W]
    dtype = torch.uint8
    range = [0,255]

target["boxes"]:
    tv_tensors.BoundingBoxes
    shape = [N,4]
    dtype = torch.float32
    format = XYXY

target["labels"]:
    shape = [N]
    dtype = torch.int64
    valori = 1...43

target["image_id"]:
    intero univoco all’interno dello split

target["area"]:
    shape = [N]
    dtype = torch.float32

target["iscrowd"]:
    shape = [N]
    dtype = torch.int64
    tutti i valori = 0
```

## Conversione delle box

Sorgente:

```text
[x_min, y_min, width, height]
```

Output:

```text
[x_min, y_min, x_max, y_max]
```

con:

```python
x_max = x_min + width
y_max = y_min + height
```

L’area viene ricalcolata dalle coordinate finali.

## Risultati reali

| Split | Annotazioni sorgente | Annotazioni mantenute | Copie rimosse | Immagini vuote |
|---|---:|---:|---:|---:|
| Train | 600 | 599 | 1 | 29 |
| Validation | 170 | 170 | 0 | 6 |
| Test | 82 | 82 | 0 | 4 |
| **Totale** | **852** | **851** | **1** | **39** |

Tutte le 545 immagini e tutti i target hanno superato i controlli.

### Campione verificato

```text
validation[3]
image_id: 12
image shape: [3,800,1360]
box XYXY: [1120,444,1142,464]
label: 13
area: 440
iscrowd: 0
```

Verifica area:

```text
(1142 - 1120) × (464 - 444) = 22 × 20 = 440
```

## Output

```text
Exercise3/outputs/step_5/adapter_validation.json
```

---

# 9. Passo 6 — Trasformazioni minime

Pipeline attuale:

```text
torch.uint8 [0,255]
→ torch.float32 [0,1]
```

Non vengono applicati:

- resize esplicito;
- normalizzazione ImageNet esterna;
- augmentation geometriche;
- augmentation fotometriche.

Le box e i target restano invariati.

## Perché non viene applicata normalizzazione manuale ImageNet

Faster R-CNN di Torchvision contiene internamente `GeneralizedRCNNTransform`, che gestisce normalizzazione, resize e batching interno. La pipeline esterna deve fornire immagini `float32` nel range `[0,1]`.

## Risultati reali

| Split | Immagini float32 | Shape unica | Range | Label osservate |
|---|---:|---|---|---:|
| Train | 383 | `[3,800,1360]` | `[0,1]` | 41 |
| Validation | 108 | `[3,800,1360]` | `[0,1]` | 35 |
| Test | 54 | `[3,800,1360]` | `[0,1]` | 31 |

Tutti i contratti sono validi.

## Output

```text
Exercise3/outputs/step_6/transform_validation.json
```

---

# 10. Passo 7 — DataLoader e `collate_fn`

## Motivo della `collate_fn`

Ogni immagine contiene un numero diverso di box. Il batch non può essere rappresentato come un unico target `[B,N,4]` con `N` fisso.

Contratto del batch:

```python
images: list[torch.Tensor]
targets: list[dict]
```

La funzione:

```python
def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
```

non esegue:

- stacking delle immagini;
- padding;
- resize;
- trasferimento alla GPU;
- modifica dei target.

## Configurazione validata

```text
Train batch size:       2
Validation/Test batch:  1
Train shuffle:          True
Validation/Test shuffle: False
num_workers:            0
pin_memory:             True sul computer locale con CUDA
persistent_workers:     False
drop_last:              False
seed:                   42
```

## Risultati reali

| Split | Batch | Immagini | Batch size osservati | Oggetti | Vuote |
|---|---:|---:|---|---:|---:|
| Train | 192 | 383 | `[1,2]` | 599 | 29 |
| Validation | 108 | 108 | `[1]` | 170 | 6 |
| Test | 54 | 54 | `[1]` | 82 | 4 |

Il train produce:

```text
191 batch completi da 2
1 batch finale da 1
```

Nessuna immagine viene scartata.

## Riproducibilità dello shuffle

Due train DataLoader ricostruiti con seed 42 hanno prodotto lo stesso ordine iniziale:

```text
[90, 146, 269, 40, 74, 308, 307, 141, 27, 279, 216, 346]
```

Primo batch train:

```text
image IDs: [90,146]
shape: [[3,800,1360],[3,800,1360]]
box per immagine: [1,2]
```

## Output

```text
Exercise3/outputs/step_7/dataloader_validation.json
```

---

# 11. Passo 8 — Visualizzazione delle ground truth

La visualizzazione opera sulla pipeline finale, non sui dati grezzi.

Per ogni campione selezionato vengono salvati:

```text
originale + ground truth
trasformata + ground truth
confronto affiancato
```

Formato visualizzato:

```text
box assolute XYXY
colore ground truth: rosso
label: <detector_label>: <nome classe GTSRB>
```

## Selezione dei campioni

L’esecuzione reale ha usato quattro campioni per split:

1. immagine con più oggetti;
2. immagine con la box più piccola;
3. immagine vuota;
4. campione aggiuntivo deterministico.

### Campioni scelti

```text
Train:
  54  — most-objects, 6 oggetti
  355 — smallest-box, 3 oggetti
  100 — empty, 0 oggetti
  212 — seeded-extra, 2 oggetti

Validation:
  8  — most-objects, 4 oggetti
  98 — smallest-box, 2 oggetti
  35 — empty, 0 oggetti
  74 — seeded-extra, 2 oggetti

Test:
  15 — most-objects, 4 oggetti
  49 — smallest-box, 3 oggetti
  35 — empty, 0 oggetti
  44 — seeded-extra, 1 oggetto
```

## Controlli automatici

Sono state confrontate tutte le immagini prima e dopo le trasformazioni.

| Split | Immagini | Oggetti | Vuote | Errore massimo conversione pixel |
|---|---:|---:|---:|---:|
| Train | 383 | 599 | 29 | `5.96e-08` |
| Validation | 108 | 170 | 6 | `5.96e-08` |
| Test | 54 | 82 | 4 | `5.96e-08` |

Tolleranza:

```text
1e-07
```

Sono rimasti invariati:

- shape;
- `image_id`;
- box;
- label;
- area;
- `iscrowd`.

## Verifica visiva effettuata

Sono state controllate immagini rappresentative train, validation e test.

Conclusioni qualitative:

- le box circondano correttamente i cartelli;
- gli oggetti sovrapposti sono annotati separatamente;
- non risultano scambi fra coordinate o interpretazioni errate di `xywh`;
- originale e trasformata sono visivamente identiche;
- le classi mostrate sono plausibili;
- le immagini vuote non mostrano box artificiali.

Limite puramente grafico:

- il testo delle label è piccolo e talvolta sovrapposto.

Possibili miglioramenti futuri per README ed error analysis:

- box numerate con legenda laterale;
- crop ingranditi dei cartelli;
- immagine completa affiancata ai crop.

## Output

```text
Exercise3/outputs/step_8/
├── ground_truth_validation.json
└── examples/
    ├── train/
    ├── validation/
    └── test/
```

---

# 12. Pipeline dati consolidata

Il flusso completo attuale è:

```text
Hugging Face Dataset
→ caricamento con datasets 3.6.0
→ lettura delle annotazioni COCO xywh
→ verifica schema e classi
→ rimozione di una sola copia duplicata esatta
→ conversione xywh → XYXY
→ mapping source category → GTSRB ID → detector label 1–43
→ costruzione target PyTorch
→ tv_tensors.Image uint8
→ conversione immagine float32 [0,1]
→ TransformedDetectionDataset
→ DataLoader con detection_collate_fn
→ list[Tensor], list[target]
```

## Contratto definitivo del campione

```python
image:
    Tensor o tv_tensors.Image
    shape = [3,H,W]
    dtype = torch.float32
    range = [0,1]

target = {
    "boxes": BoundingBoxes[N,4],   # float32, XYXY
    "labels": Tensor[N],           # int64, 1...43
    "image_id": int,
    "area": Tensor[N],             # float32
    "iscrowd": Tensor[N],          # int64, tutti 0
}
```

## Contratto definitivo del batch

```python
images: list[Tensor]
targets: list[dict]
```

Il trasferimento al device non è responsabilità del DataLoader. Verrà effettuato nel training/smoke test:

```python
images = [image.to(device, non_blocking=True) for image in images]
```

---

# 13. Decisioni progettuali consolidate

## Dataset

- mantenere gli split ufficiali;
- non usare il test per scegliere configurazioni o soglie;
- mantenere le 39 immagini vuote;
- rimuovere soltanto copie duplicate esatte;
- non effettuare altre correzioni delle box;
- conservare gli ID sorgente e una mappatura esplicita.

## Classi

- background = 0;
- classi foreground = 1–43;
- ordine foreground = ordine canonico GTSRB;
- `num_classes` Faster R-CNN = 44;
- segnalare le classi assenti dal train;
- associare sempre le metriche per classe al supporto reale.

## Box

- sorgente `xywh`;
- contratto PyTorch `XYXY`;
- coordinate assolute;
- dtype `float32`;
- area positiva e ricalcolata;
- box dentro il canvas.

## Trasformazioni

- baseline iniziale senza augmentation;
- niente horizontal flip automatico, perché può alterare il significato dei cartelli;
- niente crop aggressivi;
- niente normalizzazione ImageNet manuale esterna;
- niente resize esplicito prima del detector;
- qualunque futura trasformazione geometrica dovrà aggiornare le box in modo sincronizzato.

## DataLoader

- `collate_fn` dedicata;
- train shuffle attivo e riproducibile;
- validation/test senza shuffle;
- `drop_last=False`;
- default Windows sicuro `num_workers=0`;
- batch size da rivalutare nello smoke test e sul server.

## Anchor

- mantenere le anchor standard nella prima baseline;
- non modificarle senza evidenze quantitative;
- valutare successivamente recall e AP sugli oggetti small.

---

# 14. Problemi e rischi ancora aperti

## Dataset e valutazione

1. `animals` non è presente nel train ma compare nel test.
2. `restriction ends` non è presente nel train ma compare in validation.
3. Solo 31 classi sono osservate nel test.
4. Il dataset è piccolo e fortemente sbilanciato.
5. Molti oggetti sono piccoli rispetto all’immagine completa.
6. Le metriche per classe saranno instabili per classi con uno o pochi esempi.

## Modello

1. Faster R-CNN non è ancora stato costruito.
2. Non è ancora stato verificato il consumo di VRAM sulla RTX 3050 Ti.
3. Non è ancora stato deciso il `min_size` interno del detector.
4. Non è ancora stato verificato il comportamento del detector con immagini senza oggetti.
5. Non è ancora stato verificato il checkpoint ResNet-50 GTSRB da trasferire.
6. Non è ancora stata costruita la logica di caricamento controllato dello `state_dict`.
7. Non sono ancora presenti optimizer, scheduler, checkpoint detection, metriche o W&B per Exercise3.

## Visualizzazione

- label piccole e sovrapposte nelle scene con molti cartelli;
- per il report finale saranno preferibili crop ingranditi o box numerate.

## Console

In alcuni output incollati sono comparse righe duplicate. I JSON risultano corretti e non mostrano duplicazioni strutturali. Sembra un problema di copia o stampa, non dei dati.

---

# 15. Comandi consolidati

Dalla root:

```text
C:\Users\Alessio\pythonProject\DLA_LAB1
```

## Ispezione dataset

```powershell
python -m Exercise3.main --split validation --sample-index 3
```

## EDA

```powershell
python -m Exercise3.analysis.eda
```

## Mappatura classi

```powershell
python -m Exercise3.analysis.class_mapping
```

## Validazione adapter

```powershell
python -m Exercise3.checks.validate_adapter
```

## Validazione trasformazioni

```powershell
python -m Exercise3.checks.validate_transforms
```

## Validazione DataLoader

```powershell
python -m Exercise3.checks.validate_loaders
```

## Visualizzazione ground truth

```powershell
python -m Exercise3.checks.validate_ground_truth
```

Con quattro esempi per split:

```powershell
python -m Exercise3.checks.validate_ground_truth --samples-per-split 4
```

---

# 16. Stato dei passi

| Passo | Stato |
|---|---|
| 0 — Audit progetto | volutamente tralasciato nella fase iniziale |
| 1 — Dipendenze | gestito progressivamente; `datasets==3.6.0` consolidato |
| 2 — Caricamento dataset | completato e verificato |
| 3 — EDA detection | completato e analizzato |
| 4 — Mappatura classi | completato e verificato |
| 5 — Dataset adapter | completato e verificato su 545 immagini |
| 6 — Trasformazioni | completato e verificato |
| 6.5 — Refactor package | completato, compilato e validato |
| 7 — DataLoader | completato e verificato |
| 8 — Ground truth visualization | completato automaticamente e visivamente |
| 9 — Baseline Faster R-CNN | **prossimo passo** |
| 10 — Smoke test | non iniziato |
| 11 — Training baseline | non iniziato |
| 12 — Valutazione baseline | non iniziato |
| 13 — Predizioni qualitative | non iniziato |
| 14 — Backbone GTSRB | non iniziato |
| 15–20 | non iniziati |

---

# 17. Prossimo passo — Passo 9

Il prossimo obiettivo è costruire una baseline Faster R-CNN con:

- modello Torchvision pre-addestrato;
- nuova testa per 44 classi complessive;
- backbone inizialmente congelato;
- teste detection trainabili;
- configurazione esplicita;
- conteggio dei parametri totali e trainabili;
- elenco dei moduli congelati e addestrabili;
- stampa del device;
- controllo della memoria GPU.

Non verrà ancora eseguito un training lungo.

Dopo la costruzione del modello, il Passo 10 dovrà verificare:

```text
forward in train mode
→ dizionario delle quattro loss
→ somma delle loss
→ backward
→ optimizer step
→ inferenza in eval mode
→ boxes, labels, scores
→ controllo NaN/Inf
```

Solo dopo uno smoke test riuscito sarà opportuno implementare il training della baseline.

---

# 18. Matrice sperimentale futura

La matrice prevista rimane:

| Run | Inizializzazione backbone | Parti addestrabili |
|---|---|---|
| A | Faster R-CNN COCO/ImageNet | teste detection |
| B | ResNet-50 GTSRB fine-tuned | teste detection |
| C | ResNet-50 GTSRB fine-tuned | teste + `layer4` |
| D | ResNet-50 GTSRB fine-tuned | teste + `layer3` + `layer4` |

Tutte le run dovranno usare, salvo esperimenti esplicitamente separati:

- stessi split;
- stesso seed;
- stessa pipeline dati;
- stesso protocollo di valutazione;
- stessa risoluzione/configurazione del detector;
- stesse epoche;
- stessi criteri di checkpoint;
- metriche e supporti salvati;
- tempi e memoria GPU registrati.

Il confronto B vs A risponde direttamente alla domanda scientifica principale.

---

# 19. Metriche future da implementare

La dicitura della consegna “accuracy @ IoU=0.5” è ambigua per la detection. Verranno usate metriche standard:

- precision a IoU 0.5;
- recall a IoU 0.5;
- AP@0.5;
- mAP@0.5;
- mAP@[0.5:0.95];
- AP per classe;
- supporto per classe;
- metriche small/medium/large quando disponibili.

La validation servirà per scegliere checkpoint e soglie. Il test verrà valutato separatamente.

---

# 20. Elementi da documentare nel README finale

- scelta dell’Esercizio 3.3;
- domanda scientifica;
- uso del dataset detection Hugging Face;
- necessità di `datasets==3.6.0`;
- revisione fissata dello script remoto;
- struttura della tassonomia e background;
- classi assenti dal train;
- distribuzione small/medium/large;
- deduplicazione esatta;
- pipeline dati e contratti;
- struttura modulare del package;
- visualizzazioni ground truth;
- configurazioni delle run;
- metriche reali;
- limiti del dataset;
- AI Assistance Disclosure;
- References and External Resources.

---

# 21. Fonti di riferimento del progetto

- notebook originale `DLA-Lab1(2).ipynb`;
- riepilogo `DLA_Lab1_SUMMARY_Exercise1.1_to_2(1).md`;
- repository `https://github.com/alessiopgg/deep_learning_application.git`;
- output reali `eda_summary.json`;
- output reali `adapter_validation.json`;
- output reali `transform_validation.json`;
- output reali `dataloader_validation.json`;
- output reali `ground_truth_validation.json`;
- immagini di confronto ground truth train/validation/test.

---

# 22. Sintesi operativa finale

La preparazione dei dati per l’object detection è completa:

```text
dataset caricato
→ struttura verificata
→ EDA completata
→ tassonomia verificata
→ background aggiunto
→ mapping biunivoco costruito
→ box convertite in XYXY
→ duplicato esatto rimosso
→ target PyTorch costruiti
→ immagini convertite in float32 [0,1]
→ DataLoader con collate_fn
→ ground truth verificate automaticamente
→ ground truth verificate visivamente
```

Il progetto dispone ora di una base dati pulita, modulare, riproducibile e compatibile con Torchvision detection.

**Punto esatto da cui ripartire:** costruzione della baseline Faster R-CNN del Passo 9, senza ancora avviare training lunghi.

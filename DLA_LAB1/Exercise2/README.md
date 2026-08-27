# Exercise 2 — Pipeline configurabile e riproducibile

L'Exercise 2 consolida il fine-tuning sviluppato nell'Exercise 1.3 in una pipeline più **modulare, configurabile e riproducibile**.

Il problema scientifico non cambia: classificazione GTSRB con ResNet pre-addestrate, testa lineare o MLP e strategie `classifier`, `last_block` e `full`. Il contributo dell'esercizio è soprattutto ingegneristico: separare configurazione, dati, modello, training, checkpoint e valutazione finale del test.

```text
Exercise 1.3
pipeline sperimentale funzionante
        ↓
Exercise 2
configurazione esplicita
+ moduli separati
+ smoke test
+ metriche train/validation
+ checkpoint configurabile
+ test ufficiale separato
```

---

## Struttura

```text
Exercise2/
├── configuration.py
├── data.py
├── models.py
├── training.py
├── main.py
├── evaluate_test.py
├── configs/
│   └── default.yaml
├── assets/
│   └── per_class_f1.svg
└── outputs/
```

| File | Responsabilità |
|---|---|
| `configuration.py` | caricamento YAML, override OmegaConf e validazione |
| `data.py` | split GTSRB e DataLoader |
| `models.py` | costruzione ResNet, testa e policy di fine-tuning |
| `training.py` | runtime, loss, optimizer, train/validation e checkpoint |
| `main.py` | entry point per smoke test e training |
| `evaluate_test.py` | valutazione separata sul test ufficiale |

La versione finale accorpa le piccole utility nei moduli che ne utilizzano direttamente la logica, mantenendo la separazione delle responsabilità senza frammentare inutilmente la pipeline.

---

## Configurazione

I valori di default sono definiti in `configs/default.yaml`.

| Componente | Default |
|---|---|
| Backbone | `resnet18` |
| Testa | `linear` |
| Fine-tuning | `last_block` |
| Batch size | 32 |
| Epoche | 5 |
| Loss | Cross Entropy |
| Optimizer | AdamW |
| LR backbone | `1e-4` |
| LR testa | `1e-3` |
| Weight decay | `1e-4` |
| Validation split | 20% stratificato |
| Checkpoint | minima validation loss |
| Seed | 42 |
| Device | `auto` |

OmegaConf permette di modificare soltanto i parametri necessari:

```bash
python Exercise2/main.py \
  model.name=resnet50 \
  data.batch_size=16 \
  model.classifier_type=mlp \
  model.fine_tuning_strategy=classifier \
  training.epochs=10
```

La configurazione viene validata prima dell'esecuzione. In particolare, criterio e modalità del checkpoint devono essere coerenti:

```text
validation_loss      -> min
validation_accuracy  -> max
validation_macro_f1  -> max
```

---

## Dati e modello

Il protocollo resta coerente con l'Exercise 1.3:

| Split | Immagini |
|---|---:|
| Training interno | 21.312 |
| Validation | 5.328 |
| Test ufficiale | 12.630 |
| Classi | 43 |

Lo split train/validation è stratificato 80/20 con seed `42`.

Il preprocessing deriva direttamente dai pesi Torchvision tramite `weights.transforms()`. Sono supportati:

- ResNet-18 con `IMAGENET1K_V1`;
- ResNet-50 con `IMAGENET1K_V2`;
- testa `linear`;
- testa `mlp`;
- strategie `classifier`, `last_block`, `full`.

Durante il selective fine-tuning, gli stadi BatchNorm congelati rimangono in modalità `eval()` per evitare l'aggiornamento involontario delle running statistics.

---

## Training e checkpoint

Training e validation calcolano:

- loss;
- accuracy;
- macro-F1;
- numero di campioni e batch processati;
- tempo di esecuzione.

L'ottimizzazione usa `CrossEntropyLoss` e AdamW con learning rate differenziati:

```text
backbone    1e-4
classifier  1e-3
```

Il miglior checkpoint può essere selezionato tramite:

- `validation_loss`;
- `validation_accuracy`;
- `validation_macro_f1`.

Il checkpoint contiene:

- best epoch;
- metrica monitorata;
- valore della metrica;
- `model_state_dict`;
- `optimizer_state_dict`;
- configurazione risolta.

Prima del salvataggio i tensori vengono spostati su CPU. La scrittura è atomica: viene creato prima un file temporaneo `best_model.pt.tmp`, poi sostituito con `best_model.pt`.

Al termine del training il best checkpoint viene ricaricato automaticamente.

---

## Smoke test

Prima di una run completa è possibile verificare l'intera pipeline su pochi batch reali:

```bash
python Exercise2/main.py experiment.smoke_test_batches=2
```

Lo smoke test esegue:

```text
dati
 ↓
forward
 ↓
loss
 ↓
backward
 ↓
optimizer step
 ↓
validation
```

e termina senza avviare il training multi-epoca o creare una run completa.

---

## Valutazione separata del test

Il test ufficiale viene valutato esplicitamente a partire dal best checkpoint:

```bash
python Exercise2/evaluate_test.py \
  --checkpoint Exercise2/outputs/runs/<run_id>/best_model.pt
```

È possibile forzare il device:

```bash
python Exercise2/evaluate_test.py \
  --checkpoint Exercise2/outputs/runs/<run_id>/best_model.pt \
  --device cuda:0
```

Lo script legge la configurazione salvata nel checkpoint, ricostruisce preprocessing, DataLoader e modello e produce:

```text
test_metrics.json
classification_report.json
predictions.npz
```

Questa separazione impedisce che il test ufficiale partecipi alla selezione del modello.

---

## Run di riferimento

L'Exercise 2 è stato verificato con una singola run completa:

| Componente | Valore |
|---|---|
| Backbone | ResNet-18 |
| Strategia | `last_block` |
| Testa | MLP |
| Epoche | 10 |
| Batch size | 32 |
| Checkpoint monitor | `validation_macro_f1` |
| Checkpoint mode | `max` |
| Seed | 42 |
| Best epoch | 10 |

Risultati:

| Metrica | Valore |
|---|---:|
| Best validation macro-F1 | **0.9972** |
| Test loss | **0.1416** |
| Test accuracy | **0.9645** |
| Test macro-F1 | **0.9590** |
| Test samples | 12.630 |
| Test evaluation time | 9.87 s |

Il gap tra validation macro-F1 e test macro-F1 è di circa **3,8 punti percentuali**, indicando che il test ufficiale è sensibilmente più difficile dello split interno pur mantenendo prestazioni elevate.

Questa run serve come **verifica end-to-end della pipeline** e non come dimostrazione della superiorità di `last_block` o della testa MLP rispetto alle alternative.

<p align="center">
  <img src="assets/per_class_f1.svg"
       alt="F1-score per classe sul test ufficiale"
       width="950">
</p>

La maggior parte delle classi raggiunge F1 elevati. Le principali criticità sono:

- classe **22**: F1 `0.8585`, soprattutto per recall basso;
- classe **29**: F1 `0.8586`, con precision più bassa;
- classe **28**: F1 `0.9058`, con recall quasi perfetto ma precision inferiore.

La classe 17 viene invece classificata correttamente su tutti i 360 esempi del test.

---

## Riproduzione

L'ambiente condiviso del Laboratorio 1 è definito in [`../environment.yml`](../environment.yml).

Dalla root del repository:

```bash
conda env create -f DLA_LAB1/environment.yml
conda activate DLA2026_clean
cd DLA_LAB1
```

### Smoke test

```bash
python Exercise2/main.py experiment.smoke_test_batches=2
```

### Training con configurazione di default

```bash
python Exercise2/main.py
```

### Esempio con override

```bash
python Exercise2/main.py \
  model.name=resnet18 \
  model.classifier_type=mlp \
  model.fine_tuning_strategy=last_block \
  training.epochs=10 \
  checkpoint.monitor=validation_macro_f1 \
  checkpoint.mode=max
```

### Valutazione finale

```bash
python Exercise2/evaluate_test.py \
  --checkpoint Exercise2/outputs/runs/<run_id>/best_model.pt
```

---

## Output e riproducibilità

Una run completa genera una directory univoca:

```text
Exercise2/outputs/runs/
└── <timestamp>_<model>-<strategy>-<classifier>/
    ├── best_model.pt
    ├── test_metrics.json
    ├── classification_report.json
    └── predictions.npz
```

Il progetto controlla:

- seed Python, NumPy e PyTorch;
- seed dei DataLoader;
- split stratificato;
- configurazione salvata nel checkpoint;
- selezione del best checkpoint sulla validation;
- modalità deterministica opzionale tramite `experiment.deterministic=true`.

Con `deterministic=false`, che è il default, non viene assunto determinismo bit-a-bit delle operazioni GPU.

L'Exercise 2 non utilizza W&B nella versione finale: il tracking è locale tramite checkpoint e artifact della run.

---

## Limiti

- È stata completata una sola run di riferimento: non è una nuova campagna di confronto tra configurazioni.
- La run di riferimento usa un solo seed.
- La validation interna risulta più semplice del test ufficiale.
- La history delle epoche viene mantenuta durante il training, ma la versione corrente non la salva automaticamente in CSV o JSON.
- Il test deve essere eseguito esplicitamente dopo il training; la separazione è intenzionale e fa parte del protocollo sperimentale.

---

## Riferimenti e assistenza AI

Riferimenti principali:

- GTSRB / `torchvision.datasets.GTSRB`;
- ResNet-18 e ResNet-50 pre-addestrate di Torchvision;
- PyTorch e Torchvision;
- OmegaConf;
- Scikit-learn.

ChatGPT è stato utilizzato come supporto per chiarimenti teorici, organizzazione e consolidamento della pipeline, revisione del codice, debugging e documentazione. Le proposte generate sono state verificate e adattate; le metriche riportate derivano dagli artifact reali della run eseguita.

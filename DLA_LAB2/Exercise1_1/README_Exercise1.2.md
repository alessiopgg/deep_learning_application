# Exercise 1 — Rotten Tomatoes e DistilBERT

## Deep Learning Applications — Laboratorio 2

Questo esercizio introduce l’elaborazione di testo con Hugging Face Transformers usando il dataset Cornell Rotten Tomatoes e il checkpoint:

```text
distilbert/distilbert-base-uncased
```

Il lavoro è organizzato in modo modulare e mantiene separati caricamento del dataset, analisi esplorativa, ispezione del tokenizer, ispezione del modello Transformer e coordinamento tramite CLI.

L’ambiente Conda usato localmente è:

```text
DLA2026-transformers
```

---

## Struttura dei file

```text
Exercise1/
├── main.py
├── data.py
├── eda.py
├── transformer_inspection.py
└── outputs/
    └── exercise_1_1/
        ├── figures/
        └── results/
```

Responsabilità principali:

- `data.py`: caricamento del dataset e recupero degli split;
- `eda.py`: analisi esplorativa dell’Esercizio 1.1;
- `transformer_inspection.py`: caricamento e ispezione di tokenizer e DistilBERT;
- `main.py`: entry point CLI per i diversi esperimenti.

---

# Esercizio 1.1 — Dataset exploration

Il dataset utilizzato è:

```text
cornell-movie-review-data/rotten_tomatoes
```

## Split

| Split | Esempi |
|---|---:|
| Train | 8.530 |
| Validation | 1.066 |
| Test | 1.066 |
| Totale | 10.662 |

## Schema

```text
text: string
label: ClassLabel
```

Mapping delle classi:

```text
0 → neg
1 → pos
```

## Distribuzione delle classi

Il dataset è perfettamente bilanciato.

| Split | Negative | Positive |
|---|---:|---:|
| Train | 4.265 | 4.265 |
| Validation | 533 | 533 |
| Test | 533 | 533 |

## Controlli di integrità

L’esecuzione reale ha verificato:

- testi mancanti: `0`;
- testi non stringa: `0`;
- testi vuoti: `0`;
- label non valide: `0`;
- duplicati esatti interni agli split: `0`;
- overlap train-validation: `0`;
- overlap train-test: `0`;
- overlap validation-test: `0`.

## Lunghezza delle frasi in parole

| Split | Media | Mediana | 95° percentile | Massimo |
|---|---:|---:|---:|---:|
| Train | 20,99 | 20 | 37 | 59 |
| Validation | 21,00 | 21 | 38 | 54 |
| Test | 21,22 | 20 | 38 | 52 |

Queste statistiche sono basate sulle parole separate tramite spazi. Non descrivono ancora la lunghezza secondo il tokenizer DistilBERT.

## Comando

Dalla root `DLA_LAB2`:

```powershell
python Exercise1/main.py eda
```

---

# Esercizio 1.2 — Pre-trained BERT and Tokenizer

## Obiettivo

L’obiettivo dell’Esercizio 1.2 è comprendere concretamente il passaggio:

```text
testo grezzo
→ tokenizer
→ token e subword
→ input_ids e attention_mask
→ tensori PyTorch
→ DistilBERT
→ last_hidden_state
→ rappresentazioni contestuali
```

In questa fase non vengono eseguiti training, fine-tuning, classificazione positiva/negativa, calcolo della loss, estrazione delle feature di tutti gli split o valutazione tramite accuracy e F1.

---

## Componenti caricate

Le componenti vengono caricate con:

```python
AutoTokenizer.from_pretrained(
    "distilbert/distilbert-base-uncased"
)

AutoModel.from_pretrained(
    "distilbert/distilbert-base-uncased"
)
```

L’esecuzione reale ha prodotto:

```text
Tokenizer type: BertTokenizer
Model type: DistilBertModel
Model device: cpu
Model training mode: False
```

`AutoTokenizer` seleziona automaticamente un tokenizer compatibile con il checkpoint.

`AutoModel` carica l’encoder DistilBERT base, senza una testa di classificazione.

---

# Ispezione di una singola frase

## Comando

```powershell
python Exercise1/main.py inspect-transformer
```

## Esempio reale selezionato

```text
Split: train
Example index: 0
Label: 1
```

Testo:

```text
the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .
```

La label `1` corrisponde alla classe positiva.

## Token e subword

Il tokenizer ha prodotto:

```text
45 token prima dei token speciali
47 token complessivi inviati al modello
```

I due token aggiunti sono:

```text
[CLS]
[SEP]
```

Alcune parole vengono divise in subword:

```text
schwarzenegger → schwarz + ##ene + ##gger
claud          → cl + ##aud
damme          → dam + ##me
segal          → sega + ##l
```

Il prefisso `##` indica che il frammento continua la subword precedente.

Questo mostra che:

```text
numero di parole ≠ numero di token
```

Anche apostrofi, virgolette, virgole, trattini e punti occupano posizioni nella sequenza.

## `input_ids`

Output reale:

```text
Shape: (1, 47)
Dtype: torch.int64
Device: cpu
```

Forma simbolica:

```text
[batch_size, sequence_length]
```

Nel caso eseguito:

```text
[1, 47]
```

Significato:

- `1`: una frase nel batch;
- `47`: token della sequenza, inclusi `[CLS]` e `[SEP]`.

Alcuni ID rilevanti:

```text
[CLS] → 101
[SEP] → 102
```

Gli `input_ids` sono indici interi del vocabolario. Non sono ancora rappresentazioni semantiche.

## `attention_mask`

Output reale:

```text
Shape: (1, 47)
Dtype: torch.int64
Device: cpu
```

Valori:

```text
[[1, 1, 1, ..., 1]]
```

Interpretazione:

```text
1 → token reale
0 → posizione di padding
```

Nella prova con una sola frase non è stato necessario introdurre padding, quindi la maschera contiene soltanto `1`.

---

# Forward pass in DistilBERT

Il modello viene usato in inferenza:

```python
model.eval()

with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
```

`model.eval()` imposta il modello in modalità valutazione.

`torch.inference_mode()` disattiva il tracciamento dei gradienti.

L’esecuzione ha infatti verificato:

```text
Requires gradient: False
```

Non vengono eseguiti `backward()`, aggiornamento dei pesi o optimizer step.

## Output del modello

L’output reale è:

```text
Output type: BaseModelOutput
Output fields: ['last_hidden_state']
```

`AutoModel` non produce direttamente logits, classe predetta, probabilità o loss. Produce invece le rappresentazioni interne dell’encoder.

## `last_hidden_state`

Shape reale:

```text
(1, 47, 768)
```

Forma simbolica:

```text
[batch_size, sequence_length, hidden_size]
```

Interpretazione:

- `1`: frase nel batch;
- `47`: posizioni della sequenza;
- `768`: dimensione della rappresentazione di ogni token.

DistilBERT restituisce quindi un vettore di 768 valori per ogni posizione:

```text
[CLS] → vettore di 768 valori
the   → vettore di 768 valori
rock  → vettore di 768 valori
...
[SEP] → vettore di 768 valori
```

Per questa frase il tensore contiene:

```text
47 × 768 = 36.096 valori
```

senza considerare la dimensione esterna del batch.

## Rappresentazioni contestuali

Gli `input_ids` sono identificatori fissi nel vocabolario. Le rappresentazioni finali dipendono invece dall’intera frase.

La parola `rock`, per esempio, può avere lo stesso ID in frasi diverse, ma il suo vettore finale può cambiare perché DistilBERT considera il contesto circostante.

```text
input_id
→ indice fisso nel vocabolario

hidden state
→ rappresentazione dipendente dal contesto
```

## Rappresentazione del primo token

È stata estratta tramite:

```python
first_token_representation = (
    last_hidden_state[:, 0, :]
)
```

La posizione `0` corrisponde a `[CLS]`.

Shape reale:

```text
(1, 768)
```

Trasformazione delle shape:

```text
last_hidden_state
[1, 47, 768]

selezione della posizione 0
↓

first_token_representation
[1, 768]
```

In termini concreti:

```text
una frase
→ un vettore di 768 componenti
```

Questa rappresentazione verrà usata successivamente come feature globale della frase nell’Esercizio 1.3.

---

# Ispezione di un batch con padding dinamico

## Comando

```powershell
python Exercise1/main.py inspect-transformer-batch
```

Sono stati selezionati due esempi reali con lunghezze molto diverse.

## Esempio corto

```text
Dataset index: 26
Label: 1
Text: spiderman rocks
```

Token prima dei token speciali:

```text
spider
##man
rocks
```

Conteggio:

```text
3 token testuali
+ [CLS]
+ [SEP]
= 5 token reali
```

## Esempio lungo

```text
Dataset index: 38
Label: 1
```

Il tokenizer ha prodotto:

```text
52 token testuali
+ [CLS]
+ [SEP]
= 54 token reali
```

---

# Padding dinamico

Le due sequenze hanno lunghezze reali:

```text
5
54
```

Con:

```python
padding=True
```

entrambe vengono portate alla lunghezza della frase più lunga nel batch:

```text
54
```

La frase corta diventa:

```text
[CLS] spider ##man rocks [SEP] [PAD] ... [PAD]
```

Conteggio:

```text
5 token reali
49 token di padding
```

La frase lunga contiene:

```text
54 token reali
0 token di padding
```

Il padding è quindi dinamico rispetto al batch corrente. Le sequenze non vengono automaticamente portate a 512 token.

## Batch di `input_ids`

Shape reale:

```text
(2, 54)
```

Interpretazione:

- `2`: due frasi;
- `54`: lunghezza comune dopo il padding.

Il tokenizer ha confermato:

```text
Padding token: [PAD]
Padding token ID: 0
```

Nella prima riga, dopo `[SEP]`, gli `input_ids` assumono quindi valore `0`.

## Batch di `attention_mask`

Shape reale:

```text
(2, 54)
```

Prima frase:

```text
[1, 1, 1, 1, 1, 0, 0, ..., 0]
```

Seconda frase:

```text
[1, 1, 1, ..., 1]
```

Controllo reale:

```text
Batch position 0:
5 real tokens
49 padding tokens

Batch position 1:
54 real tokens
0 padding tokens
```

L’`attention_mask` permette al modello di distinguere il testo reale dalle posizioni aggiunte soltanto per rendere rettangolare il batch.

## Output di DistilBERT per il batch

Shape reale:

```text
last_hidden_state:
(2, 54, 768)
```

Forma simbolica:

```text
[batch_size, padded_sequence_length, hidden_size]
```

Interpretazione:

- `2`: frasi nel batch;
- `54`: posizioni dopo il padding;
- `768`: dimensione della rappresentazione per posizione.

Le rappresentazioni dei primi token hanno shape:

```text
(2, 768)
```

Quindi:

```text
prima frase   → vettore [768]
seconda frase → vettore [768]
```

In generale:

```text
B frasi
→ matrice [B, 768]
```

DistilBERT restituisce formalmente vettori anche nelle posizioni `[PAD]`, perché il tensore deve mantenere una forma rettangolare. Queste posizioni non devono però essere trattate come testo reale.

---

# Shape riepilogative

## Singola frase

```text
input_ids:
[1, L]

attention_mask:
[1, L]

last_hidden_state:
[1, L, 768]

first_token_representation:
[1, 768]
```

Nell’esecuzione reale:

```text
input_ids:                  [1, 47]
attention_mask:             [1, 47]
last_hidden_state:          [1, 47, 768]
first_token_representation: [1, 768]
```

## Batch di due frasi

```text
input_ids:
[2, Lmax]

attention_mask:
[2, Lmax]

last_hidden_state:
[2, Lmax, 768]

first_token_representations:
[2, 768]
```

Nell’esecuzione reale:

```text
input_ids:                   [2, 54]
attention_mask:              [2, 54]
last_hidden_state:           [2, 54, 768]
first_token_representations: [2, 768]
```

---

# Warning osservati

## Richieste non autenticate al Hugging Face Hub

Il messaggio relativo a `HF_TOKEN` non ha impedito il download o l’esecuzione. Il checkpoint è pubblico e il modello è stato caricato correttamente.

## Cache e symlink su Windows

Il warning sui symlink riguarda l’efficienza della cache locale. I file vengono comunque salvati, ma possono occupare più spazio su disco. Non modifica il comportamento del modello.

## Pesi `UNEXPECTED`

Durante il caricamento sono comparsi parametri come:

```text
vocab_transform
vocab_layer_norm
vocab_projector
```

Questi appartengono alla testa di masked language modeling presente nel checkpoint originale.

`AutoModel` carica soltanto il corpo encoder `DistilBertModel`, quindi la testa di pretraining viene ignorata.

L’encoder necessario è stato caricato correttamente, come dimostrano i forward pass riusciti e le shape ottenute.

---

# Risultati verificati

L’Esercizio 1.2 ha verificato realmente:

- caricamento di `AutoTokenizer`;
- caricamento di `AutoModel`;
- uso di esempi reali del dataset;
- tokenizzazione in token e subword;
- aggiunta di `[CLS]` e `[SEP]`;
- costruzione degli `input_ids`;
- costruzione dell’`attention_mask`;
- conversione in tensori PyTorch;
- esecuzione del forward pass;
- uso di `torch.inference_mode()`;
- ispezione di `BaseModelOutput`;
- ispezione di `last_hidden_state`;
- estrazione della rappresentazione del primo token;
- differenza tra frase singola e batch;
- padding dinamico;
- interpretazione delle shape;
- assenza intenzionale di una testa di classificazione.

---

# Cosa non è stato ancora fatto

Non sono ancora stati eseguiti:

- classificatore SVM;
- estrazione completa delle feature;
- elaborazione di tutti gli split;
- accuracy;
- F1-score;
- classification report;
- selezione sulla validation;
- valutazione del test;
- fine-tuning;
- uso di `Trainer`;
- misurazione globale delle lunghezze tokenizzate;
- scelta di `max_length`;
- truncation;
- uso della GPU.

I valori `47`, `5` e `54` descrivono soltanto gli esempi ispezionati e non costituiscono statistiche globali del dataset.

---

# Collegamento con l’Esercizio 1.3

Il risultato centrale dell’Esercizio 1.2 è l’identificazione della rappresentazione:

```python
last_hidden_state[:, 0, :]
```

Per ogni recensione questa operazione produce un vettore di 768 componenti.

Le shape attese per l’intero dataset saranno:

```text
train:
[8530, 768]

validation:
[1066, 768]

test:
[1066, 768]
```

Queste matrici non sono ancora state estratte.

Nell’Esercizio 1.3 il flusso previsto sarà:

```text
recensioni
→ tokenizzazione a batch
→ DistilBERT congelato
→ vettore [CLS] da 768 componenti
→ classificatore classico
→ sentiment positivo o negativo
```

La prima soluzione prevista sarà un classificatore lineare come `LinearSVC`, mantenendo validation e test separati.

---

# Comandi disponibili

Dalla root `DLA_LAB2`:

```powershell
python Exercise1/main.py eda
```

```powershell
python Exercise1/main.py inspect-transformer
```

```powershell
python Exercise1/main.py inspect-transformer-batch
```

---

# Stato corrente

```text
Exercise 1.1:
completato, eseguito e verificato

Exercise 1.2:
completato, eseguito e verificato

Exercise 1.3:
non ancora iniziato
```

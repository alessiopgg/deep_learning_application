# Exercise 1 — REINFORCE su CartPole-v1

## Panoramica

In questo esercizio è stato implementato da zero un agente **REINFORCE** per
l'ambiente `CartPole-v1` di Gymnasium.

L'obiettivo non è stato soltanto ottenere una policy capace di risolvere
CartPole, ma costruire una pipeline sperimentale che permettesse di analizzare
in modo controllato:

- l'effetto del learning rate;
- la sensibilità del training alla random seed;
- la stabilità della policy durante l'addestramento;
- l'effetto di un training più lungo;
- la differenza tra il miglior checkpoint osservato durante il training e il
  checkpoint finale;
- l'affidabilità del protocollo di evaluation.

L'implementazione finale separa training, evaluation, salvataggio degli
artifact e analisi dei risultati, permettendo di riprodurre facilmente gli
esperimenti con configurazioni differenti.

---

# 1. Implementazione

## Policy network

La policy è implementata in `models.py` tramite una semplice rete fully
connected:

```text
stato CartPole
      │
      ▼
4 input
      │
Linear(4, 64)
      │
    ReLU
      │
Linear(64, 2)
      │
      ▼
2 logits
```

Lo stato dell'ambiente è composto da quattro valori:

```text
[x, x_dot, theta, theta_dot]
```

mentre le due uscite rappresentano le due possibili azioni:

```text
0 → spostamento a sinistra
1 → spostamento a destra
```

I logits prodotti dalla rete vengono utilizzati per costruire una
distribuzione categorica:

```python
distribution = Categorical(logits=logits)
```

L'azione viene quindi **campionata** dalla distribuzione appresa dalla policy.

La policy rimane quindi stocastica sia durante il training sia durante le
evaluation utilizzate in questo esercizio.

---

## REINFORCE

Per ogni episodio viene raccolta una trajectory completa.

Per ogni step vengono memorizzati:

```text
log π(a_t | s_t)
reward_t
```

Al termine dell'episodio vengono calcolati i discounted return:

```text
G_t = r_t + gamma * G_(t+1)
```

equivalenti a:

```text
G_t = r_t + gamma r_(t+1) + gamma² r_(t+2) + ...
```

La policy loss utilizzata è:

```text
L = - Σ_t G_t log π(a_t | s_t)
```

e viene eseguito **un aggiornamento della policy per episodio**:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Il training segue quindi il flusso:

```text
stato
  │
  ▼
policy
  │
  ▼
azione campionata
  │
  ▼
ambiente
  │
  ▼
reward
  │
  ▼
trajectory completa
  │
  ▼
discounted returns
  │
  ▼
policy loss
  │
  ▼
backpropagation
  │
  ▼
aggiornamento dei pesi
```

---

# 2. Scelte implementative

L'implementazione è stata progressivamente strutturata per rendere gli
esperimenti riproducibili e facilmente confrontabili.

Le principali scelte sono state:

### Separazione modulare

```text
models.py
    └── definizione della PolicyNetwork

reinforce.py
    ├── raccolta trajectory
    ├── discounted returns
    ├── policy update
    ├── training
    └── evaluation

Exercise1/main.py
    ├── configurazione esperimento
    ├── inizializzazione ambiente
    ├── optimizer
    ├── gestione seed
    └── salvataggio risultati
```

---

### Parametri configurabili da command line

I principali parametri non sono più hard-coded e possono essere modificati
direttamente da terminale:

```text
--seed
--episodes
--gamma
--lr
--hidden-dim
--eval-every
--eval-episodes
--run-name
```

Questo ha permesso di eseguire sistematicamente esperimenti multi-seed e
confronti tra diverse configurazioni.

---

### Ambiente di training ed evaluation separati

Training ed evaluation utilizzano due istanze differenti di `CartPole-v1`.

Durante il training la policy viene aggiornata.

Durante l'evaluation:

```text
nessun backward
nessun optimizer.step()
nessuna modifica dei pesi
```

La rete viene quindi semplicemente utilizzata per misurarne la performance.

---

### Salvataggio degli artifact

Ogni run produce una directory dedicata contenente:

```text
config.json
training_metrics.csv
evaluation_metrics.csv
policy.pt
best_policy.pt
```

`policy.pt` contiene i pesi della rete **al termine del training**.

`best_policy.pt` contiene invece i pesi della rete nel momento in cui è stata
osservata la migliore evaluation periodica.

I due file sono quindi due fotografie della stessa rete in momenti diversi
dell'addestramento.

---

# 3. Protocollo di evaluation

Una parte importante dell'esercizio è stata la separazione tra training ed
evaluation.

La configurazione utilizzata è:

```text
evaluation ogni 25 episodi di training
20 episodi per ogni evaluation
massimo 500 step per episodio
```

Il procedimento è:

```text
25 episodi di training
        │
        ▼
policy corrente
        │
        ▼
pesi congelati
        │
        ▼
20 episodi di evaluation
        │
        ▼
20 reward
        │
        ▼
reward medio
        │
        ▼
ripresa del training
```

Durante i 20 episodi di evaluation i pesi della rete rimangono invariati.

Per ogni evaluation viene calcolato:

```text
average evaluation reward
average episode length
```

In `CartPole-v1` reward e lunghezza dell'episodio coincidono numericamente,
perché ogni step valido produce reward `+1`.

Il massimo reward di un episodio è:

```text
500
```

Di conseguenza:

```text
average evaluation reward = 500
```

significa che **tutti e 20 gli episodi di quella evaluation hanno raggiunto i
500 step**.

Questa evaluation periodica permette di osservare come evolve la policy
durante il training, ma rimane comunque una stima ottenuta su soli 20 episodi.

Per questo motivo, alla fine degli esperimenti è stata aggiunta anche una
**robust evaluation indipendente su 100 episodi**.

---

# 4. Esperimento 1 — Learning rate e variabilità tra seed

## Obiettivo

Le prime prove hanno mostrato che REINFORCE poteva raggiungere reward molto
elevati, ma con forti oscillazioni.

Per verificare se il comportamento dipendesse dalla configurazione o dalla
singola random seed è stato eseguito un esperimento controllato con:

```text
3 learning rate
×
5 random seed
=
15 training completi
```

Sono stati mantenuti fissi:

| Parametro | Valore |
|---|---:|
| Environment | CartPole-v1 |
| Policy | 4 → 64 → 2 |
| Activation | ReLU |
| Optimizer | Adam |
| Gamma | 0.99 |
| Training episodes | 1000 |
| Evaluation interval | 25 |
| Evaluation episodes | 20 |

Learning rate testati:

```text
0.001
0.005
0.01
```

Seed:

```text
42
123
456
789
1000
```

---

## Risultati aggregati

| Learning rate | Reward finale medio ± std | Interpretazione |
|---:|---:|---|
| **0.001** | **457.35 ± 27.44** | apprendimento lento ma stabile |
| 0.005 | 358.77 ± 132.14 | apprendimento rapido ma molto variabile |
| 0.01 | 62.29 ± 51.78 | training fortemente instabile |

![Evaluation reward medio sulle 5 seed](plots/evaluation_mean_std.png)

Il confronto multi-seed mostra un comportamento molto diverso per i tre
learning rate.

### `lr = 0.001`

Il learning rate più piccolo produce un apprendimento inizialmente più lento,
ma tutte le seed migliorano progressivamente e terminano con performance
elevate:

```text
seed 42   → 417.00
seed 123  → 440.25
seed 456  → 490.35
seed 789  → 484.70
seed 1000 → 454.45
```

La variabilità tra seed rimane relativamente contenuta.

Il comportamento suggerisce che update più piccoli permettano alla policy di
migliorare in maniera più graduale senza distruggere facilmente strategie già
apprese.

---

### `lr = 0.005`

Con `lr=0.005` l'apprendimento è generalmente più rapido, ma molto più
dipendente dalla seed.

Alcune run raggiungono e mantengono reward molto elevati, mentre altre
raggiungono temporaneamente 500 per poi peggiorare.

Ad esempio, per la seed `789`:

```text
episode 700  → 500.00
...
episode 1000 → 170.15
```

La policy è quindi stata capace di risolvere il problema, ma gli aggiornamenti
successivi non hanno preservato la soluzione raggiunta.

---

### `lr = 0.01`

Il learning rate più elevato evidenzia in modo ancora più netto l'instabilità
di vanilla REINFORCE.

![Run individuali con lr=0.01](plots/evaluation_individual_lr0.01.png)

Un esempio particolarmente evidente è la seed `456`:

```text
episode 550 → 500
episode 575 → 500
episode 600 → 500
episode 625 → 500
episode 650 → 500
episode 675 → 500

episode 700 → 10.50
episode 725 →  9.35
...
episode 1000 → 9.50
```

La policy aveva quindi raggiunto una soluzione praticamente perfetta, ma gli
aggiornamenti successivi hanno causato un vero e proprio **policy collapse**.

Questo risultato mostra che il problema non è l'incapacità di REINFORCE di
trovare una buona policy.

Il problema è la capacità di **mantenerla stabilmente** quando gli update sono
troppo aggressivi.

---

## Interpretazione

Vanilla REINFORCE utilizza una stima Monte Carlo del policy gradient, che può
presentare varianza elevata.

L'aggiornamento può essere rappresentato schematicamente come:

```text
nuovi pesi
=
vecchi pesi
+
learning rate × policy gradient stimato
```

Un learning rate elevato amplifica quindi anche gli errori e le oscillazioni
della stima del gradiente.

I risultati multi-seed mostrano chiaramente il trade-off:

```text
learning rate elevato
        ↓
apprendimento potenzialmente rapido
        ↓
update molto aggressivi
        ↓
maggiore instabilità


learning rate ridotto
        ↓
apprendimento più lento
        ↓
update più piccoli
        ↓
maggiore robustezza
```

Per gli esperimenti successivi è stato quindi selezionato:

```text
learning rate = 0.001
```

non perché raggiungesse più rapidamente il massimo, ma perché mostrava il
comportamento più consistente tra seed.

---

# 5. Esperimento 2 — Estensione del training

## Motivazione

Dopo 1000 episodi le run con `lr=0.001` risultavano ancora generalmente in
miglioramento.

È stato quindi verificato se un training più lungo permettesse alla
configurazione più conservativa di raggiungere performance migliori.

Sono state mantenute le stesse cinque seed e tutti gli altri parametri,
aumentando solamente:

```text
training episodes:
1000 → 2000
```

---

## Risultati

| Training budget | Reward finale medio ± std |
|---:|---:|
| 1000 episodi | 457.35 ± 27.44 |
| **2000 episodi** | **484.79 ± 14.26** |

![Confronto 1000 e 2000 episodi](plots/training_budget_1000_vs_2000.png)

Il training esteso produce due effetti contemporaneamente:

```text
reward medio:
457.35 → 484.79

deviazione standard:
27.44 → 14.26
```

Quindi il modello non solo migliora mediamente, ma diventa anche più
consistente tra differenti training seed.

La crescita media del reward finale è:

```text
+27.44
```

---

## Evoluzione durante i 2000 episodi

![Training esteso lr=0.001](plots/extended_evaluation_mean_std.png)

Tutte le cinque seed raggiungono almeno una volta una evaluation media pari a
500.

| Training seed | Primo episodio con evaluation = 500 | Reward finale |
|---:|---:|---:|
| 42 | 1350 | 500.00 |
| 123 | 1300 | 490.30 |
| 456 | 1150 | 476.80 |
| 789 | 1775 | 495.90 |
| 1000 | 1125 | 460.95 |

L'episodio medio del primo raggiungimento del massimo è circa:

```text
1340
```

Il training più lungo è quindi utile.

Tuttavia non elimina completamente l'instabilità di REINFORCE.

Alcune run raggiungono 500, peggiorano temporaneamente e successivamente
recuperano.

Ad esempio la seed `1000` raggiunge 500 intorno all'episodio 1125, subisce un
forte calo tra circa 1250 e 1500 episodi e torna successivamente vicino al
massimo.

La conclusione non è quindi che REINFORCE converge monotonamente a 500, ma che
un learning rate più conservativo permette di sfruttare meglio un budget di
training maggiore.

---

# 6. Esperimento 3 — Best checkpoint vs checkpoint finale

Durante il training esteso vengono salvati due checkpoint:

```text
best_policy.pt
policy.pt
```

La differenza è:

```text
best_policy.pt
=
configurazione dei pesi con la migliore
evaluation periodica osservata

policy.pt
=
configurazione dei pesi dopo l'ultimo
episodio di training
```

Il `best_policy.pt` viene selezionato sulla base delle evaluation periodiche da
20 episodi.

Tuttavia 20 episodi costituiscono ancora una stima relativamente rumorosa
della performance reale della policy.

È quindi possibile che una policy ottenga una evaluation particolarmente
favorevole senza essere necessariamente la policy più robusta in generale.

---

## Robust evaluation

Dopo la conclusione del training, entrambi i checkpoint sono stati valutati
nuovamente utilizzando:

```text
100 episodi indipendenti
evaluation seed = 1000 ... 1099
nessun aggiornamento dei pesi
```

Gli stessi 100 seed sono stati utilizzati per tutti i checkpoint, in modo da
rendere il confronto controllato.

Le metriche considerate sono:

```text
mean reward
standard deviation
median reward
minimum reward
maximum reward
success rate @500
```

---

## Risultati aggregati

| Checkpoint | Reward medio tra training seed | Success rate @500 |
|---|---:|---:|
| Best checkpoint | 479.97 ± 9.95 | 89.4% |
| **Final checkpoint** | **486.15 ± 8.01** | **89.6%** |

Il risultato è interessante perché il checkpoint denominato `best` non risulta
sistematicamente migliore nella valutazione indipendente.

Confrontando le singole training seed:

| Training seed | Best checkpoint | Final checkpoint |
|---:|---:|---:|
| 42 | 468.45 | **492.44** |
| 123 | 477.84 | **484.69** |
| 456 | 478.84 | **491.30** |
| 789 | **498.47** | 491.24 |
| 1000 | **476.26** | 471.10 |

Il checkpoint finale è migliore in tre training seed su cinque, mentre il best
checkpoint è migliore nelle altre due.

La differenza aggregata rimane relativamente piccola.

Il risultato principale è quindi:

```text
migliore evaluation osservata durante il training
≠
necessariamente migliore policy su nuovi episodi
```

Il checkpoint `best_policy.pt` è il migliore rispetto alle evaluation
periodiche disponibili durante il training.

La robust evaluation su 100 episodi indipendenti fornisce invece una stima più
affidabile delle prestazioni effettive della policy.

---

## Esempio di policy particolarmente robusta

Il `best_policy.pt` della training seed `789` ha ottenuto:

```text
Mean reward:       498.47
Std reward:         10.83
Median reward:     500.00
Minimum reward:    412.00
Maximum reward:    500.00
Success rate @500: 98%
```

Quindi 98 episodi su 100 hanno raggiunto il limite massimo di 500 step e perfino
il peggior episodio è durato 412 step.

Questo rappresenta uno degli esempi più stabili ottenuti durante gli
esperimenti.

---

# 7. Policy loss

Durante gli esperimenti è stata registrata anche la policy loss.

![Policy loss](plots/policy_loss_mean_std.png)

La loss di REINFORCE non deve essere interpretata nello stesso modo di una
classica loss supervisionata.

La funzione utilizzata è:

```text
L = - Σ_t G_t log π(a_t | s_t)
```

Quando la policy migliora:

- gli episodi possono diventare più lunghi;
- aumenta il numero di termini nella somma;
- possono aumentare i discounted return.

Di conseguenza la magnitudine della policy loss può aumentare anche mentre la
performance dell'agente migliora.

Per questo motivo la metrica principale utilizzata per confrontare le policy
rimane il **reward ottenuto durante l'evaluation**, mentre la policy loss viene
utilizzata principalmente come metrica diagnostica.

---

# 8. Risultati principali

Gli esperimenti permettono di riassumere il comportamento osservato in questo
modo.

### 1. REINFORCE è in grado di risolvere CartPole

Una rete estremamente piccola:

```text
4 → 64 → 2
```

è sufficiente per apprendere una policy capace di raggiungere il reward massimo
di 500.

---

### 2. Il learning rate è critico

`lr=0.01` permette alla policy di raggiungere rapidamente ottime performance,
ma produce frequenti policy collapse.

`lr=0.005` migliora la situazione ma rimane fortemente dipendente dalla seed.

`lr=0.001` apprende più lentamente, ma produce il comportamento più consistente.

---

### 3. Una singola seed non è sufficiente

La stessa configurazione può produrre training molto differenti.

Il confronto su cinque seed ha mostrato differenze che non sarebbero state
visibili osservando solamente la seed 42.

---

### 4. Più training è utile se gli update sono sufficientemente conservativi

Per `lr=0.001`, aumentare il budget da 1000 a 2000 episodi porta:

```text
reward finale medio:
457.35 → 484.79

std tra seed:
27.44 → 14.26
```

---

### 5. Raggiungere 500 non significa aver ottenuto una policy definitivamente stabile

Diverse policy raggiungono il massimo e successivamente peggiorano.

Questo comportamento è particolarmente evidente con learning rate elevati.

---

### 6. Il checkpoint con il miglior score osservato non è necessariamente quello che generalizza meglio

Il confronto indipendente sui 100 episodi mostra:

```text
best checkpoints:
479.97 ± 9.95

final checkpoints:
486.15 ± 8.01
```

La selezione del checkpoint deve quindi essere interpretata insieme alla
variabilità del protocollo di evaluation.

---

# 9. Collegamento con Exercise 2

Gli esperimenti dell'Exercise 1 evidenziano una caratteristica centrale di
vanilla REINFORCE:

```text
policy gradient Monte Carlo
        │
        ▼
elevata varianza
        │
        ▼
forte sensibilità alla seed
        │
        ▼
oscillazioni degli update
        │
        ▼
possibili policy collapse
```

Questo risultato motiva direttamente le tecniche studiate nell'Exercise 2:

```text
standardizzazione dei return
value baseline
advantage
variance reduction
```

L'Exercise 2 può quindi essere interpretato come il tentativo di ridurre
proprio i problemi osservati sperimentalmente in questo esercizio.

---

# 10. Riproducibilità

## Ambiente

Dalla directory:

```text
DLA_LAB3/
```

attivare l'ambiente:

```bash
conda activate DRL
```

---

## Eseguire una singola run

Esempio:

```bash
python -m Exercise1.main \
    --lr 0.001 \
    --seed 42 \
    --episodes 1000 \
    --gamma 0.99 \
    --hidden-dim 64 \
    --eval-every 25 \
    --eval-episodes 20 \
    --run-name reinforce_lr0.001_gamma0.99_h64_seed42
```

---

## Esperimento multi-seed

Il confronto principale utilizza:

```text
learning rates:
0.001
0.005
0.01

seed:
42
123
456
789
1000
```

con:

```text
1000 episodi di training
evaluation ogni 25 episodi
20 episodi per evaluation
```

---

## Training esteso

Le cinque run con `lr=0.001` vengono estese a 2000 episodi tramite:

```bash
./Exercise1/run_extended_2000.sh
```

---

## Generazione dei grafici

```bash
python -m Exercise1.plot_results
```

I principali grafici prodotti sono:

```text
plots/evaluation_mean_std.png
plots/evaluation_individual_lr0.001.png
plots/evaluation_individual_lr0.005.png
plots/evaluation_individual_lr0.01.png
plots/final_reward_summary.png
plots/training_reward_mean_std.png
plots/policy_loss_mean_std.png
plots/extended_evaluation_mean_std.png
plots/extended_evaluation_individual.png
plots/extended_training_reward_mean_std.png
plots/extended_policy_loss_mean_std.png
plots/training_budget_1000_vs_2000.png
```

---

## Robust evaluation dei checkpoint

Per confrontare `best_policy.pt` e `policy.pt` su 100 episodi indipendenti:

```bash
python -m Exercise1.evaluate_checkpoints
```

I risultati vengono salvati in:

```text
Exercise1/robust_evaluation/
```

con:

```text
checkpoint_summary.csv
seed42_best_episodes.csv
seed42_final_episodes.csv
seed123_best_episodes.csv
seed123_final_episodes.csv
...
```

---

# 11. Struttura principale

```text
DLA_LAB3/
├── models.py
├── reinforce.py
│
└── Exercise1/
    ├── main.py
    ├── plot_results.py
    ├── evaluate_checkpoints.py
    ├── run_extended_2000.sh
    ├── README.md
    │
    ├── runs/
    │   └── <run_name>/
    │       ├── config.json
    │       ├── training_metrics.csv
    │       ├── evaluation_metrics.csv
    │       ├── policy.pt
    │       └── best_policy.pt
    │
    ├── robust_evaluation/
    │   ├── checkpoint_summary.csv
    │   └── *_episodes.csv
    │
    └── plots/
        └── *.png
```

---

# Conclusione

L'Exercise 1 mostra che vanilla REINFORCE è perfettamente in grado di apprendere
`CartPole-v1`, ma evidenzia contemporaneamente i principali limiti pratici di
un policy gradient Monte Carlo privo di tecniche di variance reduction.

Il confronto multi-seed mostra che learning rate elevati possono produrre
apprendimento rapido ma fortemente instabile, fino alla completa perdita di una
policy che aveva precedentemente risolto il problema.

Un learning rate più conservativo (`0.001`) produce invece un apprendimento più
lento ma nettamente più robusto. Estendendo il training a 2000 episodi, il
reward finale medio raggiunge:

```text
484.79 ± 14.26
```

e tutte le cinque training seed raggiungono almeno una volta il reward massimo
di 500.

La successiva valutazione indipendente dei checkpoint mostra inoltre che il
checkpoint con la migliore evaluation osservata durante il training non è
necessariamente quello che ottiene la migliore performance su nuovi episodi.

L'esercizio mette quindi in evidenza tre aspetti fondamentali del Deep
Reinforcement Learning:

```text
qualità della policy
≠
singolo reward elevato

buon training
≠
training monotono

best checkpoint osservato
≠
necessariamente miglior modello generale
```

Questi risultati costituiscono la motivazione sperimentale naturale per le
tecniche di riduzione della varianza introdotte nell'Exercise 2.
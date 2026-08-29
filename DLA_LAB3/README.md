# Deep Learning Applications — Laboratorio 3

Il Laboratorio 3 studia metodi **policy-based** e **value-based** di Deep Reinforcement Learning attraverso tre esercizi progressivi su ambienti Gymnasium.

Il percorso parte dall'implementazione di **REINFORCE** su `CartPole-v1`, analizza due tecniche di riduzione della varianza e conclude con un **Deep Q-Network (DQN)** applicato a `CartPole-v1` e `LunarLander-v3`.

## Obiettivi del laboratorio

Il lavoro affronta tre domande principali:

* come apprende e quanto è stabile una policy addestrata con vanilla REINFORCE;
* quanto Standardized Returns e una Learned Value Baseline riducono la varianza del Policy Gradient;
* come Experience Replay e Target Q-Network permettono di applicare Deep Q-Learning a task con spazio delle azioni discreto.

Il protocollo sperimentale separa training ed evaluation, usa seed controllati e conserva configurazioni, metriche, checkpoint selezionati e risultati aggregati necessari alla riproduzione dell'analisi.

## Struttura del repository

```text
DLA_LAB3/
├── README.md
├── environment.yml
├── models.py
├── reinforce.py
│
├── Exercise1/
│   ├── README.md
│   ├── main.py
│   ├── evaluate_checkpoints.py
│   ├── plot_results.py
│   ├── run_extended_2000.sh
│   ├── runs/
│   ├── robust_evaluation/
│   └── plots/
│
├── Exercise2/
│   ├── README.md
│   ├── main.py
│   ├── value_baseline_main.py
│   ├── reinforce_ex2.py
│   ├── evaluate_checkpoints.py
│   ├── plot_results.py
│   ├── runs/
│   ├── robust_evaluation/
│   └── plots/
│
└── Exercise3/
    ├── README.md
    ├── dqn.py
    ├── main.py
    ├── lunarlander_main.py
    ├── evaluate_results.py
    ├── plot_results.py
    ├── run_final.sh
    ├── runs/
    ├── results/
    └── plots/
```

`models.py` e `reinforce.py` contengono le componenti condivise dagli esercizi basati su REINFORCE; l'implementazione DQN è mantenuta interamente nella directory `Exercise3`.

## Esercizi

### Exercise 1 — REINFORCE su CartPole

L'[Exercise 1](Exercise1/README.md) implementa da zero **vanilla REINFORCE** su `CartPole-v1`.

La policy è una rete:

```text
4 -> 64 -> 2
```

e viene aggiornata al termine di ogni trajectory utilizzando discounted return Monte Carlo.

Lo studio sperimentale comprende più learning rate, cinque training seed, training da 1000 e 2000 episodi e robust evaluation indipendente dei checkpoint.

Con `lr=0.001`, il training esteso porta il reward finale medio da:

```text
457.35 ± 27.44
```

a:

```text
484.79 ± 14.26
```

La robust evaluation dei checkpoint finali produce:

```text
486.15 ± 8.01
```

con success rate medio a reward `500` pari a `89.6%`.

### Exercise 2 — Variance Reduction in REINFORCE

L'[Exercise 2](Exercise2/README.md) confronta vanilla REINFORCE con due tecniche di riduzione della varianza:

```text
Standardized Returns
Learned Value Baseline
```

La Value Baseline introduce una seconda rete:

```text
4 -> 64 -> 1
```

che approssima `V(s)` e permette di utilizzare l'advantage:

```text
G_t - V(s_t)
```

Nella robust evaluation dei checkpoint finali:

| Metodo | Reward medio ± std | Success rate @500 |
|---|---:|---:|
| Vanilla REINFORCE | 486.15 ± 8.01 | 89.6% |
| Standardized Returns | 496.00 ± 5.33 | 97.0% |
| **Value Baseline** | **498.68 ± 1.22** | **98.6%** |

Il confronto mostra che entrambe le tecniche migliorano la stabilità, con la Value Baseline che produce il comportamento più consistente nel protocollo utilizzato.

### Exercise 3 — Deep Q-Learning su CartPole e LunarLander

L'[Exercise 3](Exercise3/README.md) implementa la variante **3.2 — Deep Q-Learning** con:

```text
epsilon-greedy exploration
Experience Replay
Temporal-Difference targets
Target Q-Network
```

La stessa implementazione DQN viene utilizzata per entrambi gli ambienti con architettura:

```text
CartPole-v1:     4 -> 128 -> 128 -> 2
LunarLander-v3:  8 -> 128 -> 128 -> 4
```

Le configurazioni finali vengono valutate su tre training seed (`42`, `123`, `456`) e su 100 episodi di test per seed.

| Ambiente | Reward medio tra training seed | Successo finale |
|---|---:|---:|
| `CartPole-v1` | **477.72 ± 31.51** | **88.0%** con reward >=475 |
| `LunarLander-v3` | **275.96 ± 5.38** | **97.7%** con reward >=200 |

Su CartPole, due dei tre training seed raggiungono reward `500` su tutti i 100 episodi di test; complessivamente l'`86.3%` dei 300 episodi finali raggiunge esattamente `500`.

Su LunarLander il comportamento rimane consistente tra le tre inizializzazioni, con medie per seed comprese tra `268.38` e `280.38`.

## Ambiente e riproducibilità

L'ambiente Conda di riferimento è definito in [`environment.yml`](environment.yml).

Dalla root del repository:

```bash
conda env create -f DLA_LAB3/environment.yml
conda activate DRL
cd DLA_LAB3
```

Versioni principali registrate:

* Python 3.12.13
* PyTorch 2.13.0
* Gymnasium 1.3.0
* NumPy 2.5.1
* Matplotlib 3.11.1
* pygame 2.6.1
* Box2D 2.3.10
* SWIG 4.4.1

Gli Exercise 1 e 2 utilizzano cinque training seed:

```text
42, 123, 456, 789, 1000
```

L'Exercise 3 utilizza:

```text
42, 123, 456
```

Le evaluation robuste vengono eseguite con seed espliciti e ambienti separati dal training. I risultati numerici riportati derivano dagli artifact delle run effettivamente eseguite; non viene assunto determinismo bit-a-bit tra piattaforme differenti.

## Entry point principali

I comandi seguenti vanno eseguiti dalla directory `DLA_LAB3`.

### Exercise 1

Training REINFORCE:

```bash
python -m Exercise1.main \
  --lr 0.001 \
  --seed 42 \
  --episodes 2000 \
  --gamma 0.99 \
  --hidden-dim 64 \
  --eval-every 25 \
  --eval-episodes 20
```

Campagna multi-seed estesa:

```bash
bash Exercise1/run_extended_2000.sh
```

Robust evaluation:

```bash
python -m Exercise1.evaluate_checkpoints
```

Grafici:

```bash
python -m Exercise1.plot_results
```

### Exercise 2

Vanilla REINFORCE:

```bash
python -m Exercise2.main \
  --mode vanilla \
  --seed 42 \
  --episodes 2000 \
  --lr 0.001
```

Standardized Returns:

```bash
python -m Exercise2.main \
  --mode standardized \
  --seed 42 \
  --episodes 2000 \
  --lr 0.001
```

Value Baseline:

```bash
python -m Exercise2.value_baseline_main \
  --seed 42 \
  --episodes 2000 \
  --policy-lr 0.001 \
  --value-lr 0.001
```

Robust evaluation:

```bash
python -m Exercise2.evaluate_checkpoints
```

Grafici:

```bash
python -m Exercise2.plot_results
```

### Exercise 3

Campagna finale completa:

```bash
bash Exercise3/run_final.sh
```

Training CartPole per una singola seed:

```bash
python -m Exercise3.main --seed 42
```

Training LunarLander per una singola seed:

```bash
python -m Exercise3.lunarlander_main --seed 42
```

Evaluation finale:

```bash
python -m Exercise3.evaluate_results
```

Grafici:

```bash
python -m Exercise3.plot_results
```

I dettagli relativi a metodo, configurazioni, selezione dei checkpoint e test finale sono documentati nei README specifici dei singoli esercizi.

## Tracking degli esperimenti

Il Laboratorio 3 utilizza tracking locale basato sugli artifact delle run.

Gli esperimenti REINFORCE registrano principalmente:

```text
config.json
training_metrics.csv
evaluation_metrics.csv
```

insieme ai checkpoint necessari alle robust evaluation.

L'Exercise 3 conserva per ciascuna run finale:

```text
config.json
training_metrics.csv
evaluation_metrics.csv
selected_q_network.pt
```

e raccoglie i risultati finali in CSV separati a livello di episodio, checkpoint e aggregato.

I grafici vengono prodotti dagli artifact persistiti senza ripetere il training.

## Politica del repository

Per mantenere il repository leggero vengono esclusi dal controllo versione:

* dataset;
* checkpoint e pesi non necessari alla consegna;
* output generici e log locali;
* array e feature di grandi dimensioni;
* cache;
* ambienti virtuali;
* file temporanei e materiale di sviluppo.

Vengono invece mantenuti quando utili alla documentazione e alla riproduzione:

* codice sorgente;
* configurazioni delle run;
* metriche CSV/JSON;
* risultati aggregati;
* grafici selezionati;
* checkpoint finali leggeri necessari a riprodurre le evaluation;
* README.

I risultati numerici riportati nella documentazione derivano dagli artifact delle run effettivamente eseguite.

## Navigazione

* [Exercise 1 — REINFORCE su CartPole](Exercise1/README.md)
* [Exercise 2 — Variance Reduction in REINFORCE](Exercise2/README.md)
* [Exercise 3 — Deep Q-Learning su CartPole e LunarLander](Exercise3/README.md)

## Riferimenti e assistenza AI

Il progetto utilizza principalmente **PyTorch** e **Gymnasium**.

I concetti centrali del laboratorio comprendono Policy Gradient, REINFORCE, discounted return, variance reduction, state-value function, advantage, Q-learning, Temporal-Difference learning, epsilon-greedy exploration, Experience Replay e Target Network.

ChatGPT è stato utilizzato come supporto per chiarimenti teorici, organizzazione del lavoro, revisione del codice, debugging, progettazione degli esperimenti, analisi degli artifact, costruzione dei grafici e documentazione. Le scelte implementative e i risultati riportati sono stati verificati sul codice e sugli output effettivi del progetto.

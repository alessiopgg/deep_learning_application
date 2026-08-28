# Deep Learning Applications — Laboratorio 3

Il Laboratorio 3 è dedicato allo studio di diversi metodi di **Deep Reinforcement Learning** attraverso tre esercizi sperimentali su ambienti Gymnasium.

Gli esperimenti analizzano algoritmi policy-based e value-based, tecniche di riduzione della varianza, strategie di esplorazione, stabilità del training e affidabilità dei protocolli di evaluation.

Gli ambienti utilizzati sono principalmente `CartPole-v1` e `LunarLander-v3`.

## Obiettivi del laboratorio

Gli obiettivi principali del laboratorio sono:

* implementare e valutare **REINFORCE** su `CartPole-v1`;
* studiare tecniche di **variance reduction** per rendere il training più stabile e robusto;
* implementare un **Deep Q-Network (DQN)** con Experience Replay e Target Network;
* analizzare sperimentalmente stabilità, sensibilità agli iperparametri e robustezza dei checkpoint.

Il protocollo sperimentale separa training ed evaluation, utilizza seed controllati, conserva metriche e configurazioni delle run e integra valutazioni indipendenti su 100 episodi per verificare la robustezza dei modelli.

## Struttura del repository

```text
DLA_LAB3/
├── README.md
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
    ├── runs/
    ├── results/
    └── plots/
```

`models.py` e `reinforce.py` contengono componenti condivise utilizzate nelle implementazioni basate su REINFORCE, mentre il codice DQN è mantenuto nella directory dell'Exercise 3.

## Esercizi

### Exercise 1 — REINFORCE su CartPole

L'[Exercise 1](Exercise1/README.md) implementa da zero **vanilla REINFORCE** su `CartPole-v1`.

La policy è una rete fully connected:

```text
4 → 64 → 2
```

che produce i logits di una distribuzione categorica sulle due azioni dell'ambiente.

Il training utilizza trajectory complete, discounted return Monte Carlo e un aggiornamento della policy per episodio.

Lo studio sperimentale analizza:

* tre learning rate: `0.001`, `0.005`, `0.01`;
* cinque training seed: `42`, `123`, `456`, `789`, `1000`;
* training da 1000 episodi;
* training esteso a 2000 episodi;
* differenza tra best checkpoint e checkpoint finale;
* robust evaluation indipendente su 100 episodi.

Il confronto tra learning rate mostra una forte sensibilità del vanilla REINFORCE alla dimensione degli update: learning rate elevati possono produrre rapidamente policy molto efficaci, ma anche forti oscillazioni e successivi **policy collapse**.

Con `lr=0.001`, aumentando il training budget da 1000 a 2000 episodi, il reward finale medio passa da:

```text
457.35 ± 27.44
```

a:

```text
484.79 ± 14.26
```

La robust evaluation dei checkpoint finali sulle cinque training seed produce:

```text
486.15 ± 8.01
```

con un success rate medio a reward 500 pari a `89.6%`.

### Exercise 2 — Variance Reduction in REINFORCE

L'[Exercise 2](Exercise2/README.md) studia due tecniche di riduzione della varianza del policy gradient su `CartPole-v1`:

```text
Standardized Returns
Learned Value Baseline
```

La standardizzazione utilizza return centrati e normalizzati all'interno dell'episodio:

```text
(G_t - mean(G)) / (std(G) + epsilon)
```

La Value Baseline introduce invece una rete:

```text
ValueNetwork
4 → 64 → 1
```

che approssima il valore dello stato e permette di utilizzare come learning signal:

```text
G_t - V(S_t)
```

Gli esperimenti utilizzano:

```text
training episodes = 2000
policy learning rate = 0.001
gamma = 0.99
```

con cinque training seed:

```text
42
123
456
789
1000
```

Nella robust evaluation dei checkpoint finali:

| Metodo | Reward medio ± std | Success rate @500 |
|---|---:|---:|
| Vanilla REINFORCE | 486.15 ± 8.01 | 89.6% |
| Standardized Returns | 496.00 ± 5.33 | 97.0% |
| **Value Baseline** | **498.68 ± 1.22** | **98.6%** |

La Value Baseline raggiunge inoltre per la prima volta un'evaluation media di 500 intorno all'episodio `945`.

I risultati mostrano che entrambe le tecniche migliorano la stabilità del training e la robustezza finale, con la Value Baseline che ottiene i risultati più consistenti nel protocollo sperimentale utilizzato.

### Exercise 3 — Deep Q-Learning su CartPole e LunarLander

L'[Exercise 3](Exercise3/README.md) implementa un **Deep Q-Network (DQN)** per ambienti con spazio delle azioni discreto.

L'algoritmo viene applicato a:

```text
CartPole-v1
LunarLander-v3
```

L'implementazione comprende:

* Q-Network;
* epsilon-greedy exploration;
* Experience Replay;
* Replay Buffer;
* Temporal-Difference targets;
* Target Q-Network;
* sincronizzazione periodica della target network;
* evaluation greedy indipendente dal training.

Per CartPole il modello utilizza:

```text
4 → 64 → 2
```

Per LunarLander utilizza:

```text
8 → 64 → 4
```

### CartPole-v1

Su CartPole vengono confrontati learning rate e loss differenti.

La configurazione selezionata utilizza:

```text
learning rate = 5e-4
loss          = MSE
```

e raggiunge nella robust evaluation su 100 episodi greedy:

```text
Mean reward = 280.84
Std reward  = 75.88
```

### LunarLander-v3

Su LunarLander viene inizialmente eseguito un training da 500 episodi, successivamente esteso a 1000 episodi.

Il checkpoint finale della run da 1000 episodi ottiene:

```text
Mean reward   = 172.68
Std reward    = 70.02
Median reward = 184.82
```

con:

```text
Positive episodes = 96%
Reward >= 100     = 84%
Reward >= 200     = 32%
```

La robust evaluation mostra inoltre che la selezione del checkpoint basata su poche evaluation periodiche può essere rumorosa. Per questo motivo i modelli finali vengono confrontati anche su 100 episodi indipendenti.

## Ambiente e riproducibilità

Il laboratorio è stato sviluppato e testato su **Ubuntu tramite WSL2**, utilizzando **Conda/Miniforge**.

L'ambiente utilizzato è definito nel file:

```text
environment.yml
```

e può essere ricreato con:

```bash
conda env create -f environment.yml
conda activate DRL
```

La configurazione testata utilizza principalmente:

```text
Python      3.12.13
PyTorch     2.13.0
Gymnasium   1.3.0
NumPy       2.5.1
Matplotlib  3.11.1
pygame      2.6.1
Box2D       2.3.10
SWIG        4.4.1
```

Nell'ambiente utilizzato per gli esperimenti, PyTorch disponeva inoltre di supporto **CUDA 12.9**.

Gli ambienti utilizzati sono:

```text
CartPole-v1
LunarLander-v3
```

Negli esperimenti multi-seed vengono utilizzate le training seed:

```text
42
123
456
789
1000
```

Le robust evaluation utilizzano 100 episodi con seed:

```text
1000–1099
```

La riproducibilità viene gestita attraverso seed espliciti, configurazioni salvate, metriche persistenti e valutazioni separate dal training.

Le differenze dovute alla natura stocastica degli algoritmi di reinforcement learning rimangono parte integrante dell'analisi sperimentale.

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

Training esteso:

```bash
./Exercise1/run_extended_2000.sh
```

Robust evaluation:

```bash
python -m Exercise1.evaluate_checkpoints
```

Generazione dei grafici:

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

Generazione dei grafici:

```bash
python -m Exercise2.plot_results
```

### Exercise 3

Training DQN su CartPole:

```bash
python -m Exercise3.main
```

Training DQN su LunarLander:

```bash
python -m Exercise3.lunarlander_main
```

Robust evaluation:

```bash
python -m Exercise3.evaluate_results
```

Generazione dei grafici:

```bash
python -m Exercise3.plot_results
```

I dettagli relativi agli iperparametri, al protocollo sperimentale e all'interpretazione dei risultati sono documentati nei README specifici dei singoli esercizi.

## Tracking degli esperimenti

Il Laboratorio 3 utilizza un tracking locale basato sugli artifact prodotti dalle run.

Negli esperimenti basati su REINFORCE vengono registrati principalmente:

```text
config.json
training_metrics.csv
evaluation_metrics.csv
```

insieme ai checkpoint delle reti.

Le robust evaluation producono CSV separati con risultati per episodio e statistiche aggregate.

Gli esperimenti DQN conservano analogamente metriche delle run, risultati delle valutazioni robuste e grafici utilizzati nell'analisi.

I grafici finali vengono generati a partire dagli artifact persistiti, evitando di ripetere training o evaluation soltanto per produrre le visualizzazioni.

## Politica del repository

Per mantenere il repository leggero vengono generalmente esclusi dal controllo versione:

* checkpoint e pesi dei modelli di grandi dimensioni (`.pt`, `.pth`, `.ckpt`);
* directory di checkpoint non necessarie alla riproduzione dei risultati;
* output generici e log locali;
* array e feature di grandi dimensioni (`.npy`, `.npz`);
* cache;
* ambienti virtuali;
* file temporanei.

Vengono invece mantenuti, quando utili alla documentazione e alla riproduzione dell'analisi:

* codice sorgente;
* configurazioni delle run;
* metriche CSV/JSON;
* risultati aggregati delle robust evaluation;
* grafici selezionati;
* checkpoint leggeri selezionati, necessari per riprodurre alcune evaluation;
* README.

I risultati numerici riportati nella documentazione derivano dagli artifact delle run effettivamente eseguite.

## Navigazione

* [Exercise 1 — REINFORCE su CartPole](Exercise1/README.md)
* [Exercise 2 — Variance Reduction in REINFORCE](Exercise2/README.md)
* [Exercise 3 — Deep Q-Learning su CartPole e LunarLander](Exercise3/README.md)

## Riferimenti e assistenza AI

Il progetto utilizza principalmente **PyTorch** e **Gymnasium**.

I principali concetti di Deep Reinforcement Learning studiati nel laboratorio comprendono policy gradient, REINFORCE, discounted return, variance reduction, state-value function, advantage, Q-learning, Temporal-Difference learning, epsilon-greedy exploration, Experience Replay e Target Network.

ChatGPT è stato utilizzato come supporto per chiarimenti teorici, organizzazione del lavoro, revisione del codice, debugging, progettazione degli esperimenti, analisi degli artifact e documentazione. Le scelte implementative e i risultati riportati sono stati verificati sul codice e sugli output effettivi del progetto.

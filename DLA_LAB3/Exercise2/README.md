# Exercise 2 — Variance Reduction in REINFORCE su CartPole-v1

L'Exercise 2 studia due strategie per ridurre la varianza del learning signal di **REINFORCE** su `CartPole-v1`:

```text
Standardized Returns
Learned Value Baseline
```

L'obiettivo è modificare il segnale che pesa la log-probabilità delle azioni senza cambiare il task o l'architettura della policy, e verificare sperimentalmente l'effetto su velocità di apprendimento, stabilità e robustezza finale.

Il confronto finale considera tre configurazioni:

```text
Vanilla REINFORCE
        vs
Standardized Returns
        vs
Learned Value Baseline
```

---

## Metodo — Variance Reduction in REINFORCE

L'Exercise 2 mantiene invariati ambiente e `PolicyNetwork` dell'Exercise 1 e interviene sul **segnale usato per aggiornare la policy**.

Il confronto è:

| Metodo | Segnale usato nella policy loss |
|---|---|
| Vanilla REINFORCE | `G_t` |
| Standardized Returns | `(G_t - mean(G)) / (std(G) + ε)` |
| Value Baseline | `G_t - V_w(S_t)` |

L'obiettivo è verificare se ridurre la variabilità del learning signal renda REINFORCE più stabile e meno sensibile alla training seed.

### Standardized Returns

Nel Vanilla REINFORCE la policy loss usa direttamente il discounted return:

```text
L_policy = - Σ_t G_t log πθ(A_t | S_t)
```

La prima variante standardizza i return all'interno dell'episodio:

```text
G_hat_t =
(G_t - mean(G)) / (std(G) + ε)
```

con `ε = 1e-8`.

La rete e il training loop non cambiano: cambia soltanto il coefficiente che pesa ogni `log_prob`.

### Value Baseline

La seconda variante introduce una `ValueNetwork` che stima il valore dello stato:

```text
V_w(s) ≈ v_π(s)
```

Le architetture utilizzate sono:

```text
PolicyNetwork: 4 → 64 → 2
ValueNetwork:  4 → 64 → 1
```

La policy viene aggiornata usando l'advantage stimato:

```text
A_t = G_t - V_w(S_t)
```

quindi:

```text
L_policy =
- Σ_t A_t log πθ(A_t | S_t)
```

La ValueNetwork viene invece addestrata a predire il return Monte Carlo:

```text
L_value =
MSE(V_w(S_t), G_t)
```

Nel codice:

```python
advantages = returns_tensor - values.detach()
```

Il `detach()` mantiene separati i due gradient flow: la policy loss aggiorna soltanto la `PolicyNetwork`, mentre la value loss aggiorna soltanto la `ValueNetwork`.

La baseline utilizza quindi un'informazione **dipendente dallo stato**, mentre la standardizzazione utilizza soltanto le statistiche dei return dell'episodio corrente.

---

## Implementazione

La struttura dell'esercizio è:

```text
DLA_LAB3/
├── models.py
├── reinforce.py
│
└── Exercise2/
    ├── README.md
    ├── main.py
    ├── reinforce_ex2.py
    ├── value_baseline_main.py
    ├── evaluate_checkpoints.py
    ├── plot_results.py
    ├── runs/
    ├── robust_evaluation/
    └── plots/
```

| File | Responsabilità |
|---|---|
| `../models.py` | `PolicyNetwork` e `ValueNetwork` |
| `../reinforce.py` | discounted return ed evaluation condivisa |
| `reinforce_ex2.py` | standardizzazione, trajectory con stati, policy/value update e training loop |
| `main.py` | training Vanilla o Standardized Returns |
| `value_baseline_main.py` | training con learned Value Baseline |
| `evaluate_checkpoints.py` | robust evaluation di best e final checkpoint |
| `plot_results.py` | aggregazione delle run e generazione dei grafici |

### Controllo della randomizzazione

L'inizializzazione della ValueNetwork consuma numeri casuali aggiuntivi.

Per evitare che la sola presenza della seconda rete modifichi artificialmente la successiva sequenza di campionamento della policy, `value_baseline_main.py`:

```text
inizializza PolicyNetwork
        ↓
salva lo stato RNG PyTorch
        ↓
inizializza ValueNetwork
        ↓
ripristina lo stato RNG
```

Questa scelta rende il confronto tra configurazioni più controllato.

---

## Protocollo sperimentale

La configurazione finale è:

| Parametro | Valore |
|---|---:|
| Environment | `CartPole-v1` |
| Policy | `4 → 64 → 2` |
| Activation | ReLU |
| Policy optimizer | Adam |
| Policy learning rate | `0.001` |
| Discount factor | `0.99` |
| Training episodes | `2000` |
| Evaluation interval | `25` episodi |
| Evaluation episodes | `20` |
| Hidden dimension | `64` |
| Evaluation policy | stocastica |

Per la Value Baseline:

| Parametro | Valore |
|---|---:|
| ValueNetwork | `4 → 64 → 1` |
| Activation | ReLU |
| Optimizer | Adam |
| Value learning rate | `0.001` |
| Target | Monte Carlo return |
| Loss | MSE |

Le training seed sono:

```text
42
123
456
789
1000
```

Sono quindi disponibili:

```text
3 metodi × 5 seed = 15 run
```

Ogni run da 2000 episodi produce:

```text
2000 / 25 = 80 evaluation periodiche
```

e quindi, per ogni metodo:

```text
5 seed × 80 evaluation = 400 evaluation
```

---

## Risultati durante il training

### Reward finale

Considerando l'ultima evaluation periodica delle cinque training seed:

| Metodo | Reward finale medio ± std |
|---|---:|
| Vanilla REINFORCE | 484.79 ± 14.26 |
| Standardized Returns | **498.56 ± 2.88** |
| Value Baseline | **498.43 ± 2.20** |

La differenza principale non è soltanto nel reward medio, ma nella dispersione tra training seed.

La deviazione standard finale scende da:

```text
14.26
```

a:

```text
2.88  → Standardized Returns
2.20  → Value Baseline
```

<p align="center">
  <img src="plots/evaluation_mean_std.png"
       alt="Evaluation reward medio e deviazione standard sulle cinque training seed"
       width="900">
</p>

La Value Baseline raggiunge più rapidamente la regione di reward elevato e mostra una minore variabilità nella parte finale del training.

---

### Velocità di apprendimento

È stato misurato il primo episodio nel quale una evaluation periodica raggiunge reward medio `500`.

| Metodo | Primo reward 500 medio |
|---|---:|
| Vanilla REINFORCE | 1340 |
| Standardized Returns | 1310 |
| **Value Baseline** | **945** |

La Value Baseline raggiunge quindi il massimo mediamente circa:

```text
395 episodi prima del Vanilla
365 episodi prima dello Standardized
```

---

### Permanenza nella regione ad alta performance

Sulle 400 evaluation disponibili per metodo:

| Metodo | Evaluation ≥475 | Evaluation =500 |
|---|---:|---:|
| Vanilla REINFORCE | 96 / 400 | 27 / 400 |
| Standardized Returns | 167 / 400 | 51 / 400 |
| **Value Baseline** | **219 / 400** | **103 / 400** |

Le evaluation esattamente a `500` rappresentano:

```text
Vanilla REINFORCE   6.75%
Standardized       12.75%
Value Baseline     25.75%
```

La baseline appresa non si limita quindi a raggiungere il massimo prima: rimane più frequentemente nella regione di massima performance.

---

## Analisi delle singole seed

<p align="center">
  <img src="plots/evaluation_individual.png"
       alt="Evaluation reward delle singole training seed per i tre metodi"
       width="900">
</p>

Le curve individuali evidenziano che nessuna delle tecniche elimina completamente la natura stocastica di REINFORCE.

Vanilla REINFORCE mostra la variabilità maggiore e può subire cali significativi anche dopo aver raggiunto reward elevati.

La standardizzazione riduce frequenza e intensità delle oscillazioni.

La Value Baseline porta in genere le seed nella regione di reward elevato più rapidamente e mantiene curve maggiormente concentrate vicino al massimo.

---

## Training reward

<p align="center">
  <img src="plots/training_reward_mean_std.png"
       alt="Training reward aggregato sulle cinque training seed"
       width="900">
</p>

Il grafico utilizza una moving average di `50` episodi.

Il training reward conferma la tendenza osservata nell'evaluation: la riduzione della varianza produce soprattutto un **apprendimento più consistente**, non semplicemente un picco di reward più alto.

---

## Value loss

<p align="center">
  <img src="plots/value_loss_mean_std.png"
       alt="ValueNetwork MSE aggregata sulle cinque training seed"
       width="900">
</p>

La value loss non deve essere interpretata come una classica validation loss supervisionata.

Durante il training cambiano continuamente:

```text
policy
stati visitati
lunghezza degli episodi
distribuzione dei return
target della ValueNetwork
```

Il problema di regressione è quindi non stazionario.

Una crescita temporanea della MSE non implica automaticamente che l'agente stia peggiorando.

La `value_loss` viene utilizzata come metrica diagnostica; la misura principale della qualità dell'agente rimane il reward di evaluation.

---

## Robust evaluation

Le evaluation periodiche utilizzano `20` episodi e possono essere rumorose.

Per valutare in modo più affidabile i checkpoint è stata quindi eseguita una robust evaluation con:

```text
100 episodi per checkpoint
evaluation seed = 1000 ... 1099
```

Per ogni episodio vengono controllati sia l'RNG dell'environment sia quello PyTorch.

Sono stati valutati:

```text
3 metodi
×
5 training seed
×
2 checkpoint
=
30 checkpoint
```

confrontando:

```text
best_policy.pt
policy.pt
```

per un totale di:

```text
30 × 100 = 3000 episodi
```

Durante questa valutazione la ValueNetwork non è necessaria: l'azione viene scelta esclusivamente dalla PolicyNetwork.

### Risultati aggregati

| Metodo | Checkpoint | Reward medio ± std tra training seed | Success rate @500 |
|---|---|---:|---:|
| Vanilla REINFORCE | Best | 479.97 ± 9.95 | 89.4% |
| Vanilla REINFORCE | Final | 486.15 ± 8.01 | 89.6% |
| Standardized Returns | Best | 488.76 ± 5.41 | 92.6% |
| Standardized Returns | Final | 496.00 ± 5.33 | 97.0% |
| Value Baseline | Best | 485.99 ± 3.71 | 90.2% |
| **Value Baseline** | **Final** | **498.68 ± 1.22** | **98.6%** |

<p align="center">
  <img src="plots/robust_reward_comparison.png"
       alt="Confronto robusto tra best e final checkpoint"
       width="800">
</p>

Il checkpoint finale della Value Baseline ottiene il risultato aggregato migliore:

```text
498.68 ± 1.22
```

con:

```text
98.6% success rate @500
```

---

## Robustezza rispetto alla training seed

<p align="center">
  <img src="plots/final_robust_reward_by_seed.png"
       alt="Reward robusto dei checkpoint finali per training seed"
       width="850">
</p>

I checkpoint finali producono:

| Training seed | Vanilla | Standardized | Value Baseline |
|---:|---:|---:|---:|
| 42 | 492.44 | 498.58 | **500.00** |
| 123 | 484.69 | 498.11 | **498.71** |
| 456 | 491.30 | 485.36 | **500.00** |
| 789 | 491.24 | **498.76** | 497.75 |
| 1000 | 471.10 | **499.19** | 496.92 |

Per la Value Baseline tutte le cinque training seed rimangono nell'intervallo:

```text
496.92 ≤ mean reward ≤ 500
```

Le seed `42` e `456` ottengono reward `500` in tutti i 100 episodi della robust evaluation.

---

## Success rate

<p align="center">
  <img src="plots/robust_success_rate_comparison.png"
       alt="Success rate a reward 500 nella robust evaluation"
       width="800">
</p>

Per i checkpoint finali:

```text
Vanilla REINFORCE   89.6%
Standardized        97.0%
Value Baseline      98.6%
```

La differenza tra i metodi riguarda quindi anche la probabilità empirica di raggiungere il limite massimo del task, non soltanto una piccola variazione del reward medio.

---

## Best checkpoint e checkpoint finale

Il checkpoint chiamato `best_policy.pt` viene selezionato usando evaluation periodiche da soli `20` episodi.

Nella robust evaluation da 100 episodi, il checkpoint finale risulta mediamente migliore per tutti e tre i metodi:

```text
Vanilla:
479.97 → 486.15

Standardized:
488.76 → 496.00

Value Baseline:
485.99 → 498.68
```

`best` significa quindi:

```text
migliore evaluation periodica osservata
```

non necessariamente:

```text
checkpoint più robusto su nuovi episodi
```

Il risultato evidenzia l'importanza di una valutazione indipendente e sufficientemente ampia.

---

## REINFORCE with learned baseline, non Actor-Critic TD

La ValueNetwork viene addestrata usando come target il return Monte Carlo completo:

```text
G_t
```

Non viene utilizzato un target bootstrapped come:

```text
r_t + γ V(S_(t+1))
```

L'implementazione rimane quindi **REINFORCE with learned value baseline**.

La rete di valore svolge il ruolo di baseline state-dependent per il policy gradient, ma non introduce un aggiornamento Temporal-Difference.

---

## Output e artifact

Le run Vanilla e Standardized producono:

```text
Exercise2/runs/<run_name>/
├── config.json
├── training_metrics.csv
├── evaluation_metrics.csv
├── policy.pt
└── best_policy.pt
```

Le run con Value Baseline producono inoltre:

```text
value.pt
best_value.pt
```

`training_metrics.csv` contiene:

```text
episode
reward
policy_loss
```

e, per la Value Baseline:

```text
episode
reward
policy_loss
value_loss
```

`evaluation_metrics.csv` contiene:

```text
episode
average_reward
average_length
```

La robust evaluation salva:

```text
Exercise2/robust_evaluation/
├── checkpoint_summary.csv
├── aggregated_summary.csv
└── *_episodes.csv
```

I grafici finali vengono salvati in:

```text
Exercise2/plots/
```

e comprendono:

```text
evaluation_mean_std.png
evaluation_individual.png
training_reward_mean_std.png
value_loss_mean_std.png
robust_reward_comparison.png
robust_success_rate_comparison.png
final_robust_reward_by_seed.png
```

---

## Riproduzione

I comandi vanno eseguiti dalla directory `DLA_LAB3` con l'ambiente `DRL` attivo.

### Vanilla REINFORCE

```bash
python -m Exercise2.main \
  --mode vanilla \
  --seed 42 \
  --episodes 2000 \
  --gamma 0.99 \
  --lr 0.001 \
  --hidden-dim 64 \
  --eval-every 25 \
  --eval-episodes 20 \
  --run-name ex2_vanilla_ep2000_lr0.001_seed42
```

### Standardized Returns

```bash
python -m Exercise2.main \
  --mode standardized \
  --seed 42 \
  --episodes 2000 \
  --gamma 0.99 \
  --lr 0.001 \
  --hidden-dim 64 \
  --eval-every 25 \
  --eval-episodes 20 \
  --run-name ex2_standardized_ep2000_lr0.001_seed42
```

### Value Baseline

```bash
python -m Exercise2.value_baseline_main \
  --seed 42 \
  --episodes 2000 \
  --gamma 0.99 \
  --policy-lr 0.001 \
  --value-lr 0.001 \
  --hidden-dim 64 \
  --eval-every 25 \
  --eval-episodes 20 \
  --run-name ex2_value_baseline_ep2000_plr0.001_vlr0.001_seed42
```

Il protocollo completo ripete le tre configurazioni con:

```text
42
123
456
789
1000
```

### Robust evaluation

Dopo aver generato i checkpoint:

```bash
python -m Exercise2.evaluate_checkpoints
```

### Grafici

```bash
python -m Exercise2.plot_results
```

---

## Limiti

- Il confronto riguarda `CartPole-v1` e le configurazioni effettivamente eseguite; non dimostra una superiorità universale della Value Baseline.
- Cinque training seed forniscono una misura della variabilità tra run, ma non costituiscono una caratterizzazione statistica esaustiva.
- La standardizzazione è episodica e non utilizza informazioni specifiche sullo stato.
- La ValueNetwork apprende da target Monte Carlo non stazionari, quindi la sua MSE non è direttamente confrontabile con una validation loss supervisionata.
- La policy rimane stocastica anche durante l'evaluation.
- Il checkpoint `best` è selezionato su 20 episodi e può essere favorito dal rumore della valutazione.
- I checkpoint `.pt` sono esclusi dal repository Git e devono essere rigenerati localmente per ripetere la robust evaluation completa.
- Non vengono studiate baseline più complesse, GAE, TD critic o veri algoritmi Actor-Critic: l'obiettivo rimane isolare le due tecniche richieste dall'esercizio.

---

## Conclusioni

L'Exercise 2 mostra che il modo in cui REINFORCE costruisce il learning signal influenza direttamente la stabilità dell'ottimizzazione.

La standardizzazione:

```text
G_t
↓
centratura e normalizzazione
↓
G_hat_t
```

controlla la scala degli aggiornamenti e produce policy finali più consistenti.

La Value Baseline introduce invece un riferimento dipendente dallo stato:

```text
G_t - V_w(S_t)
```

che misura quanto il return osservato sia stato migliore o peggiore rispetto a ciò che era atteso.

Nel protocollo multi-seed utilizzato, questa configurazione raggiunge più rapidamente la regione di reward massimo e produce la robust evaluation finale più consistente:

```text
498.68 ± 1.22
98.6% success rate @500
```

Il risultato supportato dagli esperimenti è quindi circoscritto al protocollo utilizzato: su `CartPole-v1`, con le cinque training seed considerate, **REINFORCE con learned Value Baseline ha mostrato il miglior compromesso osservato tra velocità di apprendimento, stabilità e robustezza finale**.

---

## Riferimenti e assistenza AI

Riferimenti principali:

- notebook ufficiale della consegna `DLA-Lab2-DRL.ipynb`;
- materiale del corso su Policy Gradient, REINFORCE e baseline;
- Gymnasium — `CartPole-v1`;
- PyTorch.

ChatGPT è stato utilizzato come supporto per chiarimenti teorici, organizzazione del lavoro, revisione del codice, debugging, progettazione degli esperimenti, analisi degli artifact e documentazione. Le configurazioni, i grafici e i risultati quantitativi riportati derivano dal codice e dagli artifact effettivamente prodotti dal progetto.

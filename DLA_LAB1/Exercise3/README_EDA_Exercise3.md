# Exercise 3.3 — Exploratory Data Analysis

## Dataset

L’EDA è stata eseguita sul dataset:

```text
keremberke/german-traffic-sign-detection
configuration: full
```

Il dataset contiene immagini complete di scene stradali annotate con bounding box e classi di cartelli.

Le annotazioni originali utilizzano il formato:

```text
[x_min, y_min, width, height]
```

La tassonomia comprende **43 classi reali**.

## Dimensione del dataset

| Split | Immagini | Oggetti | Immagini senza oggetti |
|---|---:|---:|---:|
| Train | 383 | 600 | 29 |
| Validation | 108 | 170 | 6 |
| Test | 54 | 82 | 4 |
| **Totale** | **545** | **852** | **39** |

Il numero medio di oggetti per immagine è circa **1,56**. La maggior parte delle immagini contiene uno o due cartelli, mentre il massimo osservato è 6.

Le immagini senza oggetti non vengono eliminate: sono esempi utili di background per il detector.

## Dimensioni delle immagini

Tutte le immagini hanno la stessa risoluzione:

```text
1360 × 800 pixel
```

con aspect ratio pari a:

```text
1,7
```

L’uniformità delle dimensioni rende più semplice il preprocessing e permette di confrontare direttamente le aree relative delle bounding box.

## Bounding box

Le bounding box risultano molto piccole rispetto alle immagini complete.

| Statistica | Larghezza | Altezza |
|---|---:|---:|
| Minimo | 16 px | 16 px |
| Mediana | 38 px | 37 px |
| Media | 43,4 px | 42,8 px |
| 95° percentile | 90,4 px | 89,4 px |
| Massimo | 127 px | 128 px |

La box mediana occupa circa lo **0,126%** dell’area dell’immagine. Questo conferma che la detection di oggetti piccoli sarà uno degli aspetti principali del progetto.

## Distribuzione delle scale

Usando le soglie COCO:

```text
small:  area < 32²
medium: 32² ≤ area < 96²
large:  area ≥ 96²
```

la distribuzione complessiva è:

| Scala | Box | Percentuale |
|---|---:|---:|
| Small | 315 | 37,0% |
| Medium | 503 | 59,0% |
| Large | 34 | 4,0% |

La distribuzione è abbastanza simile tra train, validation e test. Gli oggetti large sono molto rari.

## Aspect ratio

Le bounding box sono prevalentemente quadrate:

```text
mediana: 1,00
media:   1,013
```

Il 90% centrale delle box ha un aspect ratio approssimativamente compreso tra `0,92` e `1,13`.

Per la prima baseline non risultano quindi necessarie modifiche personalizzate agli aspect ratio delle anchor.

## Integrità delle annotazioni

I controlli hanno prodotto:

| Controllo | Risultato |
|---|---:|
| Box valide | 852 |
| Box invalide | 0 |
| Box degeneri | 0 |
| Box non finite | 0 |
| Box fuori immagine | 0 |
| Categorie invalide | 0 |
| Aree incoerenti | 0 |
| Annotazioni duplicate esatte | 2 righe |

Il dataset è quindi geometricamente pulito.

Le due righe duplicate appartengono allo stesso gruppo di annotazioni nel training set e rappresentano lo stesso cartello. Nel dataset adapter verrà mantenuta una sola copia, registrando esplicitamente la rimozione.

## Distribuzione delle classi

Il dataset è fortemente sbilanciato.

Nel training set:

```text
classe più frequente:
no overtaking -trucks- → 50 oggetti

classe meno frequente tra quelle presenti:
pedestrian crossing → 1 oggetto
```

Il rapporto massimo tra classi presenti è quindi `50:1`.

Alcune classi sono molto rare e le metriche aggregate dovranno essere accompagnate da:

- AP per classe;
- numero di esempi per classe;
- precision e recall;
- analisi qualitativa degli errori.

## Classi assenti dagli split

Nel training sono presenti esempi di **41 classi su 43**.

Le classi completamente assenti dal train sono:

```text
animals
restriction ends
```

La classe `animals` compare nel test, mentre `restriction ends` compare nella validation.

Questo limite deve essere considerato durante la valutazione: una detection head inizializzata da zero non può apprendere normalmente una classe mai osservata nel training.

Gli split ufficiali vengono comunque mantenuti per la baseline, in modo da conservare un protocollo riproducibile e confrontabile.

## Decisioni consolidate

Dall’EDA derivano le seguenti decisioni:

- mantenere gli split ufficiali;
- conservare le immagini senza oggetti;
- rimuovere soltanto i duplicati esatti;
- non applicare altre correzioni alle bounding box;
- non ridurre aggressivamente la risoluzione;
- prestare particolare attenzione agli oggetti small;
- mantenere inizialmente le anchor standard;
- non introdurre augmentation geometriche aggressive;
- riportare sempre metriche per classe e supporto;
- documentare esplicitamente le classi assenti dal training.

## Output prodotti

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
│   ├── split_sizes.png
│   ├── objects_per_image.png
│   ├── class_distribution_train.png
│   ├── class_distribution_validation.png
│   ├── class_distribution_test.png
│   ├── image_dimensions.png
│   ├── box_dimensions.png
│   ├── box_relative_area.png
│   ├── box_aspect_ratio.png
│   ├── object_scale_distribution.png
│   └── annotation_integrity_checks.png
└── examples/
    └── immagini annotate con ground truth
```

## Conclusione

Il dataset è piccolo ma sufficientemente pulito per costruire una baseline Faster R-CNN. Le principali difficoltà non riguardano la qualità geometrica delle annotazioni, ma:

- dimensione ridotta dei cartelli;
- forte sbilanciamento delle classi;
- classi rare o assenti dal training;
- numero limitato di immagini e annotazioni.

Questi aspetti guideranno la progettazione del dataset adapter, del protocollo di valutazione e delle strategie di fine-tuning.

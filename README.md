# Coltrane

## Description

General use, **pipeline-oriented** ML platform built on top of sklearn.
Lets user configure pipelines, load data, and evaluate pipeline against data. Who's said that improvising over `Giant Steps` has to be hard? This framework eases and standardizes research process. User can focus on configuring pipeline, or implementing core pipeline elements.

## Features

- processing research batches
- some nice plots by plotly
- data wrappers and loaders

### Data set loaders

#### Plain `.csv`

- single label

## Getting started

```sh
pip install coltrane
```


## Inspection

Load data set.

```py
from pathlib import Path

from coltrane.classification import Inspector
from coltrane.file.io.csv.single import Data

__LOG = Path('log')
__DATA_IRIS = Path('coltrane/test/data/iris.csv')
__RANDOM_STATE = 45625461

data = Data(path=__DATA_IRIS)
```

Create `Inspector` instance and run inspection. This will output some stats to provided output directory.

```py
inspector = Inspector()
summary = inspector.inspect(data, output=__LOG)
```

You can access some additional stats and plot them.

```py

# this one's useful for cost-sensitive learning
class_balance = inspector.get_class_balance(data)
inspector.plot_class_balance(class_balance)
```

Check notebook examples for more.

## Processing

```py
from pathlib import Path

from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from coltrane import Batch
from coltrane.classification import Processor
from coltrane.file.io.csv.single import Data


__LOG = Path('log')
__DATA_IRIS = Path('coltrane/test/data/iris.csv')
__RANDOM_STATE = 45625461


batch = Batch(
    Data(path=__DATA_IRIS),
    pipeline=Pipeline(
        steps=[
            ('standard-scaler', StandardScaler()),
            ('naive-bayes', GaussianNB())
        ]
    ),
    selection=RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=4,
        random_state=__RANDOM_STATE
    ),
    scorers={
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    },
    encoder=LabelEncoder()
)

processor = Processor()
stats = processor.process(batch, output=__LOG)
```

### Authors

- Piotr Rarus (piotr.rarus@gmail.com)

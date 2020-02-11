# Coltrane

## Description

General use, **pipeline-oriented** machine learning framework.
Lets user configure pipelines, load data, and evaluate pipeline against data. Who's said that improvising over `Giant Steps` has to be hard? This framework eases and standardizes research process. User can focus on configuring pipeline, or implementing core pipeline elements.

## Features

### Data set loaders

#### plain `.csv`

- single label

### Logging

- fitted pipelines
- validation metrics
- aggregated validation metrics
- confusion matrix for each split

## Getting started

```shell
pip install coltrane
```

## Example

```py
from pathlib import Path

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from coltrane import Batch
from coltrane.classification import Inspector, Processor
from coltrane.file.io.csv.single import Data


__LOG = Path('log')
__DATA_IRIS = Path('/data/iris.csv')
__RANDOM_STATE = 45625461


batch = Batch(
    data=Data(path=__DATA_IRIS),
    pipeline=Pipeline(
        steps=[
            ('standard-scaler', StandardScaler()),
            ('naive-bayes', GaussianNB())
        ]
    ),
    selection=RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=1,
        random_state=__RANDOM_STATE
    ),
    metrics=[
        accuracy_score,
        (
            precision_score,
            {
                'average': 'weighted'
            }
        ),
        (
            recall_score,
            {
                'average': 'weighted'
            }
        ),
        (
            f1_score,
            {
                'average': 'weighted'
            }
        )
    ],
    encoder=LabelEncoder(),
    multiprocessing=True
)

inspector = Inspector()
inspector.inspect([data], output=__LOG)

processor = Processor()
processor.process([batch], output=__LOG)


```

### Authors

- Piotr Rarus (piotr.rarus@gmail.com)

# Coltrane

## Description

General use, **pipeline-oriented** machine learning framework.
Lets user configure pipelines, load data, and evaluate pipeline against data. Who's said that improvising over `Giant Steps` has to be hard? This framework eases and standardizes research process. User can focus on configuring pipeline, or implementing core pipeline elements. Let us do the rest.

## Features

### Data set loaders

#### plain `.csv`

- single label
- multi label
- multi data wrapper (useful for similar data sets)

#### image based

- flat
- nested (each channel in separate file)

##### Please note

Currently we're using [OpenCV](https://www.opencv.org/) to read images. It's possible we'll switch to [scikit-image](https://scikit-image.org/) in near future, when they decide on which plugin they use. [OpenCV](https://www.opencv.org/) has some issues, when reading 16-bit images.

### Pipelines

- drop duplicates `[optional]`
- drop records with missing values `[optional]`
- imputation of missing values `[optional]`
- auto detection and encoding of categorical features `[optional]`
- re-sampling `[optional]`
- dimensionality reduction `[optional]`
- preprocessing (i.e. mean removal, variance scaling) `[optional]`
- discretization of continuous features `[optional]`
- pre-visualization of feature space `[optional]`
- splits selection
- model estimation
- evaluation

### Logging

- fitted data preparation transformers
- fitted estimators
- validation metrics
- aggregated validation metrics
- confusion matrix for each split
- aggregated confusion matrices

## Getting started

Be sure you have `virtualenv` installed on your machine.

```shell
pip install virtualenv
```

Clone this repository to your disk. Then install this package through `pip`.

```shell
cd [directory]
pip install .
```

Standalone distribution of this framework comes with exemplary data sets and pipelines. You shall check them out. They're located under `examples` folder.

### Data set configuration

```python
from coltrane import csv
```

Every data set module is well-documented via docstrings. Look them up through Intellisense, and see how they can be instantiated.

### Pipeline configuration

```python
from coltrane import Pipeline
```

Pipeline module is well-documented via docstrings. Look up the constructor through Intellisense, and see what interface should each pipeline's element implement.

### Pipeline execution

```python
from coltrane import execute
```

You can use `execute` method to evaluate pipelines against data sets. Every data set will be evaluated against each pipeline from data generator. To map data sets against pipelines, simply use this method multiple times.

`execute` interface

```python
def execute(
    data: Generator[DataSet, None, None],
    pipelines: Generator[Pipeline, None, None],
    output_path: str
):
```

### Data inspection

We provide a script, that will help you inspect your data.

`inspect` interface

```python
def inspect(
    data: Generator[DataSet, None, None],
    output_path: str
):
```

It will give you some a priori insight into data set, that could help you create meaningful pipelines.

This includes:
  
- attributes vector size
- which of the attributes hold categorical/numerical values
- features sparsity
- class balance
- visualization of features distribution in 2D space
- pandas' `description()` of features

This set of stats is calculated against:

- raw data
- data with dropped missing values
- data with dropped duplicates

### Tests

Tests are located under `tests` folder.

```shell
cd [project-path]
pytest ./tests/
```

### Authors

- Piotr Rarus (piotr.rarus@gmail.com)

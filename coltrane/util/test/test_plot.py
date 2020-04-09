import numpy as np
from pytest import fixture
from sklearn.datasets import make_classification

from ..plot import Plot


@fixture(scope='session')
def plot() -> Plot:
    return Plot(debug_mode=True)


def test_class_balance(plot: Plot):
    balance = {
        'a': 100,
        'c': 50,
        'b': 123
    }

    plot.class_balance(balance)


def test_features_distribution(plot: Plot, random_state: int):
    x, y = make_classification(
        n_samples=50,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state
    )

    plot.features_distribution(x, y)


def test_heatmap(plot: Plot):

    attributes = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
    data = np.random.uniform(size=(8, 8))

    plot.heatmap(data, 'Confusion matrix', attributes, attributes)


def test_metrics(plot: Plot):
    scores = {
        'accuracy': np.random.uniform(size=50),
        'precision': np.random.uniform(size=50),
        'recall': np.random.uniform(size=50),
        'f1': np.random.uniform(size=50),
        'mse': np.random.uniform(low=-10, high=10, size=50),
        'foo': np.random.uniform(low=0, high=10, size=50),
    }

    plot.scores(scores)

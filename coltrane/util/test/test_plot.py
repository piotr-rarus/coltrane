from pytest import fixture
from ..plot import Plot


@fixture(scope='session')
def plot() -> Plot:
    return Plot(debug_mode=True)


def test_class_balance(plot: Plot):
    balance = {
        'a': 100,
        'b': 50,
        'c': 123
    }

    plot.class_balance(balance)

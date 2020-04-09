import numpy as np
from pytest import fixture

__RANDOM_STATE = 3547632


def pytest_sessionstart(session):
    np.random.seed(__RANDOM_STATE)


@fixture(scope='session')
def random_state() -> int:
    return __RANDOM_STATE

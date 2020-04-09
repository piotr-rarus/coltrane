from pathlib import Path

from pytest import fixture
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_score, recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from coltrane import Batch
from coltrane.classification import Inspector, Processor
from coltrane.file.io.csv.single import Data

__LOG = Path('log')
__DATA_IRIS = Path('coltrane/test/data/iris.csv')


@fixture(scope='function')
def data() -> Data:
    return Data(path=__DATA_IRIS)


@fixture(scope='function')
def batch(data: Data, random_state: int) -> Batch:

    return Batch(
        data,
        pipeline=Pipeline(
            steps=[
                ('standard-scaler', StandardScaler()),
                ('naive-bayes', GaussianNB())
            ],
        ),
        selection=RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=random_state
        ),
        scorers={
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        },
        encoder=LabelEncoder(),
        multiprocessing=True
    )


def test_inspection(data: Data):
    inspector = Inspector()
    inspector.inspect(data, output=__LOG)


def test_classification(batch: Batch):
    processor = Processor()
    processor.process(batch, output=__LOG)

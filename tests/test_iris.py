from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from coltrane import Batch, csv, execute, inspect

"""
Container for filepath constants.

"""

__LOGS = 'logs'
__DATA_IRIS = 'tests\\data\\iris.csv'

__RANDOM_STATE = 45625461


def batches():

    yield Batch(
        data=csv.single_label.DataSet(path=__DATA_IRIS),
        pipeline=Pipeline(
            steps=[
                ('standard-scaler', StandardScaler()),
                ('naive-bayes', GaussianNB())
            ]
        ),
        selection=RepeatedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=__RANDOM_STATE
        ),
        validation=(
            classification_report,
            {
                'output_dict': True
            }
        )
    )

    yield Batch(
        data=csv.single_label.DataSet(path=__DATA_IRIS),
        pipeline=Pipeline(
            steps=[
                ('standard-scaler', StandardScaler()),
                ('svc', SVC(gamma='auto'))
            ]
        ),
        selection=RepeatedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=__RANDOM_STATE
        ),
        validation=(
            classification_report,
            {
                'output_dict': True
            }
        )
    )


def data_set():
    yield csv.single_label.DataSet(path=__DATA_IRIS)


def test_inspection():
    inspect(data_set, output=__LOGS)


def test_classification():
    execute(batches, output=__LOGS)

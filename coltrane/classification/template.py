from timeit import default_timer as timer

from austen import Logger
from sklearn.metrics import confusion_matrix as compute_confusion_matrix
from tqdm import tqdm

from ..batch import Batch
from ..utility import aggregate, plot


def process(batch: Batch, logger: Logger):
    # TODO: docstrings

    data = batch.data
    selection = batch.selection
    pipeline = batch.pipeline
    validate, validation_kwargs = batch.validation

    classes = set(data.labels)

    splits = selection.split(data.records, data.labels)
    splits_iter = enumerate(tqdm(splits, desc='Splits'))

    stats = []
    confusion_matrices = []

    performance = {
        'dt_fit': [],
        'dt_predict': [],
        'dt_predict_single': []
    }

    with logger.get_child('splits') as splits_logger:
        # TODO: generator doesn't have length attribute
        # TODO: so there's kinda lame progressbar :/
        # ? does all selectors implement `n_splits` prop?
        for split_index, (train_index, test_index) in splits_iter:
            with splits_logger.get_child(str(split_index)) as split_logger:

                train_records = data.records[train_index]
                train_labels = data.labels[train_index]

                test_records = data.records[test_index]
                test_labels = data.labels[test_index]

                # TODO: this looks ugly, maybe some wrapper?
                start = timer()
                pipeline.fit(train_records, train_labels)
                end = timer()
                dt_fit = end - start

                start = timer()
                pred_labels = pipeline.predict(test_records)
                end = timer()
                dt_predict = end - start
                dt_predict_single = dt_predict / len(pred_labels)

                telemetry = {
                    'dt_fit': dt_fit,
                    'dt_predict': dt_predict,
                    'dt_predict_single': dt_predict_single
                }

                evaluation = validate(
                    test_labels,
                    pred_labels,
                    **validation_kwargs
                )

                # ! will it return labels too?
                confusion_matrix = compute_confusion_matrix(
                    test_labels,
                    pred_labels
                )

                plot.confusion_matrix(
                    confusion_matrix,
                    classes,
                    split_logger
                )

                confusion_matrices.append(confusion_matrix)

                split_logger.add_entry('evaluation', evaluation)
                split_logger.add_entry('telemetry', telemetry)
                split_logger.save_obj(pipeline, 'pipeline')

                __update_performance(performance, telemetry)

                stats.append(evaluation)

    grouped_stats = aggregate.group_stats(stats)
    logger.add_entry('summary', aggregate.stats(grouped_stats))
    logger.add_entry('performance', aggregate.performance(performance))

    plot.confusion_matrix(
        aggregate.confusion_matrix(confusion_matrices),
        classes,
        logger
    )

    plot.stats(grouped_stats, logger)


def __update_performance(performance, telemetry):
    for key, value in telemetry.items():
        performance[key].append(value)

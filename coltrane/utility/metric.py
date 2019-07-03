def evaluate(test, pred, metrics):
    stats = {}

    for metric in metrics:
        name, value = __evaluate(test, pred, metric)
        stats[name] = value

    return stats


def __evaluate(test, pred, metric):
    op, kwargs = None, None

    if type(metric) is tuple and len(metric) == 2:
        op, kwargs = metric

    elif callable(metric):
        op = metric
        kwargs = {}

    value = op(test, pred, **kwargs)

    name = op.__name__

    return name, value


def group(metrics):
    """
    Groups evaluation metric values from each split, by metrics.

    Parameters
    ----------
    stats : list[dict]
        List of evaluation stats from each `train/test` split.
        Please follow the structure denoted
        by sklearn.metrics.classification_report.

    Returns
    -------
    dict
        Grouped evaluation measures.
    """

    grouped = {}

    for stat in metrics:
        for metric, value in stat.items():

            if metric not in grouped:
                grouped[metric] = []

            grouped[metric].append(value)

    return grouped

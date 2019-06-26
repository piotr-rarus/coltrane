from collections import OrderedDict


def evaluate(test, pred, metrics):
    stats = OrderedDict()

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

from collections import OrderedDict


def evaluate(self, test, pred, metrics):
    stats = OrderedDict()

    for metric in metrics:
        name, value = self.__evaluate(test, pred, metric)
        stats[name] = value

    return stats


def __evaluate(self, test, pred, metric):
    op, kwargs = None, None

    if type(metric) is tuple and len(metric) == 2:
        op, kwargs = metric

    elif callable(metric):
        op = metric
        kwargs = {}

    value = op(test, pred, **kwargs)

    name = op.__name__

    return name, value

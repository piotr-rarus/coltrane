import numpy as np


"""
Utility functions for model evaluation stats/performance aggregation.

"""


def stats(stats):
    """
    Aggregates each measure of model's evaluation.
    Computes `percentiles` and `std` for each of them.

    Parameters
    ----------
    stats : list[dict]
        List of evaluation stats from each `train/test` split.
        Please follow the structure denoted
        by sklearn.metrics.classification_report.

    Returns
    -------
    dict
        Aggregated stats.
    """

    summary = {}

    for metric, values in stats.items():
        stats = {}

        stats['mean'] = np.mean(values)
        stats['min'] = np.min(values)
        stats['max'] = np.max(values)
        stats['std'] = np.std(values)

        summary[metric] = stats

    return summary


def group_metrics(stats):
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

    for stat in stats:
        for metric, value in stat.items():

            if metric not in grouped:
                grouped[metric] = []

            grouped[metric].append(value)

    return grouped


def performance(performance):
    """
    Aggregates performance stats from each `train/test` split.
    Calculates `percentiles` and `std`.

    Parameters
    ----------
    performance : dict
        Key - label
        Value - performance measurements

    Returns
    -------
    dict
        Aggregated performance stats.
    """

    summary = {}

    for label, values in vars(performance).items():
        summary[label] = {}
        summary[label]['mean'] = np.mean(values)
        summary[label]['min'] = np.min(values)
        summary[label]['max'] = np.max(values)
        summary[label]['std'] = np.std(values)

    return summary


def balance(labels):
    """
    Calculates class balance stats.

    Parameters
    ----------
    labels : string
        Ground truth prediction

    Returns
    -------
    dict
        Stats.
    """

    summary = {}
    aggregate = np.unique(labels, return_counts=True)

    for label, count in np.transpose(aggregate):
        summary[str(label)] = count

    counts = aggregate[1]

    summary['stats'] = {
        'std': np.std(counts),
        'percentiles': np.percentile(counts, [0, 25, 50, 75, 100])
    }

    return summary


def confusion_matrix(confusion_matrices):
    """
    Computes average values among confusion matrices.

    Parameters
    ----------
    confusion_matrices : list[nd.array]
        List of confusion matrices from each selection split.

    """

    return np.average(confusion_matrices, axis=0)

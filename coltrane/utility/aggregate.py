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

    for label, measures in stats.items():
        summary[label] = {}

        for measure, values in measures.items():
            summary[label][measure] = {}

            summary[label][measure]["std"] = np.std(values)

            percentiles = np.percentile(values, [0, 25, 50, 75, 100])
            summary[label][measure]["percentiles"] = percentiles

    return summary


def group_stats(stats):
    """
    Groups evaluation metric values from each split,
    by label, then by used metric.

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
        for label, measures in stat.items():

            if type(measures) is not dict:
                continue

            for measure, value in measures.items():

                if label not in grouped:
                    grouped[label] = {}

                if measure not in grouped[label]:
                    grouped[label][measure] = []

                grouped[label][measure].append(value)

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

    for label, measures in performance.items():
        summary[label] = {}
        summary[label]['std'] = np.std(measures)

        percentiles = np.percentile(measures, [0, 25, 50, 75, 100])
        summary[label]['percentiles'] = percentiles

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

import numpy as np


"""
Utility functions for model evaluation stats/performance aggregation.

"""


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

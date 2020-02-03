def evaluate(scorers, pipeline, x_test, y_test):
    stats = {}

    for name, scorer in scorers.items():

        score = scorer(pipeline, x_test, y_test)
        stats[name] = score

    return stats


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

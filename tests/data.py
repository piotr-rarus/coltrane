# from sklearn import preprocessing

# from coltrane import csv

# from . import paths

# """
# Container for filepath constants.

# """

# LOGS = 'logs/'
# DATA_IRIS = 'tests/data/iris.csv'
# DATA_MULTI_IRIS = 'tests/data/iris-multi/'
# DATA_WINE_QUALITY_WHITE = 'tests/data/wine-quality-white.csv'


# def iris():

#     yield csv.single_label.DataSet(
#         path=paths.DATA_IRIS,
#         encoding=(
#             preprocessing.LabelEncoder,
#             {}
#         )
#     )


# def iris_multi():

#     data_sets = csv.multi.DataSets(
#         folder=paths.DATA_MULTI_IRIS,
#         data_set=csv.single_label.DataSet,
#         encoding=(
#             preprocessing.LabelEncoder,
#             {}
#         )
#     )

#     return data_sets.generate()


# def wine_quality():

#     yield csv.single_label.DataSet(
#         path=paths.DATA_WINE_QUALITY_WHITE,
#         encoding=(
#             preprocessing.LabelEncoder,
#             {}
#         )
#     )

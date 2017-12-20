import csv
from collections import namedtuple
from os.path import dirname, join

import numpy as np
import pandas as pd

from sklearn.utils import Bunch

RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])


# def add_categorical_data(data, dataframe):


def load_adult_data(module_path, data_file_name):
    with open(join(module_path, 'data', data_file_name)) as csv_file:
        data_file = list(csv.reader(csv_file))
        dataframe = pd.DataFrame(data_file)
        result = pd.DataFrame()

        attr = ['age', 'workclass', 'fnlwgt',
                'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race',
                'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country']
        feature_names = []
        result[attr[0]] = dataframe[0]
        result[attr[2]] = dataframe[2]
        result[attr[4]] = dataframe[4]
        result[attr[10]] = dataframe[10]
        result[attr[11]] = dataframe[11]
        result[attr[12]] = dataframe[12]
        for i in range(14):
            if result.__contains__(attr[i]):
                feature_names.append(attr[i])
                continue
            trans = pd.get_dummies(dataframe[i])
            for j in trans:
                result[j] = trans[j]
                feature_names.append(j)

        idx = 0
        classes = {}
        target_names = []

        for i in dataframe[14]:
            if (not i in classes):
                classes[i] = idx
                target_names.append(i)
                idx += 1

        n_samples = len(result)
        n_features = len(feature_names)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i in range(n_samples):
            val = []
            for j in feature_names:
                val.append(result[j][i])
            data[i] = np.asarray(val, dtype=np.float64)
            target[i] = np.asarray(classes[dataframe[14][i]], dtype=np.int)

    return result, target, target_names, feature_names


def load_adult(return_X_y=False):
    module_path = dirname(__file__)
    data, target, target_names, feature_names = load_adult_data(module_path, 'adult-reduced.data')

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR="no-desc",
                 feature_names=feature_names)

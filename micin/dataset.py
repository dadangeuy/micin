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
        result[0] = dataframe[0]
        result[1] = dataframe[2]
        result[2] = dataframe[4]
        result[3] = dataframe[10]
        result[4] = dataframe[11]
        result[5] = dataframe[12]
        res_idx = 6
        for i in range(14):
            if result.__contains__(attr[i]):
                feature_names.append(attr[i])
                continue
            trans = pd.get_dummies(dataframe[i])
            for j in trans:
                result[res_idx] = trans[j]
                feature_names.append(j)
                res_idx+=1

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
            tmp = []
            for j, val in enumerate(feature_names):
                print result[j][i]
                tmp.append(float(result[j][i]))
            data[i] = np.asarray(tmp, dtype=np.float64)
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

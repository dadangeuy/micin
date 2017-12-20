import csv
from os.path import dirname, join

import numpy as np
from sklearn.datasets.base import load_data
from sklearn.utils import Bunch


def load_data(module_path, data_file_name):
    with open(join(module_path, 'data', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names

def load_adult(return_X_y=False):
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'adult.data')

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR='no-desc',
                 feature_names=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
                                'workclass_?', 'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked',
                                'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
                                'workclass_State-gov', 'workclass_Without-pay', 'education_10th', 'education_11th',
                                'education_12th', 'education_1st-4th', 'education_5th-6th', 'education_7th-8th',
                                'education_9th', 'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors',
                                'education_Doctorate', 'education_HS-grad', 'education_Masters', 'education_Preschool',
                                'education_Prof-school', 'education_Some-college', 'marital-status_Divorced',
                                'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse',
                                'marital-status_Married-spouse-absent', 'marital-status_Never-married',
                                'marital-status_Separated', 'marital-status_Widowed', 'occupation_?',
                                'occupation_Adm-clerical', 'occupation_Armed-Forces', 'occupation_Craft-repair',
                                'occupation_Exec-managerial', 'occupation_Farming-fishing',
                                'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
                                'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty',
                                'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support',
                                'occupation_Transport-moving', 'relationship_Husband', 'relationship_Not-in-family',
                                'relationship_Other-relative', 'relationship_Own-child', 'relationship_Unmarried',
                                'relationship_Wife', 'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black',
                                'race_Other', 'race_White', 'sex_Female', 'sex_Male', 'native-country_?',
                                'native-country_Cambodia', 'native-country_Canada', 'native-country_China',
                                'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic',
                                'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England',
                                'native-country_France', 'native-country_Germany', 'native-country_Greece',
                                'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands',
                                'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary',
                                'native-country_India', 'native-country_Iran', 'native-country_Ireland',
                                'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan',
                                'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua',
                                'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru',
                                'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal',
                                'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South',
                                'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago',
                                'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia']
                 )

def load_adult_reduced(return_X_y=False):
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'adult-reduced.data')

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR='no-desc',
                 feature_names=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
                 )
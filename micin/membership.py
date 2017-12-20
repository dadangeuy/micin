import numpy as np

def get_mf(dataset):
    mf = []
    for i, fname in enumerate(dataset.feature_names):
        gauss = []
        for j, tname in enumerate(dataset.target_names):
            filtered = []
            for k, target in enumerate(dataset.target):
                if (target == j):
                    filtered.append(dataset.data[k][i])
            gauss.append(['gaussmf', {'mean': np.mean(filtered), 'sigma': np.var(filtered)}])
        mf.append(gauss)
    return mf
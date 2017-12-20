import numpy as np

def get_mf(dataset):
    mf = []
    print dataset.data
    for i in dataset.data[0]:
        print i
    for i, name in enumerate(dataset.feature_names):
        mf.append([])
        for j, clazz in enumerate(dataset.target_names):
            filter_data = list()
            # print dataset.data[i]
            for k, tmp in enumerate(dataset.data[0]):
                for l, tmp2 in enumerate(dataset.data[k]):
                    if dataset.target[l] == j:
                        filter_data.append(tmp2)
                    # if res == j:
                    #     filter_data.append(row[j])
            filter_data = np.array(filter_data).astype(np.float)
            # print filter_data
            mean = np.mean(filter_data)
            variance = np.var(filter_data)
            mf[i].append(['gaussmf', {'mean': mean, 'sigma': variance}])
        # print variance
                # if res
                # print val
                # print res
        # for j, val in enumerate(dataset.data[name]):
        #     print j

    return mf
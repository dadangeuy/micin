from operator import truediv

import pandas as pd
from anfis import anfis
from anfis import membership
from sklearn.model_selection import train_test_split

from membership import get_mf
from micin.dataset import load_adult

dataset = load_adult()
x = dataset.data
y = dataset.target

# sizecv = 5
# kf = StratifiedKFold(n_splits=sizecv, shuffle=True, random_state=123)
# for train, test in kf.split(X, Y):
df = pd.DataFrame(dataset.data)
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
y_test = y_test.tolist()
print "Length: ",len(y_test)
# x_train = x[:100]
# y_train = y[:100]
# x_test = x[100:]
# y_test = y[100:]
# print y_test
# for i in range(len(y_test)):
#     y_actual.append(y_test[i])
# x_train = X[train]
# y_train = Y[train]
# x_test = X[test]
# y_test = Y[test]
# print x_train
# print y_train
mf = get_mf(dataset)
mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(x_train, y_train, mfc)
anf.trainHybridJangOffLine(epochs=2)
# for i in range(0,119):
#      print round(anf.fittedValues[i],3)
y_predicted = []

for i in range(len(y_test)):
    res = round(anf.fittedValues[y_test[i]],1)
    print res
    resx = 2-res
    print "\tresx1: ",resx
    if resx-round(resx,3)>0.5:
        print "\t\t gogo"
        resx = resx+1
    print "\t\t\tresx: ",resx
    y_predicted.append(int(round(resx,2)))
    #if abs(res-0) < abs(res -1) < abs(res -2):
    #    y_predicted.append(0)
    #elif abs(res-0) > abs(res -1) < abs(res -2):
    #    y_predicted.append(1)
    #elif abs(res-0) > abs(res-1) > abs(res-2):
    #    y_predicted.append(2)

trupred = 0
print y_test
print y_predicted
# print accuracy_score(y_test, y_predicted)*100, "%"
#check accuracy
for i in range(len(y_predicted)):
    if y_predicted[i] == y_test[i]:
        trupred +=1

print "Sum of TruePrediction: ",trupred
print truediv(trupred,len(y_test))*100, "%"
# anf.plotErrors()
# anf.plotMF(12, 30)
anf.plotResults()
# print round(anf.consequents[-1][0], 6)
# print round(anf.consequents[-2][0], 6)
# print round(anf.fittedValues[9][0], 6)
# if round(anf.consequents[-1][0], 6) == -5.275538 and round(anf.consequents[-2][0], 6) == -1.990703 and round(
#         anf.fittedValues[9][0], 6) == 0.002249:
#     print 'test is good'
# anf.plotErrors()
# anf.plotResults()
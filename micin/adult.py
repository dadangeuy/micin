from operator import truediv

import pandas as pd
from anfis import anfis
from anfis import membership
from sklearn.model_selection import train_test_split

from membership import get_mf
from dataset import load_adult, load_adult_reduced

dataset = load_adult()
x = dataset.data
y = dataset.target

# for train, test in kf.split(X, Y):
df = pd.DataFrame(x)
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.8)
y_test = y_test.tolist()
print "Length: ",len(y_test)
mf = get_mf(dataset)
mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(x_train, y_train, mfc)
anf.trainHybridJangOffLine(epochs=10)
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

trupred = 0
print y_test
print y_predicted

# check accuracy
for i in range(len(y_predicted)):
    if y_predicted[i] == y_test[i]:
        trupred +=1

print "Sum of TruePrediction: ",trupred
print truediv(trupred,len(y_test))*100, "%"
anf.plotResults()
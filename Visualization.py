import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy

reader=csv.reader(open("reduced_features.csv","r"),delimiter=",")
X=list(reader)
X=numpy.array(X)
X=X.astype(numpy.float)

qrs_duration = []
for person in X:
    qrs_duration.append(person[0])
sns.distplot(qrs_duration, kde=False)
plt.xlabel("QRS Duration")
plt.ylabel("No. of instances")
plt.show()

j_value = []
for person in X:
    j_value.append(person[1])
sns.distplot(j_value, kde=False)
plt.xlabel("J value")
plt.ylabel("No. of instances")
plt.show()

amp = []
for person in X:
    amp.append(person[4])
sns.distplot(amp, kde=False)
plt.xlabel("S' wave average width for channel V1")
plt.ylabel("No. of instances")
plt.show()
reader=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(reader)
Y=numpy.array(Y)
Y=Y.astype(numpy.int)
Y=Y.ravel()

sns.countplot(Y)
plt.xlabel("Arrythmia class")
plt.ylabel("No. of instances")
plt.show()


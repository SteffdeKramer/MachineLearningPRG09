"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9

Deze code is geschreven in Python3

Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- sklearn

"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import datasets

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import KMeans


def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn"""
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)


STUDENTNUMMER = "0908034" # TODO: aanpassen aan je eigen studentnummer

assert STUDENTNUMMER != "1234567", "Verander 1234567 in je eigen studentnummer"

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)


# UNSUPERVISED LEARNING

# haal clustering data op
kmeans_training = data.clustering_training()

# extract de x waarden
X = extract_from_json_as_np_array("x", kmeans_training)

# slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[...,0]
y = X[...,1]

# teken de punten
for i in range(len(x)):
    plt.plot(x[i], y[i], 'k.') # k = zwart

plt.axis([min(x), max(x), min(y), max(y)])

plt.show()


# TODO: print deze punten uit en omcirkel de mogelijke clusters

# TODO: ontdek de clusters mbv kmeans en teken een plot met kleurtjes


kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='r', zorder=10)

for i in set(kmeans.labels_):
    index = kmeans.labels_ == i
    plt.plot(X[index,0], X[index,1], 'o')
plt.show()

# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)
Y = extract_from_json_as_np_array("y", classification_training)

a = X[...,0]
b = X[...,1]

# teken de punten
for i in range(len(a)):

    if Y[i] == 0:
        plt.plot(a[i], b[i], 'k.')
    else:
        plt.plot(a[i], b[i], 'r.')

plt.axis([min(a), max(a), min(b), max(b)])
plt.show()

# vergelijk y waarde met predict y waarde


# TODO: leer de classificaties, en kijk hoe goed je dat gedaan hebt door je voorspelling te vergelijken
# TODO: met de werkelijke waarden

clf = LogisticRegression(max_iter=5000).fit(X, Y)
Y_pred = clf.predict(X[:, :])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
Y_d_pred = clf.predict(X[:, :])
predict_1 = accuracy_score(Y, Y_pred)
predict_2 = accuracy_score(Y, Y_d_pred)

print(predict_1)
print(predict_2)


# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)

# TODO: voorspel de Y-waarden

x = X_test[...,0]
y = X_test[...,1]

# teken de punten
for i in range(len(x)):
        plt.plot(x[i], y[i], 'b.')

plt.axis([min(x), max(x), min(y), max(y)])
plt.show()

# tryal

Y_test = clf.predict(X_test)

# aasdf

clf = LogisticRegression(max_iter=5000).fit(X_test, Y_test)
Y_test_pred = clf.predict(X_test[:, :])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_test, Y_test)
Y_d_test_pred = clf.predict(X_test[:, :])

predict_1_test = accuracy_score(Y_test, Y_test_pred)
predict_2_test = accuracy_score(Y_test, Y_d_test_pred)

print(predict_1_test)
print(predict_2_test)


Z = Y_test_pred # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed voorspeld is?

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie (test): " + str(classification_test))



import numpy as np
import sklearn.metrics.pairwise as pairwise

x = np.loadtxt('./wine.txt',dtype='float',delimiter=',',usecols = list(range(1,14)))
labels = np.loadtxt('./wine.txt',dtype='float',delimiter=',',usecols = (0,))

print(x.shape)
print(labels.shape)

# 矩阵专置
v = labels.transpose()
print('v.shape: ', v.shape)

# # Compute the rbf (gaussian) kernel between X and Y:
# # K(x, y) = exp(-gamma ||x-y||^2)
xtrain = pairwise.rbf_kernel(x, x, gamma=1.0/(2*490))
print(x.shape )
print('xtrain.shape: ', xtrain.shape)

D = np.diag(1.0 / np.sqrt(xtrain.sum(axis=1)))
nxtrain = D.dot(xtrain).dot(D)

print(nxtrain)
print(nxtrain.shape)


# Model Define
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout,Activation
from keras.optimizers import Adam, Nadam, RMSprop, Adadelta
from keras import regularizers
from keras.models import Model


def LeakyReLU(alpha):
    return Activation('sigmoid')


TModel = Sequential()
# 179 to 128
TModel.add(Dense(128, input_dim=178))
TModel.add(Activation('sigmoid'))
TModel.add(Dropout(0.8))

# 128 to 64
TModel.add(Dense(64, activity_regularizer = regularizers.l1(10e-5), name='embed', activation='sigmoid'))
TModel.add(Dropout(0.8))

# 64 to 128
TModel.add(Dense(128))
TModel.add(Activation('sigmoid'))
TModel.add(Dropout(0.8))

# 128 to 178
TModel.add(Dense(178))
TModel.add(Activation('sigmoid'))

TModel.compile(optimizer=Adam(lr=0.005, decay=1e-2), loss='mean_squared_error')
TModel.fit(nxtrain, nxtrain, nb_epoch=100, batch_size=2, shuffle=True)

mm = Model(input= TModel.input,output=TModel.get_layer('embed').output)
X = mm.predict(nxtrain)



print("X's shape: ", X.shape)

from sklearn.cluster import KMeans
kmeans_sae = KMeans(n_clusters=3, init='random', random_state=None,max_iter=500).fit(X)
kmeans_raw = KMeans(n_clusters=3, init='random', random_state=None,max_iter=500).fit(x)


def randindex(labels1,labels2):
    tp,tn,fp,fn = 0.0,0.0,0.0,0.0
    for point1 in range(len(labels1)):
        for point2 in range(len(labels2)):
            tp += 1 if labels1[point1] == labels1[point2] and labels2[point1] == labels2[point2] else 0
            tn += 1 if labels1[point1] != labels1[point2] and labels2[point1] != labels2[point2] else 0
            fp += 1 if labels1[point1] != labels1[point2] and labels2[point1] == labels2[point2] else 0
            fn += 1 if labels1[point1] == labels1[point2] and labels2[point1] != labels2[point2] else 0
    return (tp+tn) /(tp+tn+fp+fn)

# kmeans_sae is : 0.542860749905315
# kmeans_raw is : 0.720237343769726
print(kmeans_sae.labels_)
print('kmeans_sae is :', randindex(kmeans_sae.labels_, labels))
print('kmeans_raw is :', randindex(kmeans_raw.labels_, labels))





















































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
from keras.layers import Input, Dense, Activation, Dropout,Activation
from keras.optimizers import Adam, Nadam, RMSprop, Adadelta
from keras import regularizers
from keras.models import Model


def LeakyReLU(alpha):
    return Activation('sigmoid')

# First Dense Layout
inputs = Input(shape=(178,))
Encoder = Dense(128)(inputs)
Encoder = LeakyReLU(alpha=0.1)(Encoder)
Encoder = Dropout(0.8)(Encoder)

# AutoEncoder

Encoder = Dense(64, activity_regularizer = regularizers.l1(10e-5), name='embed', activation='sigmoid')(Encoder)
Encoder = Dropout(0.8)(Encoder)

Decoder = Dense(128)(Encoder)
Decoder = LeakyReLU(alpha=0.1)(Decoder)
Decoder = Dropout(0.8)(Decoder)

Decoder = Dense(178)(Decoder)
Decoder = LeakyReLU(alpha=0.1)(Decoder)

opt = Adam(lr=0.005, decay=1e-2)
ae = Model(input = inputs, output = Decoder)
ae.compile(optimizer=opt, loss='mean_squared_error')
ae.fit(nxtrain,nxtrain,nb_epoch= 100, batch_size = 2, shuffle=True)

mm = Model(input= ae.input,output=ae.get_layer('embed').output)
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

print(kmeans_sae.labels_)
print('kmeans_sae is :', randindex(kmeans_sae.labels_, labels))
print('kmeans_raw is :', randindex(kmeans_raw.labels_, labels))




















































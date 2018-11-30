
import numpy as np
import sklearn.metrics.pairwise as pairwise
from sklearn import preprocessing

x = np.loadtxt('./wine.txt',dtype='float',delimiter=',',usecols = list(range(1,14)))
labels = np.loadtxt('./wine.txt',dtype='float',delimiter=',',usecols = (0,))

# x = normalize(x)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(X=x)

print('x :', x)
print(x.shape)
print(labels.shape)

# 矩阵专置
v = labels.transpose()
print('v.shape: ', v.shape)

# # Compute the rbf (gaussian) kernel between X and Y:
# # K(x, y) = exp(-gamma ||x-y||^2)
# xtrain = pairwise.rbf_kernel(x, x, gamma=1.0/(2*490))
xtrain = pairwise.cosine_similarity(x, x)
print(xtrain)
print('xtrain.shape: ', xtrain.shape)

D = np.diag(1.0 / np.sqrt(xtrain.sum(axis=1)))
nxtrain = D.dot(xtrain).dot(D)

# nxtrain = xtrain

print(nxtrain)
print(nxtrain.shape)


# Model Define
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout,Activation
from keras.optimizers import Adam, Nadam, RMSprop, Adadelta
from keras import regularizers
from keras.models import Model
# from keras.constraints import maxnorm


KL = 0.0
p = 0.01



def l2_sae():
    pass


TModel = Sequential()

TModel.add(Dense(178, input_dim=178,name='first', activation='sigmoid'))
# TModel.add(Dropout(0.8))

TModel.add(Dense(128, name='second', activation='sigmoid'))
# TModel.add(Dropout(0.8))

TModel.add(Dense(64, name='embed', activation='sigmoid'))
# TModel.add(Dropout(0.8))

TModel.add(Dense(128, name='four', activation='sigmoid'))
# TModel.add(Dropout(0.8))

TModel.add(Dense(178, name='five', activation='sigmoid'))
# TModel.add(Dropout(0.8))



import keras.backend as K
from keract import get_activations
import tensorflow as tf

def sae_square_loss(beta, p):

    # 计算所有隐藏层稀疏指的和
    def layer_activations(layername):
        return tf.reduce_mean(TModel.get_layer(layername).output, axis=0)

    def sparse_result(rho, layername):
        rho_hat = layer_activations(layername)
        return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

    def KL(p):
        First = tf.reduce_sum(sparse_result(p, 'first'))
        Second = tf.reduce_sum(sparse_result(p, 'second'))
        Embed = tf.reduce_sum(sparse_result(p, 'embed'))
        return First + Second + Embed

    def loss(y_true, y_pred):
        # res = (K.sum(K.l2_normalize(y_true - y_pred))) + beta*(p * K.log(p / K.mean(activations)) + (1.0 - p)*K.log((1.0-p)/(1.0-K.mean(activations))))
        # res =  K.sqrt( K.sum( (y_true - y_pred)**2 ) ) + beta*(p * K.log(p / K.mean(activations)) + (1.0 - p)*K.log((1.0-p)/(1.0-K.mean(activations))))
        # res = K.sqrt(K.sum((y_true - y_pred)**2)) + beta * KL(p)
        res = tf.reduce_mean(tf.reduce_sum((y_true - y_pred)**2, axis=1)) + beta * KL(p)
        # res = tf.reduce_mean(tf.reduce_sum((y_true - y_pred)**2, axis=1))
        return res
    return loss

# TModel.compile(optimizer=Adam(lr=0.005, decay=1e-2), loss='mean_squared_error')
# embed
TModel.compile(optimizer=Adam(lr=0.005, decay=1e-2), loss=sae_square_loss(beta=0.01, p = 0.5))

TModel.fit(nxtrain, nxtrain, nb_epoch=100, batch_size=2, verbose=2)



mm = Model(input = TModel.input, output=TModel.get_layer('embed').output )
mm.summary()
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
# print(kmeans_raw.labels_)
print('kmeans_sae is :', randindex(kmeans_sae.labels_, labels))
# print('kmeans_raw is :', randindex(kmeans_raw.labels_, labels))




















































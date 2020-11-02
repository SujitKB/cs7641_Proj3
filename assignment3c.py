# Train with Neural Network

import pandas as pd
import numpy as np
import time as t
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss, mean_squared_error

from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture

from preprocess_data import breast_cancer_diagnostic

RANDOM_STATE = 42

# Classifier Comparison metrics
nn_accuracy_list   = [0.] * 7
nn_train_time = [0.] * 7
nn_query_time = [0.] * 7

X, Y = breast_cancer_diagnostic()

train_sizes = np.linspace(0.01, 1.0, 5)
idx=0

# Split data into Train and Test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=RANDOM_STATE)
# print (Y_train.unique(), Y_test.unique())

def learningcurve(mlp, X_train, Y_train, titlesuffix, fileprefix):

    train_sizes = np.linspace(0.01, 1.0, 5)

    train_sizes, train_scores, validation_scores = learning_curve(mlp,
                                                                  X_train, Y_train,
                                                                  train_sizes=train_sizes, cv=5)

    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis=1), marker='o', label='Training score')
    plt.plot(train_sizes, validation_scores.mean(axis=1), marker='o', label='Cross-validation score')
    plt.title('Learning Curve - Neural Network - ' + titlesuffix )
    plt.xlabel('No. of Training Instances')
    plt.ylabel("Classification Score")
    plt.legend()
    plt.grid()
    plt.savefig(fileprefix + '_ANN_LC.png')
    plt.clf()

def valcurve (mlp, X_train, Y_train, param_name, param_range, title, fileprefix):
    # Alpha (Regularization parameter)
    #alpha_range = np.logspace(-6, 4, 5)
    train_scores, test_scores = validation_curve(mlp, X_train, Y_train, param_name=param_name, param_range=param_range,
                                                 cv=5)

    plt.figure()
    plt.semilogx(param_range, np.mean(train_scores, axis=1), label='Training score')
    plt.semilogx(param_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Model Complexity Curve - ANN - ' + title)
    plt.xlabel(param_name)
    plt.ylabel("Classification Score")
    plt.grid()
    plt.savefig(fileprefix + '_ANN_VC.png')


# Base Classifier Model with tuned Hyperparameter from assignment 1
mlp = MLPClassifier(hidden_layer_sizes=(4, 3), alpha=10, learning_rate_init=0.001, random_state=RANDOM_STATE, max_iter=3000)

tStart = time.time()
mlp.fit(X_train, Y_train)
tEnd = time.time()
nn_train_time[idx] = tEnd - tStart

tStart = time.time()
Y_pred = mlp.predict(X_test)
tEnd = time.time()
nn_query_time[idx] = tEnd - tStart

accuracy_mlp = mlp.score(X_test, Y_test) * 100

nn_accuracy_list[idx] = accuracy_mlp
print('Accuracy of ANN with tuning is %.2f%%' % (accuracy_mlp))
learningcurve(mlp, X_train, Y_train, "Baseline (Hyp. Param. Tuned)", "baseline_tuned")

# ######################## PCA Classifier Model
idx+=1
# Data Transformation with reduced components
pca = PCA(n_components=14)
#X_transform = pca.fit_transform(X)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(4, 3), random_state=RANDOM_STATE, max_iter=3000)

mlp.fit(X_train_pca, Y_train)

Y_pred = mlp.predict(X_test_pca)

nn_accuracy = accuracy_score(Y_test, Y_pred)

print('Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

learningcurve(mlp, X_train_pca, Y_train, "PCA (No Tunning)", "pca_untune")

# Alpha (Regularization parameter)
alpha_range = np.logspace(-6, 4, 5)
valcurve(mlp, X_train_pca, Y_train, 'alpha', alpha_range, "PCA", 'PCA_Alpha')

# Learning_Rate
learningR_range = np.logspace(-6, 2, 5)
valcurve(mlp, X_train_pca, Y_train, 'learning_rate_init', learningR_range, "PCA", 'PCA_LearningRate')

# Tuned
mlp = MLPClassifier(hidden_layer_sizes=(4, 3), 
                    learning_rate_init=0.01, alpha=0.1,
                    random_state=RANDOM_STATE, max_iter=3000)

tStart = time.time()
mlp.fit(X_train_pca, Y_train)
tEnd = time.time()
nn_train_time[idx] = tEnd - tStart

tStart = time.time()
Y_pred = mlp.predict(X_test_pca)
tEnd = time.time()
nn_query_time[idx] = tEnd - tStart

nn_accuracy = accuracy_score(Y_test, Y_pred)
nn_accuracy_list[idx] = nn_accuracy * 100
print('PCA: Accuracy of neural network with hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))
learningcurve(mlp, X_train_pca, Y_train, "PCA (Hyp. Param. Tuned)", "pca_tuned")


# ######################## ICA Classifier Model
idx+=1

ica = ICA(n_components=3, random_state=RANDOM_STATE, whiten=True, max_iter=1000)
ica.fit(X_train)
X_train_ica = ica.transform(X_train)
X_test_ica = ica.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(4, 3), random_state=RANDOM_STATE, max_iter=3000)

mlp.fit(X_train_ica, Y_train)
Y_pred = mlp.predict(X_test_ica)
nn_accuracy = accuracy_score(Y_test, Y_pred)
print('ICA: Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

learningcurve(mlp, X_train_ica, Y_train, "ICA (No Tunning)", "ica_untune")

# Alpha (Regularization parameter)
alpha_range = np.logspace(-6, 4, 5)
valcurve(mlp, X_train_ica, Y_train, 'alpha', alpha_range, "ICA", 'ICA_Alpha')

# Learning_Rate
learningR_range = np.logspace(-6, 2, 5)
valcurve(mlp, X_train_ica, Y_train, 'learning_rate_init', learningR_range, "ICA", 'ICA_LearningRate')

# Tuned
mlp = MLPClassifier(hidden_layer_sizes=(4, 3),
                    learning_rate_init=0.1, alpha=0.0001,
                    #learning_rate_init=1e-2, alpha=1e-5,
                    random_state=RANDOM_STATE, max_iter=3000)

tStart = time.time()
mlp.fit(X_train_ica, Y_train)
tEnd = time.time()
nn_train_time[idx] = tEnd - tStart

tStart = time.time()
Y_pred = mlp.predict(X_test_ica)
tEnd = time.time()
nn_query_time[idx] = tEnd - tStart

nn_accuracy = accuracy_score(Y_test, Y_pred)
nn_accuracy_list[idx] = nn_accuracy * 100

print('ICA: Accuracy of neural network with hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))
learningcurve(mlp, X_train_ica, Y_train, "ICA (Hyp. Param. Tuned)", "ica_tuned")


# ######################## Gaussian Randomized Projection
idx+=1

grp = GaussianRandomProjection(n_components=3, random_state=RANDOM_STATE)
grp.fit(X_train)
X_train_grp = grp.transform(X_train)
X_test_grp = grp.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(4, 3), random_state=RANDOM_STATE, max_iter=3000)

mlp.fit(X_train_grp, Y_train)
Y_pred = mlp.predict(X_test_grp)
nn_accuracy = accuracy_score(Y_test, Y_pred)
print('GRP: Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

learningcurve(mlp, X_train_grp, Y_train, "GRP (No Tunning)", "grp_untune")

# Alpha (Regularization parameter)
alpha_range = np.logspace(-6, 4, 5)
valcurve(mlp, X_train_grp, Y_train, 'alpha', alpha_range, "Gaussian RP", 'GRP_Alpha')

# Learning_Rate
learningR_range = np.logspace(-6, 2, 5)
valcurve(mlp, X_train_grp, Y_train, 'learning_rate_init', learningR_range, "Gaussian RP", 'GRP_LearningRate')

# Tuned
mlp = MLPClassifier(hidden_layer_sizes=(4, 3),
                    learning_rate_init=10e-3, alpha=10e1,
                    random_state=RANDOM_STATE, max_iter=3000)

tStart = time.time()
mlp.fit(X_train_grp, Y_train)
tEnd = time.time()
nn_train_time[idx] = tEnd - tStart

tStart = time.time()
Y_pred = mlp.predict(X_test_grp)
tEnd = time.time()
nn_query_time[idx] = tEnd - tStart

nn_accuracy = accuracy_score(Y_test, Y_pred)
nn_accuracy_list[idx] = nn_accuracy * 100

print('GRP: Accuracy of neural network with hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))
learningcurve(mlp, X_train_grp, Y_train, "Gaussian RP (Hyp. Param. Tuned)", "grp_tuned")


# ######################## SelectKBest
idx+=1

skb = SelectKBest(score_func=chi2, k=15)
skb.fit(X_train,Y_train)
X_train_skb = skb.transform(X_train)
X_test_skb = skb.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(4, 3), random_state=RANDOM_STATE, max_iter=3000)

mlp.fit(X_train_skb, Y_train)
Y_pred = mlp.predict(X_test_skb)
nn_accuracy = accuracy_score(Y_test, Y_pred)
print('SelKBest: Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

learningcurve(mlp, X_train_skb, Y_train, "SelectKBest (No Tuning)", "skb_untune")

# Alpha (Regularization parameter)
alpha_range = np.logspace(-6, 4, 5)
valcurve(mlp, X_train_skb, Y_train, 'alpha', alpha_range, "SelectKBest", 'SKB_Alpha')

# Learning_Rate
learningR_range = np.logspace(-6, 2, 5)
valcurve(mlp, X_train_skb, Y_train, 'learning_rate_init', learningR_range, "SelectKBest", 'SKB_LearningRate')

# Tuned
mlp = MLPClassifier(hidden_layer_sizes=(4, 3),
                    #learning_rate_init=10e-3, alpha=10e1,
                    learning_rate_init=10e-2, alpha=100,
                    random_state=RANDOM_STATE, max_iter=3000)

tStart = time.time()
mlp.fit(X_train_skb, Y_train)
tEnd = time.time()
nn_train_time[idx] = tEnd - tStart

tStart = time.time()
Y_pred = mlp.predict(X_test_skb)
tEnd = time.time()
nn_query_time[idx] = tEnd - tStart

nn_accuracy = accuracy_score(Y_test, Y_pred)
nn_accuracy_list[idx] = nn_accuracy * 100

print('SelectKBest: Accuracy of neural network with hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))
learningcurve(mlp, X_train_skb, Y_train, "SelectKBest (Hyp. Param. Tuned)", "skb_tuned")


# ######################## K-Means
idx+=1

km = KMeans(n_clusters=2, max_iter=400, random_state=RANDOM_STATE)
km.fit(X_train)
X_train_km = km.transform(X_train)
X_test_km = km.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(4, 3), random_state=RANDOM_STATE, max_iter=3000)

mlp.fit(X_train_km, Y_train)
Y_pred = mlp.predict(X_test_km)
nn_accuracy = accuracy_score(Y_test, Y_pred)
print('KMeans: Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

learningcurve(mlp, X_train_km, Y_train, "K-Means (No Tuning)", "km_untune")

# Alpha (Regularization parameter)
alpha_range = np.logspace(-6, 4, 5)
valcurve(mlp, X_train_km, Y_train, 'alpha', alpha_range, "KMeans", 'KM_Alpha')

# Learning_Rate
learningR_range = np.logspace(-6, 2, 5)
valcurve(mlp, X_train_km, Y_train, 'learning_rate_init', learningR_range, "KMeans", 'KM_LearningRate')

# Tuned
mlp = MLPClassifier(hidden_layer_sizes=(4, 3),
                    #learning_rate_init=10e-3, alpha=10e1,
                    learning_rate_init=0.01, alpha=10,
                    random_state=RANDOM_STATE, max_iter=3000)

tStart = time.time()
mlp.fit(X_train_km, Y_train)
tEnd = time.time()
nn_train_time[idx] = tEnd - tStart

tStart = time.time()
Y_pred = mlp.predict(X_test_km)
tEnd = time.time()
nn_query_time[idx] = tEnd - tStart

nn_accuracy = accuracy_score(Y_test, Y_pred)
nn_accuracy_list[idx] = nn_accuracy * 100

print('KMeans: Accuracy of neural network with hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))
learningcurve(mlp, X_train_km, Y_train, "KMeans (Hyp. Param. Tuned)", "km_tuned")



# ######################## Expectation Maximization
idx+=1

gmm = GaussianMixture(n_components=2, max_iter=400, random_state=RANDOM_STATE)
gmm.fit(X_train)

X_train_gmm = gmm.predict_proba(X_train)
X_test_gmm = gmm.predict_proba(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(4, 3), random_state=RANDOM_STATE, max_iter=3000)

mlp.fit(X_train_gmm, Y_train)
Y_pred = mlp.predict(X_test_gmm)
nn_accuracy = accuracy_score(Y_test, Y_pred)
print('GMM: Accuracy of neural network without hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))

learningcurve(mlp, X_train_gmm, Y_train, "EM-GMM (No Tuning)", "gmm_untune")

# Alpha (Regularization parameter)
alpha_range = np.logspace(-6, 4, 5)
valcurve(mlp, X_train_gmm, Y_train, 'alpha', alpha_range, "EM-GMM", 'GMM_Alpha')

# Learning_Rate
learningR_range = np.logspace(-6, 2, 5)
valcurve(mlp, X_train_gmm, Y_train, 'learning_rate_init', learningR_range, "EM-GMM", 'GMM_LearningRate')

# Tuned
mlp = MLPClassifier(hidden_layer_sizes=(4, 3),
                    #learning_rate_init=0.01, alpha=1,
                    learning_rate_init=1.e-02, alpha=1.e-01,
                    random_state=RANDOM_STATE, max_iter=3000)

tStart = time.time()
mlp.fit(X_train_gmm, Y_train)
tEnd = time.time()
nn_train_time[idx] = tEnd - tStart

tStart = time.time()
Y_pred = mlp.predict(X_test_gmm)
tEnd = time.time()
nn_query_time[idx] = tEnd - tStart

nn_accuracy = accuracy_score(Y_test, Y_pred)
nn_accuracy_list[idx] = nn_accuracy * 100
print(alpha_range, learningR_range)
print('GMM: Accuracy of neural network with hyperparameter tuning is %.2f%%' % (nn_accuracy * 100))
learningcurve(mlp, X_train_gmm, Y_train, "EM-GMM (Hyp. Param. Tuned)", "gmm_tuned")

# Overall Comparison Plots

################ Comparison ###################
learners = ('Baseline', 'PCA', 'ICA', 'Gaussian RP', 'SelectKBest', 'KMeans', 'EM-GMM')
y_pos = np.arange(len(learners))
print(nn_accuracy_list)
print(nn_query_time)
print(nn_train_time)
## Accuracy Comparison
plt.figure(figsize=(10,15))
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.labelweight'] = 'bold'

plt.bar(y_pos, nn_accuracy_list, width=0.5)
plt.xticks(y_pos, learners, rotation=30)
plt.grid(True)
plt.ylim((90, 100))
plt.title('Comparison of Maximum Accuracy Score')
plt.ylabel('Accuracy')
plt.savefig('Accuracy.png')
plt.clf()

# ####################Training Time
plt.figure(figsize=(10,15))
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.labelweight'] = 'bold'

plt.bar(y_pos, nn_train_time, width=0.5)
plt.xticks(y_pos, learners, rotation=30)
plt.grid(True)
plt.ylim((0.1, 1.2))
plt.title('Comparison of Training Time')
plt.ylabel('Train Time (sec)')
plt.savefig('TrainTime.png')
plt.clf()

# ###################Query/Predict Time

plt.figure(figsize=(12,15))
plt.rcParams.update({'font.size':20})
plt.rcParams['axes.labelweight'] = 'bold'

plt.bar(y_pos, nn_query_time, width=0.5)
plt.xticks(y_pos, learners, rotation=30)
plt.grid(True)
plt.ylim((0.0001, 0.00125))
plt.title('Comparison of Query Time')
plt.ylabel('Query Time (seconds)')
plt.savefig('QueryTime.png')
plt.clf()


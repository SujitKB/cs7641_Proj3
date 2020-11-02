import pandas as pd
import numpy as np
import time as t
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss, mean_squared_error

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture

from preprocess_data import breast_cancer_diagnostic
from preprocess_data import vehicle_silhouettes


RANDOM_STATE = 42

def plot_scatter(km, X, plotfile, clusters):

    #caller: plot_scatter(km_bcw, X, 'bcw_scatter.png', 2)
    centroids = km.cluster_centers_

    # Plot the clustered data
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.scatter(X[km.labels_ == 0, 0], X[km.labels_ == 0, 1],
                c='green', label='cluster 1')
    plt.scatter(X[km.labels_ == 1, 0], X[km.labels_ == 1, 1],
                c='blue', label='cluster 2')

    if clusters == 4:
        plt.scatter(X[km.labels_ == 2, 0], X[km.labels_ == 2, 1],
                    c='orange', label='cluster 3')
        plt.scatter(X[km.labels_ == 3, 0], X[km.labels_ == 3, 1],
                    c='purple', label='cluster 4')

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=600,
                c='r', label='centroid')

    plt.legend()
    plt.xlim([-3, 5])
    plt.ylim([-3, 5])
    plt.xlabel('Eruption time in mins')
    plt.ylabel('Waiting time to next eruption')
    plt.title('Visualization of clustered data', fontweight='bold')
    ax.set_aspect('equal');
    plt.savefig(plotfile)

# ################################# Breast Cancer ################################## #
X1, Y1 = breast_cancer_diagnostic()

# ############################# Vehicle Silhouettes ################################ #
X2, Y2 = vehicle_silhouettes ("Vehicle_Silhouettes.csv")


def ClusterPlot(X1, X2, prefix):
    ssd1 = []
    ssd2 = []
    k_range = np.arange(1, 21)

    ######################################################################################
    # ################################# K-Means Clustering ############################# #
    for idx, k in enumerate(k_range):

        km1 = KMeans(n_clusters=k, max_iter=400, random_state=RANDOM_STATE)
        km1.fit(X1)
        ssd1.append(km1.inertia_)

        km2 = KMeans(n_clusters=k, max_iter=400, random_state=RANDOM_STATE)
        km2.fit(X2)
        ssd2.append(km2.inertia_)
        # print(km1.n_iter_, km2.n_iter_)

    # Plot elbow curve
    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.xlim([0, 20])
    #plt.ylim([-3, 5])
    plt.plot(k_range, ssd1, '-o')
    plt.xlabel('Number Of Clusters')
    plt.ylabel('Sum Of Squared Distance')
    plt.title('SSD v/s Cluster Size (Breast Cancer Diagnostic)', fontweight='bold')
    plt.savefig('bc_' + prefix + '_elbow.png')

    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.xlim([0, 20])
    #plt.ylim([-3, 5])
    plt.plot(k_range, ssd2, '-o')
    plt.xlabel('Number Of Clusters')
    plt.ylabel('Sum Of Squared Distance')
    plt.title('SSD v/s Cluster Size (Vehicle Silhouettes)', fontweight='bold')
    plt.savefig('vs_' + prefix + '_elbow.png')

    # ######################### Calculate silhouette_score ######################## #
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    # Breast Cancer Data
    for k in np.arange(2, 11):
        km1 = KMeans(n_clusters=k, max_iter=400, random_state=RANDOM_STATE)
        km1.fit(X1)
        silhouette_avg = silhouette_score(X1, km1.labels_)
        adj_mutu_info = adjusted_mutual_info_score(Y1, km1.labels_)
        print("For 'Breast Cancer data' with n_clusters =", k,
              "The average silhouette_score is :", silhouette_avg,
              " and AMI : ", adj_mutu_info)

    # Vehicle Silhouette
    for k in np.arange(2, 11):
        km2 = KMeans(n_clusters=k, max_iter=400, random_state=RANDOM_STATE)
        km2.fit(X2)
        silhouette_avg = silhouette_score(X2, km2.labels_)
        adj_mutu_info = adjusted_mutual_info_score(Y2, km2.labels_)
        print("For 'Vehicle Silhouette data' with n_clusters =", k,
              "The average silhouette_score is :", silhouette_avg,
              " and AMI : ", adj_mutu_info)

    ######################################################################################
    # ####################### Expectation Maximization Clustering ###################### #

    bic1 = []
    bic2 = []
    silh1 = []
    silh2 = []
    k_range = np.arange(2, 41)

    for idx, k in enumerate(k_range):
        em1 = GaussianMixture(n_components=k, max_iter=400, random_state=RANDOM_STATE)
        em1.fit(X1)
        bic1.append(em1.bic(X1))
        label = em1.predict(X1)
        silh1.append(silhouette_score(X1, label))
        adj_mutu_info = adjusted_mutual_info_score(Y1, label)
        print("Breast Cancer, cluster= ", k, "BIC= ", bic1[idx], "AMI= ", adj_mutu_info)
        # homog_score = homogeneity_score(ufcY, label)
        # silh_EM[cluster] = sil_coeff
        # homog_EM[cluster] = homog_score
        # log_likelihood_EM[cluster] = gmm.score(ufcX)
        # print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, silh1[idx]))
        # print("For n_clusters={}, The homogeneity_score is {}".format(cluster, homog_score))
        # print("For n_clusters={}, The log_likelihood score is {}".format(cluster, log_likelihood_EM[cluster]))

    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.grid()
    plt.plot(k_range, bic1)
    plt.title("BIC v/s Components (Breast Cancer)")
    plt.xlabel("Number of Components")
    plt.ylabel("BIC")
    plt.savefig('bc_' + prefix + '_EM_BIC.png')

    '''plt.clf()
    plt.figure(figsize=(6, 6))
    plt.grid()
    plt.plot(k_range, silh1)
    plt.title("Silhouette Score v/s Components (Breast Cancer)")
    plt.xlabel("Number of Components")
    plt.ylabel("Silhouette Score")
    plt.savefig('bc_' + prefix + '_EM_Silh.png')'''

    for idx, k in enumerate(k_range):
        em2 = GaussianMixture(n_components=k, max_iter=400, random_state=RANDOM_STATE)
        em2.fit(X2)
        bic2.append(em2.bic(X2))
        label = em2.predict(X2)
        silh2.append(silhouette_score(X2, label))
        adj_mutu_info = adjusted_mutual_info_score(Y2, label)
        # homog_score = homogeneity_score(ufcY, label)
        # silh_EM[cluster] = sil_coeff
        # homog_EM[cluster] = homog_score
        # log_likelihood_EM[cluster] = gmm.score(ufcX)
        #print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, silh2[idx]))
        # print("For n_clusters={}, The homogeneity_score is {}".format(cluster, homog_score))
        #print("For n_clusters={}, The log_likelihood score is {}".format(cluster, log_likelihood_EM[cluster]))
        print("Vehicle Silh, cluster= ", k, "BIC= ", bic2[idx], "AMI= ", adj_mutu_info)

    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.title("BIC v/s Components (Vehicle Silhouette)")
    plt.plot(k_range, bic2)
    plt.xlabel("Number of Components")
    plt.ylabel("BIC")
    plt.savefig('vs_' + prefix + '_EM_BIC.png')


# This section will implement 4 different dimensionality reduction techniques on both the phishing and the banking dataset. Then, k-means and EM clustering will be performed for each (dataset * dim_reduction) combination to see how the clustering compares with using the full datasets. The 4 dimensionality reduction techniques are:
# - Principal Components Analysis (PCA). Optimal number of PC chosen by inspecting % variance explained and the eigenvalues.
# - Independent Components Analysis (ICA). Optimal number of IC chosen by inspecting kurtosis.
# - Random Components Analysis (RCA) (otherwise known as Randomized Projections). Optimal number of RC chosen by inspecting reconstruction error.
# - Random Forest Classifier (RFC). Optimal number of components chosen by feature importance.

# ########################### PCA ################################## #
# Standardize data before performing PCA
X1 = scale(X1)
X2 = scale(X2)
pca1 = PCA(random_state=RANDOM_STATE)
pca1.fit(X1)

pca2 = PCA(random_state=RANDOM_STATE)
pca2.fit(X2)


plt.figure()
plt.plot(np.arange(1, pca1.explained_variance_ratio_.size + 1), np.cumsum(pca1.explained_variance_ratio_))
plt.xticks(np.arange(1, pca1.explained_variance_ratio_.size + 1))
plt.xlabel('Components')
plt.ylabel('Cumulative Variance')
plt.title('Breast Cancer - PCA')
plt.grid()
plt.savefig('bc_pca_var_cum.png')

plt.figure()
plt.plot(np.arange(1, pca2.explained_variance_ratio_.size + 1), np.cumsum(pca2.explained_variance_ratio_))
plt.xticks(np.arange(1, pca2.explained_variance_ratio_.size + 1))
plt.xlabel('Components')
plt.ylabel('Cumulative Variance')
plt.title('Vehicle Silhouettes - PCA')
plt.grid()
plt.savefig('vs_pca_var_cum.png')

# Choose the no. of components that capture 90% of the variance
component1_90pct = 14       # Breast cancer - 98%
component2_90pct = 9        # Vehicle Silhouettes - 98%

# Data Transformation with reduced components
pca1 = PCA(n_components=component1_90pct)
X1_transform = pca1.fit_transform(X1)
pca2 = PCA(n_components=component2_90pct)
X2_transform = pca2.fit_transform(X2)
ClusterPlot(X1_transform, X2_transform, 'PCA')


# ########################### ICA ################################## #
def run_ICA(X, Y, title):

    dims = list(np.arange(1, (X.shape[1])))
    dims.append(X.shape[1])
    kurt = []
    #print(dims)
    for dim in dims:
        ica = ICA(n_components=dim, random_state=RANDOM_STATE, whiten=True, max_iter=1000)

        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis - " + title)
    plt.xlabel("Independent Components")
    plt.ylabel("Average Kurtosis Across Independent Components")
    plt.plot(dims, kurt, '-o')
    plt.grid(False)
    plt.savefig(title + '_ICA_Kurtosis.png')

# Breast Cancer
run_ICA (X1, Y1, "Breast_Cancer")
k = 3
ica = ICA(n_components=k, random_state=RANDOM_STATE, whiten=True, max_iter=1000)
ica_transform1 = ica.fit_transform(X1)

# Vehicle Silhouette
run_ICA (X2, Y2, "Vehicle_Silhouette")
k = 4
ica = ICA(n_components=k, random_state=RANDOM_STATE, whiten=True, max_iter=1000)
ica_transform2 = ica.fit_transform(X2)

ClusterPlot(ica_transform1, ica_transform2, 'ICA')


# ########################### Randomized Projections ############################ #

def run_RP (X, Y, title):

    #X = scale(X)
    dims = list(np.arange(1, X.shape[1]))
    dims.append(X.shape[1])

    # Reconstruction Error
    rmse = []
    for idx, dim in enumerate(dims):
        rp = GaussianRandomProjection(n_components=dim, random_state=RANDOM_STATE)
        X_transform = rp.fit_transform(X)

        # reconstruction
        A = np.linalg.pinv(rp.components_.T)
        reconstructed = np.dot(X_transform, A)
        rc_err = mean_squared_error(X, reconstructed)
        rmse.append(rc_err)
    #     print(dim, ": ", rc_err)
    plt.figure()
    plt.grid()
    plt.title("Gaussian Randomized Projection - RMSE - " + title)
    plt.plot(dims, rmse,'-o')
    plt.xlabel("Number Of Dimensions")
    plt.ylabel("RMSE")
    plt.savefig(title + '_GRP_RMSE.png')

run_RP(X1, Y1, "Breast_Cancer")
dim=14
rp = GaussianRandomProjection(n_components=dim, random_state=RANDOM_STATE)
X1_transform = rp.fit_transform(X1)

run_RP(X2, Y2, "Vehicle_Silhouette")
dim=12
rp = GaussianRandomProjection(n_components=dim, random_state=RANDOM_STATE)
X2_transform = rp.fit_transform(X2)

ClusterPlot(X1_transform, X2_transform, 'GRP')


# ########################### Feature Selection (SelectKBest) ##############################
def run_SelectKBest (X, Y, best_feature_num, title):

    # apply SelectKBest class to extract top 10 best features
    best_features = SelectKBest(score_func=chi2, k=best_feature_num)
    fit = best_features.fit(X, Y)
    X_transform = best_features.fit_transform(X, Y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    # Concatenate the two data frames for better visualization
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Features', 'Score']  # naming the dataframe columns
    #print(X_transform.shape)
    #print(feature_scores.nlargest(best_feature_num, 'Score'))  # print n best features
    return X_transform

X1_transform = run_SelectKBest (X1, Y1, 15, "Breast_Cancer")

X2_transform = run_SelectKBest (X2, Y2, 9, "Vehicle_Silhouette")
ClusterPlot(X1_transform, X2_transform, 'SelectKBest')


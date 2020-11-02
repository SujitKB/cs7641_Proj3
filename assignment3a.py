import pandas as pd
import numpy as np
import time as t
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture

from preprocess_data import breast_cancer_diagnostic
from preprocess_data import vehicle_silhouettes


RANDOM_STATE = 42
ssd1 = []
ssd2 = []
k_range = np.arange(1, 21)
'''
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
'''

# ################################# Breast Cancer ################################## #
X1, Y1 = breast_cancer_diagnostic()

# ############################# Vehicle Silhouettes ################################ #
X2, Y2 = vehicle_silhouettes ("Vehicle_Silhouettes.csv")

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
plt.savefig('bc_elbow.png')

plt.figure(figsize=(6, 6))
plt.grid(True)
plt.xlim([0, 20])
#plt.ylim([-3, 5])
plt.plot(k_range, ssd2, '-o')
plt.xlabel('Number Of Clusters')
plt.ylabel('Sum Of Squared Distance')
plt.title('SSD v/s Cluster Size (Vehicle Silhouettes)', fontweight='bold')
plt.savefig('vs_elbow.png')

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
    print("Breast Cancer ' with n_clusters =", k,
          "Avg. silhouette_score: ", silhouette_avg,
          "Adj. Mutual Info: ", adj_mutu_info)

# Vehicle Silhouette
for k in np.arange(2, 11):
    km2 = KMeans(n_clusters=k, max_iter=400, random_state=RANDOM_STATE)
    km2.fit(X2)
    silhouette_avg = silhouette_score(X2, km2.labels_)
    adj_mutu_info = adjusted_mutual_info_score(Y2, km2.labels_)
    print("Vehicle Silhouette with n_clusters =", k,
          "Avg. Silhouette Score: ", silhouette_avg,
          "Adj. Mutual Info: ", adj_mutu_info)

for i, k in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # Run the Kmeans algorithm
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(X1)
    centroids = km.cluster_centers_

    # Get silhouette samples
    silhouette_vals = silhouette_samples(X1, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02);

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
    # homog_score = homogeneity_score(ufcY, label)
    # silh_EM[cluster] = sil_coeff
    # homog_EM[cluster] = homog_score
    # log_likelihood_EM[cluster] = gmm.score(ufcX)
    print("For BC, n_clusters={}, Silh Coeff. is {}, BIC is {}, AMI is {}".format(k, silh1[idx], bic1[idx],adj_mutu_info))
    # print("For n_clusters={}, The homogeneity_score is {}".format(cluster, homog_score))
    # print("For n_clusters={}, The log_likelihood score is {}".format(cluster, log_likelihood_EM[cluster]))

plt.clf()
plt.figure(figsize=(6, 6))
plt.grid()
plt.plot(k_range, bic1)
plt.title("BIC v/s Components (Breast Cancer)")
plt.xlabel("Number of Components")
plt.ylabel("BIC")
plt.savefig("bc_EM_BIC.png")

plt.clf()
plt.figure(figsize=(6, 6))
plt.grid()
plt.plot(k_range, silh1)
plt.title("Silhouette Score v/s Components (Breast Cancer)")
plt.xlabel("Number of Components")
plt.ylabel("Silhouette Score")
plt.savefig("bc_EM_Silh.png")

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
    print("For VS, n_clusters={}, Silhouette Coefficient is {}, BIC is {}, AMI is {}".format(k, silh2[idx], bic2[idx],adj_mutu_info))
    # print("For n_clusters={}, The homogeneity_score is {}".format(cluster, homog_score))
    # print("For n_clusters={}, The log_likelihood score is {}".format(cluster, log_likelihood_EM[cluster]))

plt.clf()
plt.figure(figsize=(6, 6))
plt.grid(True)
plt.title("BIC v/s Components (Vehicle Silhouette)")
plt.plot(k_range, bic2)
plt.xlabel("Number of Components")
plt.ylabel("BIC")
plt.savefig("vs_EM_BIC.png")

plt.clf()
plt.figure(figsize=(6, 6))
plt.grid()
plt.plot(k_range, silh2)
plt.title("Silhouette Score v/s Components (Vehicle Silhouette)")
plt.xlabel("Number of Components")
plt.ylabel("Silhouette Score")
plt.savefig("vs_EM_Silh.png")
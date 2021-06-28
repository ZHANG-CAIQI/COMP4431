from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Read in the Iris Data set

iris = datasets.load_iris()


# Implement K-means with sklearn and find the centroids for 3 clusters and 4 clusters.

print(KMeans(n_clusters=3).fit(iris.data).cluster_centers_)

kmeans_3_clusters = KMeans(n_clusters=3).fit(iris.data[:, 2:4])

plt.figure(figsize=(6, 5))
plt.title("sdfasdfad")
# plot all samples with different color
plt.scatter(iris.data[:, 2], iris.data[:, 3], s=150, c=kmeans_3_clusters.labels_)
plt.scatter(kmeans_3_clusters.cluster_centers_[:, 0],
            kmeans_3_clusters.cluster_centers_[:, 1],
            marker='*',
            s=150,
            color='blue',
            label='Centers')  # plot two centers
plt.legend(loc='best')  # plot legend
plt.xlabel('column 3')
plt.ylabel('column 4')

print(KMeans(n_clusters=4).fit(iris.data).cluster_centers_)

kmeans_4_clusters = KMeans(n_clusters=4).fit(iris.data[:, 2:4])

plt.figure(figsize=(6, 5))
# plot all samples with different color
plt.scatter(iris.data[:, 2], iris.data[:, 3], s=150, c=kmeans_4_clusters.labels_)
plt.scatter(kmeans_4_clusters.cluster_centers_[:, 0],
            kmeans_4_clusters.cluster_centers_[:, 1],
            marker='*',
            s=150,
            color='blue',
            label='Centers')  # plot two centers
plt.legend(loc='best')  # plot legend
plt.xlabel('column 3')
plt.ylabel('column 4')

# Normalize your data and repeat step2 and step3
normalized_iris = preprocessing.normalize(iris.data)

print(KMeans(n_clusters=3).fit(normalized_iris).cluster_centers_)

kmeans_3_clusters_nomalized = KMeans(n_clusters=3).fit(normalized_iris[:, 2:4])

plt.figure(figsize=(6, 5))
# plot all samples with different color
plt.scatter(normalized_iris[:, 2], normalized_iris[:, 3], s=150, c=kmeans_3_clusters_nomalized.labels_)
plt.scatter(kmeans_3_clusters_nomalized.cluster_centers_[:, 0],
            kmeans_3_clusters_nomalized.cluster_centers_[:, 1],
            marker='*',
            s=150,
            color='blue',
            label='Centers')  # plot two centers
plt.legend(loc='best')  # plot legend
plt.xlabel('column 3')
plt.ylabel('column 4')

print(KMeans(n_clusters=4).fit(normalized_iris).cluster_centers_)

kmeans_4_clusters_nomalized = KMeans(n_clusters=4).fit(normalized_iris[:, 2:4])
print(kmeans_4_clusters_nomalized.labels_)

plt.figure(figsize=(6, 5))
# plot all samples with different color
plt.scatter(normalized_iris[:, 2], normalized_iris[:, 3], s=150, c=kmeans_4_clusters_nomalized.labels_)
plt.scatter(kmeans_4_clusters_nomalized.cluster_centers_[:, 0],
            kmeans_4_clusters_nomalized.cluster_centers_[:, 1],
            marker='*',
            s=150,
            color='blue',
            label='Centers')  # plot two centers
plt.legend(loc='best')  # plot legend
plt.xlabel('column 3')
plt.ylabel('column 4')

plt.show()
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class KMeansClustering:

    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroid):
        return np.sqrt(np.sum((centroid - data_point) ** 2))

    def fit(self, X, max_iterations=200):
        global y
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))

        for _ in range(max_iterations):
            y = []

            for dataPoint in X:
                distances = [KMeansClustering.euclidean_distance(dataPoint, centroid) for centroid in self.centroids]
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)

            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y


data_points = []
with open('driver-data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')

    next(csvreader, None)

    for row in csvreader:
        try:
            data_point = np.array([float(value) for value in row[1:]])
            data_points.append(data_point)
        except ValueError as e:
            continue

data_points = np.array(data_points)

# data_points = np.random.randint(100, size=(50000, 2))
kmeans = KMeansClustering(5)
labels = kmeans.fit(data_points)


def show_2d():
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)),
                marker="*", s=200)
    plt.show()


def show_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data_points[:, 0]
    y = data_points[:, 1]
    z = data_points[:, 2]

    scatter = ax.scatter(x, y, z, c=labels)

    ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], kmeans.centroids[:, 2],
                c=range(len(kmeans.centroids)), marker="*", s=200)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

show_2d()

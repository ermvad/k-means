import numpy
import math
import csv
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

filename = "resource/dots.csv"
clusters = 4
epsilon = 0.1


def euclidian(a, b):
    return math.sqrt(numpy.sum((a - b) ** 2))


def kmeans(data, cl, eps=1.):
    centers = numpy.random.permutation(numpy.unique(data, axis=0))[:cl]
    while True:
        labels = numpy.zeros(data.shape[0])
        for d in range(data.shape[0]):
            e = numpy.zeros((0, 2))
            for c in range(cl):
                e = numpy.append(e, [[euclidian(data[d, :], centers[c, :]), c]], axis=0)
            labels[d] = e[numpy.argmin(e[:, 0])][1]
        new_centers = numpy.array([data[labels == i, :].mean(axis=0) for i in range(cl)])
        if abs(numpy.sum(new_centers - centers)) < eps:
            break
        centers = new_centers
    return new_centers


def main():
    cords = numpy.zeros((0, 2))
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            cords = numpy.append(cords, [[int(row[0]), int(row[1])]], axis=0)
    cords,y = make_blobs(n_samples=300,centers=4,cluster_std=0.60,random_state=0)
    kmeans_centers = kmeans(cords, clusters, epsilon)
    plt.scatter(cords[:, 0], cords[:, 1], color='g')
    plt.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], color='r')
    plt.show()


if __name__ == "__main__":
    main()
import numpy
import math
import csv
import matplotlib.pyplot as plt

filename = "dots.csv"
clusters = 3
epsilon = 2


def euclidian(a, b):
    return math.sqrt(numpy.sum((a - b)**2))


def kmeans(data, cl, eps=1):
    centers = data[:cl]
    
    while True:
        labels = numpy.zeros(data.shape[0])
        for d in range(data.shape[0]):
            e = numpy.zeros((0, 2))
            for c in range(cl):
                e = numpy.append(e, [[euclidian(data[d,:], centers[c,:]), c]], axis=0)
            labels[d] = e[numpy.argmin(e[:, 0])][1]
        new_centers = numpy.array([data[labels == i, :].mean(0) for i in range(cl)])
        if numpy.mean(new_centers - centers) < eps:
            break
        centers = new_centers
    return new_centers


def main():
    cords = numpy.zeros((0, 2))
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            cords = numpy.append(cords, [[int(row[0]), int(row[1])]], axis=0)
    
    kmeans_centers = kmeans(cords, clusters, epsilon)
    plt.scatter(cords[:, 0], cords[:, 1], color='g')
    plt.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], color='r')
    plt.show()


if __name__ == "__main__":
    main()

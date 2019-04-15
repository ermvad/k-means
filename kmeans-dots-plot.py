import numpy
import math
import csv
import matplotlib.pyplot as plt

filename = "resource/dots.csv"
clusters = 3
epsilon = 1


def make_plot(d1, d2, i):
    plt.scatter(d1[:, 0], d1[:, 1], color='g')
    plt.scatter(d2[:, 0], d2[:, 1], color='r')
    plt.savefig('imagess/' + str(i) + '.png', transparent=False, frameon=False, quality=100)
    plt.clf()


def euclidian(a, b):
    return math.sqrt(numpy.sum((a - b)**2))


def kmeans(data, cl, eps=1):
    iteration = 0
    centers = numpy.random.permutation(numpy.unique(data, axis=0))[:cl]
    make_plot(data, centers, iteration)
    while True:
        labels = numpy.zeros(data.shape[0])
        for d in range(data.shape[0]):
            e = numpy.zeros((0, 2))
            for c in range(cl):
                e = numpy.append(e, [[euclidian(data[d,:], centers[c,:]), c]], axis=0)
            labels[d] = e[numpy.argmin(e[:, 0])][1]
        new_centers = numpy.array([data[labels == i, :].mean(0) for i in range(cl)])
        iteration += 1
        if numpy.mean(new_centers - centers) < eps:
            make_plot(data, new_centers, iteration)
            break
        centers = new_centers
        make_plot(data, centers, iteration)
    print('Total iterations: ' + str(iteration))
    return new_centers


def main():
    cords = numpy.zeros((0, 2))
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            cords = numpy.append(cords, [[int(row[0]), int(row[1])]], axis=0)
    
    kmeans_centers = kmeans(cords, clusters, epsilon)


if __name__ == "__main__":
    main()

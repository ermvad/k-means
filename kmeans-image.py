import numpy
import math
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

clusters = 3
epsilon = 0.1


def euclidian(a, b):
    return math.sqrt(numpy.sum((a - b)**2))


def kmeans(data, cl, eps=1.):
    centers = numpy.random.permutation(numpy.unique(data, axis=0))[:cl]
    while True:
        labels = numpy.zeros(data.shape[0])
        for d in range(data.shape[0]):
            e = numpy.zeros((0, 2))
            for c in range(cl):
                e = numpy.append(e, [[euclidian(data[d,:], centers[c,:]), c]], axis=0)
            labels[d] = e[numpy.argmin(e[:, 0])][1]
        new_centers = numpy.array([data[labels == i, :].mean(0) for i in range(cl)], dtype=int)
        if abs(numpy.sum(new_centers - centers)) < eps:
            break
        centers = new_centers
    return new_centers


def main():
    numpy.random.seed()
    
    img = Image.open("resource/image.jpg")
    img.show()
    img_rgb = img.convert("RGB")
    pixels = numpy.array(img_rgb.getdata())
    
    kmeans_centers = kmeans(pixels, clusters, epsilon)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], color='g')
    ax.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], kmeans_centers[:, 2], color='r')
    
    for i in range(kmeans_centers.shape[0]):
        result = Image.new("RGB", (50,50), (kmeans_centers[i][0], kmeans_centers[i][1], kmeans_centers[i][2]))
        result.show()
    plt.show()
            

if __name__ == "__main__":
    main()

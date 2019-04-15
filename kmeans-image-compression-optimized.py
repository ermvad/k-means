import numpy
import math
import matplotlib.pyplot as plt
from PIL import Image

clusters = 16
epsilon = 2


def euclidian(a, b):
    return math.sqrt(numpy.sum((a - b)**2))


def kmeans_with_copression(data, cl, eps=1):
    centers = data[:cl]
    centers += numpy.random.randint(0,10)
    img = data.copy()
    while True:
        labels = numpy.zeros(data.shape[0])
        for d in range(data.shape[0]):
            e = numpy.zeros((0, 2))
            for c in range(cl):
                e = numpy.append(e, [[euclidian(data[d,:], centers[c,:]), c]], axis=0)
            labels[d] = e[numpy.argmin(e[:, 0])][1]
        new_centers = numpy.array([data[labels == i, :].mean(0) for i in range(cl)], dtype=int)
        if numpy.mean(new_centers - centers) < eps:
            break
        centers = new_centers
    for i in range(img.shape[0]):
        e = numpy.zeros((0, 2))
        for c in range(centers.shape[0]):
            e = numpy.append(e, [[euclidian(img[i, :], new_centers[c, :]), c]], axis=0)
        img[i] = new_centers[int(e[numpy.argmin(e[:, 0])][1]), :]
    return new_centers, img


def main():
    numpy.random.seed()
    
    img = Image.open("image.jpg")
    width, height = img.size
    img_rgb = img.convert("RGB")
    pixels = numpy.array(img_rgb.getdata())

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(pixels.reshape((width, height, 3)))

    kmeans_centers,compressed_image = kmeans_with_copression(pixels, clusters, epsilon)

    for c in range(clusters):
        plt.figure(2)
        plt.subplot(1,clusters,c+1)
        plt.axis('off')
        plt.imshow(kmeans_centers[c].reshape((1, 1, 3)))


    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(compressed_image.reshape((width, height, 3)))

    plt.show()
            

if __name__ == "__main__":
    main()

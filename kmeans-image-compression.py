import numpy
import math
import matplotlib.pyplot as plt
from PIL import Image
clusters = 8
epsilon = 0.1


def euclidian(a, b):
    return math.sqrt(numpy.sum((a - b)**2))


def kmeans(data, cl, eps=1.):
    centers = numpy.random.permutation(numpy.unique(data, axis=0))[:cl]
    while True:
        labels = numpy.zeros(data.shape[0], dtype=int)
        for d in range(data.shape[0]):
            e = numpy.zeros((0, 2))
            for c in range(cl):
                e = numpy.append(e, [[euclidian(data[d,:], centers[c,:]), c]], axis=0)
            labels[d] = e[numpy.argmin(e[:, 0])][1]
        new_centers = numpy.array([data[labels == i, :].mean(axis=0) for i in range(cl)], dtype=int)
        if numpy.all(new_centers==centers):
            break
        centers = new_centers
    return new_centers


def compress(img, centers):
    for i in range(img.shape[0]):
        e = numpy.zeros((0, 2))
        for c in range(centers.shape[0]):
            e = numpy.append(e, [[euclidian(img[i, :], centers[c, :]), c]], axis=0)
        img[i] = centers[int(e[numpy.argmin(e[:, 0])][1]), :]
    return img


def main():
    numpy.random.seed()
    
    img = Image.open("resource/image4.jpg")
    width, height = img.size
    img_rgb = img.convert("RGB")
    pixels = numpy.array(img_rgb.getdata())

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(pixels.reshape((height, width, 3)))

    kmeans_centers = kmeans(pixels, clusters, epsilon)
    print(kmeans_centers)

    for c in range(clusters):
        plt.figure(2)
        plt.subplot(1, clusters, c+1)
        plt.axis('off')
        plt.imshow(kmeans_centers[c].reshape((1, 1, 3)))

    pixels_compressed = compress(pixels, kmeans_centers)

    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(pixels_compressed.reshape((height, width, 3)))

    plt.show()
            

if __name__ == "__main__":
    main()

import numpy
import math
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_sample_image

clusters = 16
epsilon = 2


def euclidian(a, b):
    return math.sqrt(numpy.sum((a - b)**2))


def kmeans(data, cl, eps=1):
    centers = data[:cl]
    centers += numpy.random.randint(0,10)
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


def compress(img, centers):
    for i in range(img.shape[0]):
        e = numpy.zeros((0, 2))
        for c in range(centers.shape[0]):
            e = numpy.append(e, [[euclidian(img[i, :], centers[c, :]), c]], axis=0)
        img[i] = centers[int(e[numpy.argmin(e[:, 0])][1]), :]
    return img


def main():
    numpy.random.seed()
    
    img = Image.open("image.jpg")
    width, height = img.size
    img_rgb = img.convert("RGB")
    pixels = numpy.array(img_rgb.getdata())

    plt.subplot(121)
    plt.imshow(pixels.reshape((width, height, 3)))

    kmeans_centers = kmeans(pixels, clusters, epsilon)

    pixels_compressed = compress(pixels, kmeans_centers)

    plt.subplot(122)
    plt.imshow(pixels_compressed.reshape((width, height, 3)))

    plt.show()
            

if __name__ == "__main__":
    main()

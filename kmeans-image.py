import numpy
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

clusters = 3
epsilon = 2


def euclidian(a, b):
    return numpy.linalg.norm(a - b)


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
        if(numpy.mean(new_centers - centers) < eps):
            break
        centers = new_centers
    return new_centers


def main():
    numpy.random.seed()
    
    img = Image.open("image.jpg")
    img.show()
    img_rgb = img.convert("RGB")
    pixels = numpy.array(img_rgb.getdata())
    
    kmeans_centers = kmeans(pixels, clusters, epsilon)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pixels[:, 0], pixels[:, 1], pixels[:, 2], color='g')
    ax.scatter(kmeans_centers[:, 0], kmeans_centers[:, 1], kmeans_centers[:, 2], color='r')
    
    for i in range(kmeans_centers.shape[0]):
        result = Image.new("RGB", (20,20), (int(kmeans_centers[i][0]), int(kmeans_centers[i][1]), int(kmeans_centers[i][2])))
        result.show()
    plt.show()
            

if __name__ == "__main__":
    main()

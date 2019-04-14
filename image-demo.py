import numpy
import matplotlib.pyplot as plt
from PIL import Image
thickness = 5
color = [255, 0, 0]


def main():
    img = Image.open("resource/image.jpg")
    img.show()
    img_rgb = img.convert("RGB")
    width, height = img.size
    pixels = numpy.array(img_rgb.getdata())
    for i in range(0, width):
        pixels[i] = color
    plt.imshow(pixels.reshape((width, height, 3)))
    plt.show()


if __name__ == "__main__":
    main()

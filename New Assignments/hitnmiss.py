import matplotlib.pyplot as ppt
import numpy as np


def morph(image, kernel):
    # leaving 2 pixels margin boundary
    l = int(kernel.shape[0])
    b = int(kernel.shape[1])

    mul_image = np.zeros_like(image)

    x = 0
    y = 0
    pixels = []
    image2 = np.zeros_like(image)

    while ((x + b) < image.shape[1]) & ((y + l) < image.shape[0]):
        mul_image[y:y + l, x:x + b] = np.multiply(image[y:y + l, x:x + b], kernel)

        # considering only corner pixel
        # (center of kernel)
        count = 0

        for x1 in range(x, x+l):
            for y1 in range(y, y+b):

                print(mul_image[x1, y1])
                if mul_image[y1, x1] == 255:
                    count += 1

                if count == 3:
                    pixels.append([y1, x1])
                    count = 0

        x += 1
        y += 1
        print(pixels)
        #mul_image[y:y + l, x:x + b] = 0
        #mul_image[int((y + l) / 2), int((x + b) / 2)] = 255

    for [y2, x2] in pixels:
        image2[y2, x2] = 255

    return image2


class hitnmiss:

    def __init__(self):
        self.initUI()

    def initUI(self):
        ppt.figure(1)

        kernel = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 255, 255, 255, 255, 255, 255, 0, 0],
                           [0, 255, 255, 0, 0, 255, 0, 255, 0],
                           [0, 0, 0, 255, 0, 255, 255, 255, 0],
                           [0, 255, 255, 0, 255, 255, 255, 255, 0],
                           [0, 255, 255, 0, 255, 0, 255, 255, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                          )

        ppt.subplot(111)
        ppt.xlabel("kernel")
        ppt.imshow(kernel, cmap='gray')

        # defining the kernel
        kernel_bottomleft = np.array(([-1, 1, -1], [0, 1, 1], [0, 0, -1]), dtype=int)
        kernel_bottomright = np.array(([-1, 1, -1], [1, 1, 0], [-1, 0, 0]), dtype=int)
        kernel_topleft = np.array(([0, 0, -1], [0, 1, 1], [-1, 1, -1]), dtype=int)
        kernel_topright = np.array(([-1, 0, -0], [1, 1, 0], [-1, 1, -1]), dtype=int)

        ppt.figure(2)

        ppt.subplot(221)
        ppt.xlabel("Top left kernel")
        ppt.imshow(kernel_topleft, cmap='gray')

        ppt.subplot(222)
        ppt.xlabel("Top right kernel")
        ppt.imshow(kernel_topright, cmap='gray')

        ppt.subplot(223)
        ppt.xlabel("bottom left kernel")
        ppt.imshow(kernel_bottomleft, cmap='gray')

        ppt.subplot(224)
        ppt.xlabel("bottom right kernel")
        ppt.imshow(kernel_bottomright, cmap='gray')

        # convoluting the kernels

        ppt.figure(3)

        topleft_image = morph(kernel, kernel_topleft)
        ppt.subplot(221)
        ppt.xlabel("Top Left Corner")
        ppt.imshow(topleft_image, cmap='gray')

        topright_image = morph(kernel, kernel_topright)
        ppt.subplot(222)
        ppt.xlabel("Top Right Corner")
        ppt.imshow(topright_image, cmap='gray')

        bottomleft = morph(kernel, kernel_bottomleft)
        ppt.subplot(223)
        ppt.xlabel("BottomLeftCorner")
        ppt.imshow(bottomleft, cmap='gray')

        bottomright = morph(kernel, kernel_bottomright)
        ppt.subplot(224)
        ppt.xlabel("BottomRightCorner")
        ppt.imshow(bottomright, cmap='gray')

        ppt.show()


if __name__ == '__main__':
    hitnmiss()

from tkinter import filedialog
#Just for reading images
import cv2
import matplotlib.pyplot as ppt
import numpy as np
from matplotlib.widgets import Button


def convoluteEdgeKernel(image, kernel):
    #flip kernel
    flip = np.flipud(np.fliplr(kernel))
    # leaving 2 pixels margin boundary
    l = int(kernel.shape[0] // 2)
    b = int(kernel.shape[1] // 2)

    image1 = np.zeros_like(image)
    for x in range(b, image.shape[1] - b):
        for y in range(l, image.shape[0] - l):
            g_val = np.sum(np.multiply(flip, image[y - l:y + l + 1, x - b:x + b + 1]))

            #only edge pixels
            g_val = 0 if g_val < 175 else 255
            image1[y + l, x + b] = g_val
    return image1


def gkern(l, sig=1.):

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) // (2. * sig**2))

    return kernel / np.sum(kernel)


class Edge:
    path = 'home_one.jpg'

    def __init__(self):
        self.initUI()

    def initUI(self):
        image = cv2.imread(Edge.path, cv2.IMREAD_GRAYSCALE)
        ppt.figure(1)

        # gaussian blur to remove noise
        g_kernel = gkern(3, 1)
        gauss_blur = convoluteEdgeKernel(image, kernel=g_kernel)

        #prewitt
        prewittx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitty = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        image_prewittx = convoluteEdgeKernel(gauss_blur, kernel=prewittx)
        image_prewitty = convoluteEdgeKernel(gauss_blur, kernel=prewitty)

        # sobel
        kernelx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        image_sobelx = convoluteEdgeKernel(gauss_blur, kernelx)
        image_sobely = convoluteEdgeKernel(gauss_blur, kernely)

        #calculating gradients
        ppt.subplot(335)
        ppt.xlabel("Original Image")
        ppt.imshow(image, cmap='gray')
        #ppt.imsave("wiener_filter.png", result)

        ppt.subplot(331)
        ppt.xlabel("prewitt x")
        ppt.imshow(image_prewittx, cmap='gray')
        ppt.imsave("prewittx.png", image_prewittx)


        ppt.subplot(332)
        ppt.xlabel("prewitt y")
        ppt.imshow(image_prewitty, cmap='gray')
        ppt.imsave("prewitty.png", image_prewitty)

        ppt.subplot(333)
        ppt.xlabel("prewitt xy")
        ppt.imshow(image_prewittx+image_prewitty, cmap='gray')
        ppt.imsave("prewittxy.png", image_prewittx+image_prewitty)

        ppt.subplot(337)
        ppt.xlabel("sobel x")
        ppt.imshow(image_sobelx, cmap='gray')
        ppt.imsave("sobelx.png", image_sobelx)

        ppt.subplot(338)
        ppt.xlabel("sobel y")
        ppt.imshow(image_sobely, cmap='gray')
        ppt.imsave("sobely.png", image_sobely)

        ppt.subplot(339)
        ppt.xlabel("sobel combined")
        ppt.imshow(image_sobelx + image_sobely, cmap='gray')
        ppt.imsave("sobel_combined.png", image_sobelx+image_sobely)

        axprev = ppt.axes([0.45, 0.005, 0.15, 0.075])
        img_bttn = Button(axprev, 'Open Image')
        img_bttn.on_clicked(lambda x: self.button_click())
        ppt.show()

    def button_click(self):
        Edge.path = filedialog.askopenfilename(filetypes=[("Image File", '.png')])
        self.initUI()


if __name__ == '__main__':
    Edge()

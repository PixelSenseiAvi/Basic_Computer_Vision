from tkinter import *
import cv2
import matplotlib as mpl

mpl.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog


class restore2:
    path = '/home/cloud/Desktop/TheCameraman.png'

    def __init__(self, master):
        self.frame1 = Frame(master)
        self.frame2 = Frame(master)
        self.frame3 = Frame(master)
        hbtn = Button(self.frame2, text="OPEN IMAGE", command=lambda: self.button_click(master))
        hbtn.pack(fill="none", expand=True)
        self.initUI(master)

    def initUI(self, master):
        self.frame1.grid(row=0, column=0)
        self.frame2.grid(row=10, column=0)
        self.frame3.grid(row=0, column=9)

        if len(restore2.path) > 0:
            img = cv2.imread(restore2.path, 0)
            k = 0.002
            #threshold
            eps = 1E-1
            fourier_img = (np.fft.fft2(img))
            N = 5 + 3 * np.random.randn(len(fourier_img), len(fourier_img[0]))
            H_values = [[H(k, i, j) for j in range(len(fourier_img[0]))] for i in range(len(fourier_img))]

            G = [[hij * fij + nij for hij, fij, nij in zip(hi, fi, ni)] for hi, fi, ni in zip(H_values, fourier_img, N)]
            degraded_image = abs(np.fft.ifft2(G))
            degraded_image = np.uint8(degraded_image)
            H_reversed = [[1 / Hij if Hij > eps else 0 for Hij in Hi] for Hi in H_values]
            restored_image = [[gij * hrij for gij, hrij in zip(gi, hri)] for gi, hri in zip(G, H_reversed)]
            restored_image = abs(np.fft.ifft2(restored_image))
            restored_image = np.uint8(restored_image)

            fig = Figure(figsize=(8, 8))
            a = fig.add_subplot(221)
            a.set_title("Original Image")
            a.imshow(img, cmap='gray')
            b = fig.add_subplot(222)
            b.set_title("Degraded image")
            b.imshow(degraded_image, cmap='gray')
            c = fig.add_subplot(223)
            c.set_title("Restored Image")
            c.imshow(restored_image, cmap='gray')

            canvas = FigureCanvasTkAgg(fig, self.frame1)
            canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, rowspan=8)
            canvas.draw()

    def button_click(self, master):
        restore2.path = filedialog.askopenfilename(filetypes=[("Image File", '.png')])
        self.initUI(master)


def H(k, u, v, mu=0, mv=0):
    return np.exp(-k * ((u - mu) ** 2 + (v - mv) ** 2) ** (5 / 6))


def main():
    root = Tk()
    root.title("Image Restoration")
    restore2(root)
    root.mainloop()


if __name__ == '__main__':
    main()

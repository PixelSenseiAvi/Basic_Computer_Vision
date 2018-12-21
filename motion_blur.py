from tkinter import *
import cv2
import matplotlib as mpl

mpl.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog


class motion_blur:
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

        if len(motion_blur.path) > 0:
            img = cv2.imread(motion_blur.path, 0)
            size = 15

            # generating the kernel
            kernel_motion_blur = np.zeros((size, size))
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
            kernel_motion_blur = kernel_motion_blur / size

            # applying kernel to the input image
            output = cv2.filter2D(img, -1, kernel_motion_blur)

            degraded_img, restored_img = winear_filtering(img, img.shape[0], img.shape[1])

            fig = Figure(figsize=(8, 8))
            a = fig.add_subplot(221)
            a.set_title("Original Image")
            a.imshow(img, cmap='gray')
            b = fig.add_subplot(222)
            b.set_title("Degraded image- Motion blur")
            b.imshow(degraded_img, cmap='gray')
            c = fig.add_subplot(223)
            c.set_title("Restored Image")
            c.imshow(restored_img, cmap='gray')

            canvas = FigureCanvasTkAgg(fig, self.frame1)
            canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, rowspan=8)
            canvas.draw()

    def button_click(self, master):
        motion_blur.path = filedialog.askopenfilename(filetypes=[("Image File", '.png')])
        self.initUI(master)


def winear_filtering(img, width, height):
    k = 0.001
    eps = 1E-5
    pic_a = np.asarray(img)
    F = np.fft.fft2(pic_a)
    # random noise
    noise = 5 + 3 * np.random.randn(len(F), len(F[0]))
    H_values = [[H(k, i, j) for j in range(len(F[0]))] for i in range(len(F))]
    N = np.fft.fft2(noise)
    # degraded image
    G = [[hij * fij + nij for hij, fij, nij in zip(hi, fi, ni)] for hi, fi, ni in zip(H_values, F, N)]
    degraded_image = abs(np.fft.ifft2(G))

    Sn = [[abs(n) ** 2 for n in ni] for ni in np.fft.fft2(N)]
    nA = sum([sum(sni) for sni in Sn]) / (len(Sn) * len(Sn[0]))
    Sf = [[abs(n) ** 2 for n in ni] for ni in np.fft.fft2(pic_a)]
    fA = sum([sum(sfi) for sfi in Sf]) / (len(Sf) * len(Sf[0]))
    R = nA / fA
    R *= 0.001

    H_reversed = [[1 / Hij if Hij > eps else 0 for Hij in Hi] for Hi in H_values]
    H_squared = [[abs(Hij ** 2) for Hij in Hi] for Hi in H_values]

    # mean square error/ wiener filtering
    W = []
    for i in range(len(H_values)):
        W.append([])
        for j in range(len(H_values[0])):
            val = H_reversed[i][j] * H_squared[i][j] / (H_values[i][j] * H_squared[i][j] + R) * G[i][j]
            W[i].append(val)

    restored_image = abs(np.fft.ifft2(W))

    bad_pic = np.absolute(degraded_image.astype(np.uint8))

    good_pic = np.absolute(np.asarray(restored_image).astype(np.uint8))

    return bad_pic, good_pic


def H(k, u, v, mu=0, mv=0):
    return np.exp(-k * ((u - mu) ** 2 + (v - mv) ** 2) ** (5 / 6))


def main():
    root = Tk()
    root.title("Image Restoration")
    motion_blur(root)
    root.mainloop()


if __name__ == '__main__':
    main()

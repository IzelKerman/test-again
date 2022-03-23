from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import tkinter as tk
from PIL import ImageTk
import numpy as np
import csv
import time
import datetime
import random
import compute


def cut_3(n):
    if n % 3 == 0:
        x = n/3
        return [x, x, x]
    elif n % 3 == 1:
        x = (n - 1) / 3
        return [x, x + 1, x]
    else:
        x = (n - 2) / 3
        return [x + 1, x, x + 1]


def cutting(v, with_center=False):
    #x = [[a, b], [c, d]]
    len = [v[i][1] - v[i][0] for i in range(2)]
    deltas = [cut_3(len_i) for len_i in len]
    new = [[v[0][0]], [v[1][0]]]
    for i in range(3):
        new[0].append(new[0][-1] + deltas[0][i])
        new[1].append(new[1][-1] + deltas[1][i])
    X = [[[new[0][i], new[0][i+1]], [new[1][j], new[1][j+1]], True] for i in range(3) for j in range(3)]
    if not with_center:
        X[4][2] = False
    return X


def find_divisors(n, d, k):
    """
    returns the greatest k such that d^k divides n
    :param n: the number that we want to divide
    :param d: the divisor
    :param k: the greatest number such that d^k | n
    :return: k
    """
    if n % d == 0:
        return find_divisors(n / d, d, k + 1)
    else:
        return k


#window = tk.Tk()
#window.title("Hello there")

Im = Image.open("Images/0.png")
pixels = Im.load()
pixels_copy = Image.open("Images/1.png").load()

#img = tk.PhotoImage(file='Images/24.png')
#img = ImageTk.PhotoImage(Im)
#label = tk.Label(window, image=img)
#label.pack()

X = [[[0, Im.size[0]-1], [0, Im.size[1]-1], True]]

"""def update_image():
    global X
    im = ImageTk.PhotoImage(Im)
    label.configure(image=im)
    label.image = im
    label.after(20, update_image)
    #ImageDraw.Draw(Im).rectangle(X[0], fill=pixels_copy[int((X[0][0][0] + X[0][0][1])/2), int((X[0][1][0] + X[0][1][1])/2)])
    for i in range(12):
        if X[0][2]:
            #ImageDraw.Draw(Im).rectangle([(X[0][0][0], X[0][1][0]), (X[0][0][1], X[0][1][1])], fill=(int(255 * random.random()), int(255 * random.random()), int(255 * random.random())))
            ImageDraw.Draw(Im).rectangle([(X[0][0][0], X[0][1][0]), (X[0][0][1]-1, X[0][1][1]-1)], fill=pixels_copy[int((X[0][0][0] + X[0][0][1])/2), int((X[0][1][0] + X[0][1][1])/2)])
        X_add = cutting(X[0])
        del X[0]
        X = X + X_add"""

"""
def update_image():
    global X
    im = ImageTk.PhotoImage(Im)
    label.configure(image=im)
    label.image = im
    print(time.process_time())
    label.after(1000, update_image)
    print(time.process_time())
    print("--------------------")
    ImageDraw.Draw(Im).rectangle([(500, 500), (1000, 1000)], fill=(int(100*(1+np.sin(time.process_time()))), int(105*(1+np.sin(2 * time.process_time()))), int(100*(1+np.sin(1.5 * time.process_time())))))
"""


class ImagePerso:
    def __init__(self, window, image_path, system=None):
        self.window = window
        self.im = Image.open(image_path)
        #self.imtk = ImageTk.PhotoImage(self.im)
        self.pixels_copy = Image.open(image_path).load()
        #self.label = tk.Label(self.window, image=self.imtk)
        #self.label.pack()
        self.size = self.im.size
        self.i = [int(self.size[i]) for i in range(2)]
        self.Mod = [5 * np.copy(self.size)[i] for i in range(2)]
        self.dx = np.copy(self.size)[0]
        self.dy = np.copy(self.size)[1]
        self.kx = 1
        self.ky = 1
        self.qx = 1
        self.qy = 1
        self.qmin = min(self.qx, self.qy)
        self.i = 0
        self.j = 0
        self.k = 0
        self.system = system

    def next_rectangle(self):
        if self.Mod[0] == 1 and self.Mod[1] == 1:
            return 0
        elif self.i == self.qx - 1 and self.j == self.qy - 1:
            if self.Mod[0] % 5 == 0:
                self.Mod[0] = self.Mod[0] / 5
            elif self.Mod[0] % 3 == 0:
                self.Mod[0] = self.Mod[0] / 3
            elif self.Mod[0] % 2 == 0:
                self.Mod[0] = self.Mod[0] / 2

            if self.Mod[1] % 5 == 0:
                self.Mod[1] = self.Mod[1] / 5
            elif self.Mod[1] % 3 == 0:
                self.Mod[1] = self.Mod[1] / 3
            elif self.Mod[1] % 2 == 0:
                self.Mod[1] = self.Mod[1] / 2

            self.next_qs()
        else:
            self.next_position()

        #if not ((self.i + np.ceil(self.dx / 2) - 1) % self.Mod[0] == 0 and (self.j + np.ceil(self.dy / 2) - 1) % self.Mod[1] == 0):
        #if not (np.ceil((self.i + 1 / 2) * self.dx) % np.ceil(self.Mod[0] / 2) == 0 and np.ceil((self.j + 1 / 2) * self.dy) % np.ceil(self.Mod[1] / 2) == 0):
        #if not ((self.i * self.dx) % self.Mod[0] == np.ceil(self.dx / 2) - 1 and (self.j * self.dy) % self.Mod[1] == np.ceil(self.dy / 2) - 1):
        """if not (self.i % self.kx == np.ceil(self.kx / 2) - 1 and self.j % self.ky == np.ceil(self.ky / 2) - 1) or self.Mod[0] == self.size[0]:
            #print(np.ceil((self.i + 1 / 2) * self.dx), np.ceil(self.Mod[0] / 2))
            #print("well, there should be something", self.i, self.j)
            rectangle_position = [(self.i * self.dx, self.j * self.dy), ((self.i + 1) * self.dx - 1, (self.j + 1) * self.dy - 1)]
            #color = self.pixels_copy[(self.i + np.ceil(self.dx / 2) - 1) * self.qx, (self.j + np.ceil(self.dy / 2) - 1) * self.qy]
            color = self.pixels_copy[np.ceil((self.i + 1 / 2) * self.dx) - 1, np.ceil((self.j + 1 / 2) * self.dy) - 1]
            #color = (int(255 * random.random()), int(255 * random.random()), int(255 * random.random()))
            #ImageDraw.Draw(self.im).rectangle(rectangle_position, color)
            return rectangle_position, color"""
        if not (self.i % self.kx == 0 and self.j % self.ky == 0) or self.Mod[0] == self.size[0] or self.kx == 1 == self.ky:
            rectangle_position = [(self.i * self.dx, self.j * self.dy), ((self.i + 1) * self.dx - 1, (self.j + 1) * self.dy - 1)]
            if self.system is None:
                color = self.pixels_copy[self.i * self.dx, self.j * self.dy]
            else:
                color = self.system.compute_angle(self.i * self.dx, self.j * self.dy)
            return rectangle_position, color
        else:
            #print("Nothing happened", self.i, self.j)
            return self.next_rectangle()

    def next_qs(self):
        print("next", self.Mod)
        self.kx = 1
        if self.Mod[0] % 5 == 0:
            self.dx = self.dx / 5
            self.qx = 5 * self.qx
            self.kx = 5
        elif self.Mod[0] % 3 == 0:
            self.dx = self.dx / 3
            self.qx = 3 * self.qx
            self.kx = 3
        elif self.Mod[0] % 2 == 0:
            self.dx = self.dx / 2
            self.qx = 2 * self.qx
            self.kx = 2
        self.i = 0

        self.ky = 1
        if self.Mod[1] % 5 == 0:
            self.dy = self.dy / 5
            self.qy = 5 * self.qy
            self.ky = 5
        elif self.Mod[1] % 3 == 0:
            self.dy = self.dy / 3
            self.qy = 3 * self.qy
            self.ky = 3
        elif self.Mod[1] % 2 == 0:
            self.dy = self.dy / 2
            self.qy = 2 * self.qy
            self.ky = 2
        self.j = 0

        self.k = 0

        self.qmin = min(self.qx, self.qy)

        print(self.qx, self.qy)

    def next_position(self):
        if self.k < self.qmin - 1:
            if self.j == 0:
                self.k += 1
                self.i = 0
                self.j = self.k
            else:
                self.j -= 1
                self.i += 1
        #elif self.k <= self.qmin - 1 and self.qy <= self.qx:
        elif self.k < self.qx + self.qy - 1 - self.qmin and self.qy <= self.qx:
            if self.j == 0:
                self.k += 1
                #self.i = self.k - self.qmin + 1
                self.j = self.qy - 1
                self.i = self.k - self.j
            else:
                self.j -= 1
                self.i += 1
        #elif self.k == self.qmin - 1:
        elif self.k < self.qx + self.qy - 1 - self.qmin:
            if self.i == self.qx - 1:
                self.k += 1
                self.i = 0
                #self.j = self.k - self.qy + 1
                self.j = self.k
            else:
                self.j -= 1
                self.i += 1
        elif self.qx + self.qy - 1 - self.qmin <= self.k:
            if self.i == self.qx - 1:
                self.k += 1
                self.j = self.qy - 1
                self.i = self.k - self.j
            else:
                self.j -= 1
                self.i += 1

    def go_to_step(self, n):
        for i in range(n):
            if self.Mod[0] % 5 == 0:
                self.Mod[0] = self.Mod[0] / 5
            elif self.Mod[0] % 3 == 0:
                self.Mod[0] = self.Mod[0] / 3
            elif self.Mod[0] % 2 == 0:
                self.Mod[0] = self.Mod[0] / 2

            if self.Mod[1] % 5 == 0:
                self.Mod[1] = self.Mod[1] / 5
            elif self.Mod[1] % 3 == 0:
                self.Mod[1] = self.Mod[1] / 3
            elif self.Mod[1] % 2 == 0:
                self.Mod[1] = self.Mod[1] / 2

            self.next_qs()


"""
def update_image():
    global X
    im = ImageTk.PhotoImage(Im)
    label.configure(image=im)
    label.image = im
    label.after(20, update_image)
    #ImageDraw.Draw(Im).rectangle(X[0], fill=pixels_copy[int((X[0][0][0] + X[0][0][1])/2), int((X[0][1][0] + X[0][1][1])/2)])
    for i in range(12):
        if X[0][2]:
            #ImageDraw.Draw(Im).rectangle([(X[0][0][0], X[0][1][0]), (X[0][0][1], X[0][1][1])], fill=(int(255 * random.random()), int(255 * random.random()), int(255 * random.random())))
            ImageDraw.Draw(Im).rectangle([(X[0][0][0], X[0][1][0]), (X[0][0][1]-1, X[0][1][1]-1)], fill=pixels_copy[int((X[0][0][0] + X[0][0][1])/2), int((X[0][1][0] + X[0][1][1])/2)])
        X_add = cutting(X[0])
        del X[0]
        X = X + X_add"""

"""
window = tk.Tk()
window.title("Hello there")
im = Image.open("Images/0.png")
imtk = ImageTk.PhotoImage(im)
label = tk.Label(window, image=imtk)
label.pack()
imp = ImagePerso(window, "Images/0.png")
"""

"""def update_image_2():
    imp.label.configure(image=imp.imtk)
    imp.label.image = imp.imtk
    imp.label.after(1000, update_image_2)
    for i in range(1):
        rectangle_position, color = imp.next_rectangle()
        ImageDraw.Draw(imp.im).rectangle(rectangle_position, color)"""
"""
def update_image_2():
    global imtk
    imtk = ImageTk.PhotoImage(im)
    label.configure(image=imtk)
    label.image = imtk
    label.after(10, update_image_2)
    for i in range(100):
        rectangle_position, color = imp.next_rectangle()
        ImageDraw.Draw(im).rectangle(rectangle_position, color)"""

syst = compute.System(0.99999, [30, np.pi / 4, 0], [np.pi / 3, np.pi / 3], [1080, 1080], "Images/Hello_there.png", "Images/background.png")
syst.create_image("Images/Hello_there.png")

window = tk.Tk()
window.title("Hello there")
#im = Image.open("Images/Hello_there.png")
im = Image.open("Out/2022-03-23--17-39-00.png")
imtk = ImageTk.PhotoImage(im)
label = tk.Label(window, image=imtk)
label.pack()
#imp = ImagePerso(window, "Images/Hello_there.png", syst)
imp = ImagePerso(window, "Out/2022-03-23--17-39-00.png", syst)
imp.go_to_step(7)


def update_image_3():
    global imtk
    imtk = ImageTk.PhotoImage(im)
    label.configure(image=imtk)
    label.image = imtk
    if imp.i == imp.size[0] - 1 and imp.j == imp.size[1] - 1:
        im.save("Out/" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + ".png")
        return 0
    elif imp.i == imp.qx - 1 and imp.j == imp.qy - 1 and not imp.qx == 1:
        im.save("Out/" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")+".png")
    for i in range(1):
        rectangle_position, color = imp.next_rectangle()
        ImageDraw.Draw(im).rectangle(rectangle_position, color)
    label.after(1, update_image_3)


window.after(100, update_image_3)
window.mainloop()

"""
pixels = im.load()
for i in range(im.size[0]):
    for j in range(im.size[1]):
        pixels[i, j] = syst.compute_angle(i, j)
        if j == 0:
            print(i / im.size[0])
im.show()"""

"""
def update_image():
    global tkimg1
    tkimg1 = ImageTk.PhotoImage(Image.open('Images/0.png'))
    label.config(image=tkimg1)
    label.after(1000, update_image)
w = tk.Tk()
im = Image.open('Images/24.png')
tkimg1 = ImageTk.PhotoImage(im)
label = tk.Label(w, image=tkimg1)
label.pack()
w.after(1000, update_image)
w.mainloop()"""
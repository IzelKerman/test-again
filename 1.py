from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
import tkinter as tk
from PIL import ImageTk
import numpy as np
import csv
import time
import random


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


window = tk.Tk()
window.title("Hello there")

Im = Image.open("Images/1.png")
pixels = Im.load()
pixels_copy = Image.open("Images/1.png").load()

#img = tk.PhotoImage(file='Images/24.png')
img = ImageTk.PhotoImage(Im)
label = tk.Label(window, image=img)
label.pack()

X = [[[0, Im.size[0]-1], [0, Im.size[1]-1], True]]

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
        X = X + X_add


window.after(1000, update_image)
window.mainloop()

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
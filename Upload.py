import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

ws = tk.Tk()
ws.geometry("600x600")
ws.title("Image Detection")
f1 = ("Sitka Small", 22, "bold")
l1 = tk.Label(ws, text = "Upload Image & Display", bg = "yellow", width = 30, font = f1)
l1.grid(row = 1, column = 2, columnspan = 50)
b1 = tk.Button(ws, text = "Select Image", width = 10, bg = "pink", font = ("Sitka Small", 10, "bold"), command = lambda : upload_file())
b1.grid(row = 2, column = 1, columnspan = 50)


def upload_file():
    f_types = [("JPG Files", "*.jpg"), ("PNG Files", "*.png"), ("JPEG Files", "*.jpeg"), ("PDF Files", "*.pdf")]
    filename = tk.filedialog.askopenfilename(multiple = True, filetypes = f_types)
    col = 25
    row = 6
    for f in filename:
        img = Image.open(f)
        img = img.resize((500, 500))
        # img = img.resize((180, 180))
        # img = img.resize((120, 120))
        # img = img.resize((100, 100))
        img = ImageTk.PhotoImage(img)
        e1 = tk.Label(ws)
        e1.grid(row = row, column = col)
        e1.image = img
        e1["image"] = img
        # if(col == 1):
        #     row = row + 1
        # else:
        #     col = col + 1
ws.mainloop()


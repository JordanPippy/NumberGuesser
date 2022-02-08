from tkinter import *
from tkinter import ttk
from PIL import Image, ImageChops
import model

lasx = None
laxy = None
canvas = None

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_smth(event):
    global lasx, lasy
    canvas.create_oval((lasx, lasy, event.x, event.y), 
                      fill='black', 
                      width=20)
    lasx, lasy = event.x, event.y

def save():
    global canvas
    ps = canvas.postscript(file="image1.ps", colormode='color')    
    psimage=Image.open("image1.ps")
    psimage = psimage.resize((28,28))
    #psimage = psimage.convert('1')
    psimage = ImageChops.invert(psimage)
    
    psimage = psimage.convert('L')
    psimage = process_img(psimage)
    psimage.save("image1.png")
    model.make_predictions()

def clear():
    global canvas
    canvas.delete('all')

def process_img(psimage):
    pixels = psimage.load()
    for i in range(28):
        for j in range(28):
            if (pixels[i,j] != 0):
                pixels[i,j] = 255
    return psimage


app = Tk()
app.geometry("400x400")


canvas = Canvas(app, bg='white')
canvas.pack(anchor='nw', fill='both', expand=1)

canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)

saveButton = ttk.Button(app, text='Save', command=save)
clearButton = ttk.Button(app, text="Clear", command=clear)

saveButton.pack(ipadx=5, ipady=5, expand=True)
clearButton.pack(ipadx=5, ipady=5, expand=True)




app.mainloop()

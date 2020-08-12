from tkinter import *


from PIL import Image, ImageTk

def testyours():
    import ecg2

root = Tk()
root.title("ECG Classification")
canvas = Canvas(root, width=600, height=600)
canvas.pack()

heading = Label(root, text="ECG Classification", font=('Algerian',25))
heading.place(relx=0.1, relheight=0.1, relwidth=0.8)

load = Image.open("images/heart.jpg")
render = ImageTk.PhotoImage(load)
img = Label(root, image=render)
img.place(rely=0.1, relheight=0.78)

test = Button(root, text="Test Yours", font=('Arial Black',14), bg='#6B36EE', command=lambda: testyours())
test.place(relx=0.3, rely=0.89, relheight=0.1, relwidth=0.4)

root.mainloop()
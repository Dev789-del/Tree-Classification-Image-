from tkinter import *  
import sys
sys.path.append('C:\\Users\\Nguyen Quang Huy\\OneDrive\\Máy tính\\Tree_Classification_with_Image\\model')
import model_AI

#Define app main
main_app = Tk()
main_app.title("Tree Classification with Image")
main_app.geometry("600x600")

#Define image box
load_image = Button(main_app, text="Load Image", command=model_AI.load_data)
load_image.pack(pady = 20)

plot_image = Button(main_app, text="Plot Image", command=model_AI.plot_data)    
plot_image.pack()

#Run code 
main_app.mainloop()

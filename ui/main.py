from tkinter import *  
import sys
sys.path.append('C:\\Users\\Nguyen Quang Huy\\OneDrive\\Máy tính\\Tree_Classification_with_Image\\model')
import model_AI
from model_AI import *

#Define app main
main_app = Tk()
main_app.title("Tree Classification with Image")
main_app.geometry("600x600")

#Define image box
load_image = Button(main_app, text = "Load Image", command = Tree_Model.load_image)
load_image.pack(pady = 20)

#Define data box
load_data = Button(main_app, text = "Load Data", command = Tree_Model.load_data)
load_data.pack(pady = 20)

#Define plot image box
plot_image = Button(main_app, text = "Plot Image", command = Tree_Model.plot_image)    
plot_image.pack()

#Define output listbox
output_listbox = Listbox(main_app, width = 50)
output_listbox.pack(pady = 20)

#Run code 
main_app.mainloop()

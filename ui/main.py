from Tree-Classification-with-Image import model


class Main(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title("Main")
        self.pack(fill=tk.BOTH, expand=1)

        self.button = tk.Button(self, text="Go to Page 1", command= load_data)
        self.button.pack()

        self.button = tk.Button(self, text="Go to Page 2", command= show_data)
        self.button.pack()

    def onButton1Click(self):
        self.parent.switch_frame(Page1)
    
    def onButton2Click(self):
        self.parent.switch_frame(Page2)

Main.mainloop()


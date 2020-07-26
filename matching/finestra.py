 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:09:03 2019

@author: mazzone
"""

import tkinter as tk
import findUser as fu
import query as q
import joblib



class Example(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        # create a prompt, an input box, an output label,
        # and a button to do the computation
        self.prompt = tk.Label(self, text="Inserisci la tua domanda:", anchor="w")
        self.entry = tk.Entry(self)
        self.submit = tk.Button(self, text="InviaRichiesta", command = self.calculateQuery)
        command2= self.findUserRequest
        self.output = tk.Label(self, text="")

        # lay the widgets out on the screen. 
        self.prompt.pack(side="top", fill="x")
        self.entry.pack(side="top", fill="x", padx=20)
        self.output.pack(side="top", fill="x", expand=True)
        self.submit.pack(side="right")

    def calculateQuery(self):
        # get the value from the input widget, convert
        # it to an int, and do a calculation
        try:
            #i="devo aggiustare i miei vestiti... c'è una sarta che può aiutarmi??"
            i = str(self.entry.get())
        except ValueError:
            i = "Please enter string only"
          
        i="devo aggiustare i miei vestiti... c'è una sarta che può aiutarmi??"
        query, abstr= q.queryRappresentation(i) 
        #query = joblib.load('query') 
        print(query)
        intent = str(query[1])
        print(intent)
        name, descrizione, Ntel= fu.FindUser(query)
        self.output.configure(text=intent)
        self.output.configure(text=abstr)
        name, descrizione, Ntel= fu.FindUser(query)
        self.output.configure(text=name)
        self.output.configure(text=descrizione)
        self.output.configure(text=Ntel)
            
    def findUserRequest(self):
       
        name, descrizione, Ntel= fu.FindUser(query)
        # set the output widget to have our result
        self.output.configure(text=name)
        self.output.configure(text=descrizione)
        self.output.configure(text=Ntel)

# if this is run as a program (versus being imported),
# create a root window and an instance of our example,
# then start the event loop

if __name__ == "__main__":
    root = tk.Tk()
    Example(root).pack(fill="both", expand=True)
    root.mainloop()
from tkinter import *
import os


class MyWindow:
    def __init__(self, win):
        
        self.lbl1=Label(win, text='First folder')
        #self.lbl2=Label(win, text='Second number')
        #self.lbl3=Label(win, text='Result')
        self.lbl4=Label(win,text='')
        self.t1=Entry(bd=3)
        #self.t2=Entry()
        #self.t3=Entry()
        #self.btn1 = Button(win, text='Add')
        #self.btn2=Button(win, text='Subtract')
        self.lbl1.place(x=100, y=50)
        self.t1.place(x=200, y=50)
        #self.lbl2.place(x=100, y=100)
        #self.t2.place(x=200, y=100)
        self.b1=Button(win, text='create folder', command=self.add)
        #self.b1=Button(win, text='Add', command=self.add)
        self.b2=Button(win, text='next', command =self.sub(win))
        #self.b2.bind('<Button-1>', self.sub)
        self.b1.place(x=100, y=150)
        self.b2.place(x=200, y=150)
        #self.lbl3.place(x=100, y=200)
        self.lbl4.place(x=200, y=250)
        #self.t3.place(x=200, y=200)
    def add(self):
        '''self.t3.delete(0, 'end')
        num1=int(self.t1.get())
        num2=int(self.t2.get())
        result=num1+num2
        self.t3.insert(END, str(result))'''
        try:
            directory = './'+str(self.t1.get())+'/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.lbl4.config(text='created successully!')
            #self.t3.insert(END, 'created successfully!')
        except OSError:
            print('Error: Creating directory. ' +  directory)

    '''def sub(self,win):
        self.t3.delete(0, 'end')
        num1=int(self.t1.get())
        num2=int(self.t2.get())
        result=num1-num2
        self.t3.insert(END, str(result))
        f = Frame(win)
        lb= Label(f,text='second page').pack()
        f.tkraise()'''


window=Tk()
mywin=MyWindow(window)
window.title('Hello Python')
window.geometry("400x300+10+10")
window.mainloop()
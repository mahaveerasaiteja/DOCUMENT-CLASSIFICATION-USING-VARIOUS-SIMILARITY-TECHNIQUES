''' FAILED SOME FILES'''

import tkinter as tk
from tkinter.filedialog import askopenfilename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np 
import pandas as pd
import os
from shutil import move
from preprotest import preproces

root = tk.Tk()
root.geometry("1000x700+50+0")
root.title("Text Classification")
xref, yref = 100,100
file_loc = ["action", "comedy", "romance"]
record = {"action":0, "comedy":1, "romance":2}
rev_record ={i:j for j,i in record.items()}
vecotorizer = TfidfVectorizer()
class classify:
    def __init__(self, root):
        self.frame = tk.Frame(root)
        self.frame.place(x = 0, y = 0, width = 1000, height = 700)
        tk.Label(self.frame, text = "Advance Document Classification", font = ("Arial", 30, "bold"), fg = "green", width = 30).place(x = xref+70, y = yref-50)
        tk.Label(self.frame, text = "Select the file", font = ("Arial", 20, "bold"), width = 15).place(x = xref, y = yref+100)
        self.fl = tk.Entry(self.frame, font = ("Arial", 20))
        self.fl.place(x = xref+250, y = yref+100)
        tk.Button(self.frame, text = "Select", font = ("Arial", 16),bg= "lightgreen", command = self.file_select).place(x = xref+550, y = yref+100)
        tk.Button(self.frame, text = "Cluster", font = ("Arial", 30, "bold"),bg= "lightgreen", command = self.cl).place(x = xref+550, y = yref+400)

    
    def file_select(self):
        self.path = askopenfilename()
        self.fl.delete(0, "end")
        self.fl.insert(0, self.path.split("/")[-1])
        
    def cl(self):
        # for k in self.path:
        user = open(self.path, "r").read()
        user = preproces(user)
       # print(user) test data needs to be pre processed

        doc = []
        raw_data = ""
        for i in file_loc:
            for j in os.listdir(i):
                temp = open(i+"/"+j, "r").read()
                temp = preproces(temp)
                doc.append([temp, i])
                raw_data += temp

        #raw_data = preproces(raw_data)

        vecotorizer.fit([raw_data])
        data = np.array(doc)
        data = pd.DataFrame({"feature": data[:,0], "label": data[:,1]})
        data["label"].replace(record.keys(), record.values(), inplace = True)

        train = vecotorizer.transform(data["feature"]).toarray()
        query = vecotorizer.transform([user]).toarray()
        clf_euclidean = NearestNeighbors(n_neighbors=2)
        clf_euclidean.fit(train, data["label"])
        ep = clf_euclidean.kneighbors(query)
        clf_cosine = NearestNeighbors(n_neighbors=2,metric= "cosine")
        clf_cosine.fit(train, data["label"])
        ep2 = clf_cosine.kneighbors(query)
        clf_jaccard = NearestNeighbors(n_neighbors=2, metric= "jaccard")
        clf_jaccard.fit(train, data["label"])
        ep3 = clf_jaccard.kneighbors(query)
        ans = [data["label"][ep[1][0][0]], data["label"][ep2[1][0][0]], data["label"][ep3[1][0][0]]]
        sim = [ep[0][0][0], ep2[0][0][0], ep3[0][0][0]]
        out = self.answer(ans)
        move(self.path, rev_record[out]+"/"+self.path.split("/")[-1])
        tk.Label(self.frame, text = f" Clustered category: {rev_record[out]}\n\n\nEuclidean: {sim[0]}\nCosine: {sim[1]}\nJaccard: {sim[2]}", font = ("arial", 20), fg = "green").place(x =xref+100, y = yref+300)


    def answer(self, ans):          #needs explanation(which is the best similarity selected to place the document)
        count_dic = ans[0]
        cnt = ans.count(ans[0])
        for i in ans:
            if cnt<ans.count(i):
                count_dic = i
                cnt = ans.count(i)
        return count_dic

gui = classify(root)
root.mainloop()
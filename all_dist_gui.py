import tkinter as tk
from tkinter.filedialog import askopenfilename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np 
import pandas as pd
import os
from shutil import move
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import nltk
import pyemd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize 
import os
import re
from shutil import move
import tkinter as tk
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.similarities import WmdSimilarity

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
       # print(user) test data needs to be pre processed
        doc = []
        #raw_data = ""
        raw_data = []
        for i in file_loc:
            for j in os.listdir(i):
                temp = open(i+"/"+j, "r").read()
                doc.append([temp, i])
                #raw_data += temp
                raw_data.append(temp)

        #print(raw_data) raw data needs to be preprocessed

        vecotorizer.fit(raw_data)
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

        #=============LSA=================
        svd = TruncatedSVD(100)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        X_train_lsa = lsa.fit_transform(train)
        X_test_lsa = lsa.transform(query)
        clf_lsa = NearestNeighbors(n_neighbors=2,metric="cosine")
        clf_lsa.fit(X_train_lsa,data["label"])
        ep4 = clf_lsa.kneighbors(X_test_lsa)
        print("lsa")
        print("and answer is",data["label"][ep4[1][0][0]],"which is: ",file_loc[data["label"][ep4[1][0][0]]])

        #==========LDA==================
        lda = LatentDirichletAllocation()

        vec = CountVectorizer(analyzer='word',                        # minimum reqd occurences of a word 
                            stop_words='english',             # remove stop words
                            lowercase=True,                   # convert all words to lowercase
                            token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                            # max_features=50000,             # max number of uniq words
                            )

        train2 = vec.fit_transform(raw_data)
        query2 = vec.transform([user])
        lda.fit(train2,data["label"])
        train_lda = lda.transform(train2)
        qu_lda = lda.transform(query2)
        clf_lda = NearestNeighbors(n_neighbors=2,metric="cosine")
        clf_lda.fit(train_lda,data["label"])
        ep5 = clf_lda.kneighbors(qu_lda)
        print("lda")
        print("and answer is",data["label"][ep5[1][0][0]],"which is: ",file_loc[data["label"][ep5[1][0][0]]])

        #=============Doc2vec=============
        def tokenize_text(text):
            tokens = []
            for sent in nltk.sent_tokenize(text):
                for word in nltk.word_tokenize(sent):
                    if len(word) < 2:
                        continue
                    tokens.append(word.lower())
            return tokens
        train_documents = []
        for i in range(len(raw_data)):
            train_documents.append(TaggedDocument(words=tokenize_text(raw_data[i]),tags=[data["label"][i]]))

        model_dbow = Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, alpha=0.025, min_alpha=0.001)
        model_dbow.build_vocab(train_documents)
        model_dbow.train(train_documents,total_examples=len(train_documents), epochs=30)
        def vector_for_learning(model, input_docs):
            sents = input_docs
            targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
            return targets, feature_vectors
        model_dbow.save('./actual.d2v')

        y_train_emd, X_train_emb = vector_for_learning(model_dbow, train_documents)
        test_doc = word_tokenize(user.lower())
        ep6 = model_dbow.docvecs.most_similar(positive=[model_dbow.infer_vector(test_doc)],topn=1)
        print("doc2vec")
        print("and answer is",data["label"][ep6[0][0]],"which is: ",file_loc[data["label"][ep6[0][0]]])

        #============word movers============
        train_tokens = []
        for i in range(len(raw_data)):
            train_tokens.append(tokenize_text(raw_data[i]))
        
        instance = WmdSimilarity(train_tokens, model_dbow,num_best=1)
        ep7 = instance[tokenize_text(user)]
        print("wordmovers:")
        print("and answer is",data["label"][ep7[0][0]],"which is: ",file_loc[data["label"][ep7[0][0]]])
        
        #=============soft-cosine=============
        from gensim import corpora

        dictionary = corpora.Dictionary(train_tokens)
        
        import gensim.downloader as api
        from gensim.models import WordEmbeddingSimilarityIndex
        from gensim.similarities import SparseTermSimilarityMatrix,SoftCosineSimilarity

        print("start")
        w2v_model = api.load("glove-wiki-gigaword-50")
        print("done1")
        similarity_index = WordEmbeddingSimilarityIndex(w2v_model)
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)
        #print(similarity_matrix)
        bow_corpus = [dictionary.doc2bow(document) for document in train_tokens]
        te_corpus = [dictionary.doc2bow(tokenize_text(user))]

        docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=1)
        ep8 = docsim_index[dictionary.doc2bow(tokenize_text(user))]
        print("soft cosine:")
        print("and answer is",data["label"][ep8[0][0]],"which is: ",file_loc[data["label"][ep8[0][0]]])


        ans = [data["label"][ep[1][0][0]], data["label"][ep2[1][0][0]], data["label"][ep3[1][0][0]],data["label"][ep4[1][0][0]],data["label"][ep5[1][0][0]],data["label"][ep6[0][0]],data["label"][ep7[0][0]],data["label"][ep8[0][0]]]
        print(ans)
        sim = [ep[0][0][0], ep2[0][0][0], ep3[0][0][0],ep4[0][0][0],ep5[0][0][0],ep6[0][1],ep7[0][1],ep8[0][1]]
        #print(sim)
        out = self.answer(ans)
        move(self.path, rev_record[out]+"/"+self.path.split("/")[-1])
        #print(" Clustered category:",{rev_record[out]},"\n\n\nEuclidean: ",{sim[0]},"\nCosine:",{sim[1]},"\nJaccard:",{sim[2]},"\nLSA:",{sim[3]},"\nLDA:",{sim[4]},"\nDoc2vec:",{sim[5]},"\nwordmovers:",{sim[6]},"\nsoft cosin:",{sim[7]})
        tk.Label(self.frame, text = f" Clustered category: {rev_record[out]}\n\nEuclidean: {sim[0]}\nCosine: {sim[1]}\nJaccard: {sim[2]}\nLSA: {sim[3]}\nLDA:{sim[4]}\nDoc2vec:{sim[5]}\nwordmovers:{sim[6]}\nsoft cosin:{sim[7]}", font = ("arial", 20), fg = "green").place(x =xref+100, y = yref+200)


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
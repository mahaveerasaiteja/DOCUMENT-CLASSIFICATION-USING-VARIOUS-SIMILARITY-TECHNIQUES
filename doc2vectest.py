#!/usr/bin/env python
# coding: utf-8

# In[61]:


from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import utils
import csv
from tqdm import tqdm
import multiprocessing
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
import pandas as pd
import numpy as np


# In[62]:


tqdm.pandas(desc="progress-bar")
# Function for tokenizing
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens
# Initializing the variables
train_documents = []
test_documents = []

categories = ['comp.graphics', 'sci.space','rec.sport.baseball','sci.electronics']
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)
for i in range(len(newsgroups_train.data)):
    train_documents.append(TaggedDocument(words=tokenize_text(newsgroups_train.data[i]),tags=[newsgroups_train.target[i]]))
for j in range(len(newsgroups_test.data)):
    test_documents.append(TaggedDocument(words=tokenize_text(newsgroups_test.data[j]),tags=[newsgroups_test.target[j]]))

#print(train_documents[0])


# In[63]:


model_dbow = Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, alpha=0.025, min_alpha=0.001)
model_dbow.build_vocab([x for x in tqdm(train_documents)])
train_documents  = utils.shuffle(train_documents)
model_dbow.train(train_documents,total_examples=len(train_documents), epochs=30)
def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors
model_dbow.save('./movieModel.d2v')


# In[64]:


y_train, X_train = vector_for_learning(model_dbow, train_documents)
y_test, X_test = vector_for_learning(model_dbow, test_documents)

#print(X_train[0])


# In[66]:
'''svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

X_train_lsa = lsa.fit_transform(X_train)
X_test_lsa = lsa.transform(X_test)'''




knn_cos = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_cos.fit(X_train, y_train)

print(knn_cos.score(X_test,y_test))

# Classify the test vectors.
y_pred = knn_cos.predict(X_test)
print(classification_report(y_test, y_pred, target_names=categories))
cm = pd.DataFrame(confusion_matrix(y_test, y_pred),index=['comp.graphics:true', 'sci.space:true','rec.sport.baseball:true','sci.electronics:true'],columns=['comp.graphics:pred', 'sci.space:pred','rec.sport.baseball:pred','sci.electronics:pred'])
#cm = pd.DataFrame(pd.crosstab(y_test,y_pred),index=categories,columns=categories)
cw = cm.sum(axis=0)
row_df = pd.DataFrame([cw],index=["All"])
cm = pd.concat([ cm,row_df])
cm["All"] = cm.sum(axis=1)
print("confusion matrix:\n",cm)
print("accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

from nltk.tokenize import word_tokenize
test_doc = word_tokenize("That is a good device".lower())
ep6 = model_dbow.docvecs.most_similar(positive=[model_dbow.infer_vector(test_doc)],topn=1)
print(ep6)
print(ep6[0])
print(ep6[0][0],ep6[0][1])
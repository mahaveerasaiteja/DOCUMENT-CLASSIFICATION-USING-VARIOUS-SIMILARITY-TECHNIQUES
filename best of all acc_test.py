from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np 
import pandas as pd

categories = ['comp.graphics', 'sci.space','rec.sport.baseball','sci.electronics']
def answer(ans):
    count_dic = ans[0]
    cnt = ans.count(ans[0])
    for i in ans:
        if cnt<ans.count(i):
            count_dic = i
            cnt = ans.count(i)
    return count_dic
'''#keeping a single document in each class
cate = ['comp.graphics']
cate2 = ['sci.space']
cate3 = ['rec.sport.baseball']
X_train = []
newsgroups_train1 = fetch_20newsgroups(subset='train',categories=cate)
X_train.append(newsgroups_train1.data[0])
newsgroups_train2 = fetch_20newsgroups(subset='train',categories=cate2)
X_train.append(newsgroups_train2.data[0])
newsgroups_train3 = fetch_20newsgroups(subset='train',categories=cate3)
X_train.append(newsgroups_train3.data[0])
print(len(X_train))
y_train = [0,1,2]
categories = ['comp.graphics', 'sci.space','rec.sport.baseball']
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)
X_test = newsgroups_test.data
y_test = newsgroups_test.target'''
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

train2 = vectorizer.transform(X_train).toarray()
#print(type(train))  >>>  <class 'numpy.ndarray'>
query2 = vectorizer.transform(X_test).toarray()
#print(type(query))  >>>  <class 'numpy.ndarray'>

#------------cosine---------------
nbrsc = NearestNeighbors(n_neighbors=2, metric="cosine").fit(train2,y_train)
distancesc,indicesc = nbrsc.kneighbors(query2)
ansic = []
y_predc = []
for i in range(len(indicesc)):
    ansic.append(indicesc[i][0])
    #print("actual:",newsgroups_test.target[i],"predicted:",newsgroups_train.target[ansi[i]])
    y_predc.append(y_train[ansic[i]])
    #print("actual:",newsgroups_train.target_names[newsgroups_test.target[i]],"predicted:",newsgroups_train.target_names[newsgroups_train.target[ansi[i]]])

#y_predc = np.array(y_predc)
print("cosine done")
#-----------jaccard------------
nbrsj = NearestNeighbors(n_neighbors=2, metric="jaccard").fit(train2,y_train)
distancesj,indicesj = nbrsj.kneighbors(query2)
ansij = []
y_predj = []
for i in range(len(indicesj)):
    ansij.append(indicesj[i][0])
    #print("actual:",newsgroups_test.target[i],"predicted:",newsgroups_train.target[ansi[i]])
    y_predj.append(y_train[ansij[i]])
    #print("actual:",newsgroups_train.target_names[newsgroups_test.target[i]],"predicted:",newsgroups_train.target_names[newsgroups_train.target[ansi[i]]])

print("jaccard done")
#------------------eucledean------------
nbrse = NearestNeighbors(n_neighbors=2).fit(train2,y_train)
distancese,indicese = nbrse.kneighbors(query2)
ansie = []
y_prede = []
for i in range(len(indicese)):
    ansie.append(indicese[i][0])
    #print("actual:",newsgroups_test.target[i],"predicted:",newsgroups_train.target[ansi[i]])
    y_prede.append(y_train[ansie[i]])
    #print("actual:",newsgroups_train.target_names[newsgroups_test.target[i]],"predicted:",newsgroups_train.target_names[newsgroups_train.target[ansi[i]]])
print("euclidean done")

#-----lsa--------
'''from overalltesting import p 
print(p[0])
'''

#---------lda----------
y_pred =[]
for i in range(len(y_predc)):
    fin = [y_predc[i],y_predj[i],y_prede[i]]
    y_pred.append(answer(fin))



print("best of all done")

y_pred = np.array(y_pred)
print(classification_report(y_test, y_pred, target_names=categories))
cm = pd.DataFrame(confusion_matrix(y_test, y_pred),index=['comp.graphics:true', 'sci.space:true','rec.sport.baseball:true','sci.electronics:true'],columns=['comp.graphics:pred', 'sci.space:pred','rec.sport.baseball:pred','sci.electronics:pred'])
#cm = pd.DataFrame(pd.crosstab(y_test,y_pred),index=categories,columns=categories)
cw = cm.sum(axis=0)

row_df = pd.DataFrame([cw],index=["All"])
cm = pd.concat([ cm,row_df])
cm["All"] = cm.sum(axis=1)
print("confusion matrix:\n",cm)
print("accuracy:",metrics.accuracy_score(y_test, y_pred)*100)




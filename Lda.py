import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import re, nltk,gensim,spacy


categories = ['comp.graphics', 'sci.space','rec.sport.baseball','sci.electronics']

newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Remove Emails
X_train = [re.sub('\S*@\S*\s?', '', sent) for sent in X_train]
X_test = [re.sub('\S*@\S*\s?', '', sent) for sent in X_test]

# Remove new line characters
X_train = [re.sub('\s+', ' ', sent) for sent in X_train]
X_test = [re.sub('\s+', ' ', sent) for sent in X_test]

# Remove distracting single quotes
X_train = [re.sub("\'", "", sent) for sent in X_train]
X_test = [re.sub("\'", "", sent) for sent in X_test]

X_train = [re.sub(">", "", sent) for sent in X_train]
X_test = [re.sub(">", "", sent) for sent in X_test]


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(X_train))
data_wordstest = list(sent_to_words(X_test))

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
X_train= lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
X_test = lemmatization(data_wordstest, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#lda = LatentDirichletAllocation()
lda = LatentDirichletAllocation(max_iter=10,               # Max learning iterations
                                      learning_method='online',   
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                     )



vec = CountVectorizer(analyzer='word',       
                            min_df=10,                        # minimum reqd occurences of a word 
                            stop_words='english',             # remove stop words
                            lowercase=True,                   # convert all words to lowercase
                            token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                            # max_features=50000,             # max number of uniq words
                            )

#vec = CountVectorizer(stop_words='english')

train2 = vec.fit_transform(X_train)
query2 = vec.transform(X_test)
#print(X_train)
lda.fit(train2,y_train)
train = lda.transform(train2)
qu = lda.transform(query2)

'''from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))
X_train_lsa = lsa.fit_transform(train)
X_test_lsa = lsa.transform(qu)'''

knn_lda = KNeighborsClassifier(n_neighbors=5,metric='cosine')
knn_lda.fit(train, y_train)

y_pred = knn_lda.predict(qu)

y_pred = np.array(y_pred)

print(classification_report(y_test, y_pred, target_names=categories))
cm = pd.DataFrame(confusion_matrix(y_test, y_pred),index=['comp.graphics:true', 'sci.space:true','rec.sport.baseball:true','sci.electronics:true'],columns=['comp.graphics:pred', 'sci.space:pred','rec.sport.baseball:pred','sci.electronics:pred'])
#cm = pd.DataFrame(pd.crosstab(y_test,y_pred),index=categories,columns=categories)
cw = cm.sum(axis=0)
cm["All"] = cm.sum(axis=1)
row_df = pd.DataFrame([cw],index=["All"])
cm = pd.concat([ cm,row_df])
print("confusion matrix:\n",cm)
print("accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

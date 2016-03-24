import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import re, math
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import pickle
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer

#short_pos = open("short_reviews/pro_GMO_Hedge","r",encoding='utf-8',errors='ignore').read()
#short_neg = open("short_reviews/anti_GMO_Hedge","r",encoding='utf-8',errors='ignore').read()
#short_pos = open("short_reviews/positive.txt","r").read()
#short_neg = open("short_reviews/negative.txt","r").read()
short_pos = open("test-pro.txt","r").read()
short_neg = open("test-con.txt","r").read()

labels = []
Features = []

   
for p in short_pos.split('\n'):
    Features.append(p)
    labels.append("pos")
    
for p in short_neg.split('\n'):
    Features.append(p)
    labels.append("neg")
   
#get 6-Fold cross validation for Accuracy,Recall,Prediction
#num_folds = 6
#training = posFeatures + negFeatures
#cv = cross_validation.KFold(len(training),n_folds=6, shuffle=True, random_state=42)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(Features, labels, test_size=0.2, random_state=42)
#word_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
word_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)

features_train = word_vectorizer.fit_transform(features_train).toarray()
features_test = word_vectorizer.transform(features_test).toarray()

print(features_train)
print(features_test)

print(features_train.size + features_test.size)
print(len(labels_train)+ len(labels_test))
features_train = features_train[:1000]
labels_train = labels_train[:1000]





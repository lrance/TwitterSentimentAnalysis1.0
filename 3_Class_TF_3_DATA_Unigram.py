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


short_pos = open("manunally data/Dataset_pro_Manunally.txt","r").read()
short_neg = open("manunally data/Dataset_con_Manunally.txt","r").read()
short_Neutral = open("manunally data/Dataset_Neutral_Manunally.txt","r").read()

labels = []
Features = []
   
for p in short_pos.split('\n'):
    Features.append(p)
    labels.append("pos")
       
for p in short_neg.split('\n'):
    Features.append(p)
    labels.append("neg")

for p in short_Neutral.split('\n'):
    Features.append(p)
    labels.append("neu")
    
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(Features, labels, test_size=0.25, random_state=42)
word_vectorizer = CountVectorizer(analyzer='word', min_df=1)

features_train = word_vectorizer.fit_transform(features_train).toarray()
features_test = word_vectorizer.transform(features_test).toarray()
print(features_train.size + features_test.size)
print(len(labels_train)+ len(labels_test))
features_train = features_train[:1000]
labels_train = labels_train[:1000]

###### BernoulliNB Classifier
BernoulliNB = BernoulliNB()
BernoulliNB.fit(features_train,labels_train)
print("BernoulliNB_classifier accuracy percent:", (BernoulliNB.score(features_test,labels_test)))

###### MultinomialNB Classifier
MultinomialNB = MultinomialNB()
MultinomialNB.fit(features_train,labels_train)
print("MNB_classifier accuracy percent:", (MultinomialNB.score(features_test,labels_test)))

###### Logistic Regression
lr = LogisticRegression()
lr.fit(features_train,labels_train)
print("Logistic Regression_classifier accuracy percent:",(lr.score(features_test,labels_test)))

###### LinearSVC
LinearSVC_classifier = LinearSVC()
LinearSVC_classifier.fit(features_train,labels_train)
print("LinearSVC_classifier accuracy percent:", (LinearSVC_classifier.score(features_test,labels_test)))




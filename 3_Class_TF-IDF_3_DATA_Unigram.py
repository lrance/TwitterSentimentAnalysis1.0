import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
import string
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_selection import SelectKBest, chi2


short_pos = open("manunally data/Dataset_pro_Manunally.txt","r").read()
short_neg = open("manunally data/Dataset_con_Manunally.txt","r").read()
short_Neutral = open("manunally data/Dataset_Neutral_Manunally.txt","r").read()

all_words = []
documents = []
tfLine = []

for p in short_pos.split('\n'):
    tfLine.append(p)
    documents.append("pos")
    
for p in short_neg.split('\n'):
    tfLine.append(p)
    documents.append("neg")
    
for p in short_Neutral.split('\n'):
    tfLine.append(p)
    documents.append("neu")
            
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(tfLine, documents, test_size=0.25, random_state=42)
#tf = TfidfVectorizer(sublinear_tf = True, analyzer='word', ngram_range=(1,3), lowercase=True, min_df=0, stop_words='english')
tf = TfidfVectorizer(sublinear_tf = True, analyzer='word', lowercase=True, min_df=0)

features_train = tf.fit_transform(features_train).toarray()
features_test  = tf.transform(features_test).toarray()


print(features_train.size + features_test.size)
print(len(labels_train)+ len(labels_test))

features_train = features_train[:1000]
labels_train   = labels_train[:1000]

#LogisticRegression
lr = LogisticRegression()
lr.fit(features_train,labels_train)
print("Logistic Regression_classifier accuracy percent:",(lr.score(features_test,labels_test)))


###BernoulliNB
BernoulliNB = BernoulliNB()
BernoulliNB.fit(features_train,labels_train)
print("BernoulliNB_classifier accuracy percent:", (BernoulliNB.score(features_test,labels_test)))

###trains a Naive Bayes Classifier
#classifier = NaiveBayesClassifier.train(trainFeatures)

###trains a MultinomialNB Classifier
MultinomialNB = MultinomialNB()
MultinomialNB.fit(features_train,labels_train)
print("MNB_classifier accuracy percent:", (MultinomialNB.score(features_test,labels_test)))

######
LinearSVC_classifier = LinearSVC()
LinearSVC_classifier.fit(features_train,labels_train)
print("LinearSVC_classifier accuracy percent:", (LinearSVC_classifier.score(features_test,labels_test)))

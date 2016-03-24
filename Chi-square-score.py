import nltk
import random
from sklearn import cross_validation
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import re, math
import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from sklearn.svm import SVC, LinearSVC
from nltk.corpus import stopwords
import numpy as np
from scipy import interp
import pylab as pl
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt

    
#short_pos = open("short_reviews/positive.txt","r").read()
#short_neg = open("short_reviews/negative.txt","r").read()
#short_pos = open('short_reviews/pro_GMO_Hedge','r',errors='ignore').read()
#short_neg = open('short_reviews/anti_GMO_Hedge','r',errors='ignore').read()
short_pos = open("manunally data/Dataset_pro_Manunally.txt","r").read()
short_neg = open("manunally data/Dataset_con_Manunally.txt","r").read()

#build frequency distibution of all words and
#then frequency distributions of words within positive and negative labels

#builds dictionary of word scores based on chi-squared test
def create_word_scores():

    posWord_score = []
    negWord_score = []

    for i in short_pos.split('\n'):
        posWords = word_tokenize(i)
        posWord_score.append(posWords)

    for i in short_neg.split('\n'):
        negWords = word_tokenize(i)
        negWord_score.append(negWords)
        
    word_scores = {}

    posWord_score = list(itertools.chain(*posWord_score))
    negWord_score = list(itertools.chain(*negWord_score))

    #build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    
    for word in posWord_score:
        word_fd[word.lower()] += 1
        cond_word_fd["pos"][word.lower()] += 1
    for word in negWord_score:
        word_fd[word.lower()] += 1
        cond_word_fd["neg"][word.lower()] += 1
        
    #finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd["pos"].N()
    neg_word_count = cond_word_fd["neg"].N()
    total_word_count = pos_word_count + neg_word_count
    
    #Chi-Squared Informative Gain
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd["pos"][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd["neg"][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
        
    return word_scores

#finds word scores
word_scores = create_word_scores()
#print(word_scores)

#Bigram Collocations
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words,bigrams)])

#BaseLine Bag of Words Feature Extraction
def word_feats(words):
    return dict([(word, True) for word in words])
#Remove Stopwords Feature Extraction
def word_stop(words):
    return dict([word, True] for word in words if not word in stopwords.words('english'))

#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda ws: ws[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

#number of features set as 10000 default
best_words = find_best_words(word_scores, 100)
print(best_words)
#evaluate_features(best_word_features)
#evaluate_features(word_stop)
#evaluate_features(bigram_word_feats)

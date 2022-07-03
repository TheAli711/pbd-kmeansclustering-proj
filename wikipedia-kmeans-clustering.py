#!/usr/bin/env python
# coding: utf-8

import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# This code assums that you have articles in the folder named "articles" on the same level as this file.
paths = os.listdir("./articles")
current_directory = os.getcwd()
abs_paths = [current_directory +"/articles/"+s for s in paths]
start = time.time()
tfidf = TfidfVectorizer(input='filename', strip_accents="unicode", stop_words="english")
tfidf = tfidf.fit_transform(abs_paths[slice(0,50)])
end = time.time()
duration = end-start
print("Time Taken to do TF-IDF: " + str(duration))
start = time.time()
kmeans =  KMeans(n_clusters=8)
kmeans.fit(tfidf)
end = time.time()
duration = end-start
print("Time Taken to do KMeans Clustering: " + str(duration))
freq = {}
for item in kmeans.labels_:
    if (item in freq):
       freq[item] += 1
    else:
        freq[item] = 1

print("Cluster : Number of Articles")
for key, value in freq.items():
    print ("% d \t: % d"%(key, value))


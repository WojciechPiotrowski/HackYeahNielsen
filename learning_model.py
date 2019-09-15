# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 08:22:50 2019

@author: daan9001
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import pickle

from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from wordcloud import WordCloud, STOPWORDS

import common_words
import matplotlib.pyplot as plt


# *********** START Function definitions ***********
def show_wordcloud():

    stopwords = set(STOPWORDS)

    for i in range(len(labelnb)):

        label=data_gr.Labels.loc[i]
        text =data_gr['text1'].loc[i]

        wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=70,
            max_font_size=40,
            scale=3,
            random_state=1,
            colormap='Dark2'

        ).generate(str(text))

        fig = plt.figure(1, figsize=(15, 45))
        fig.add_subplot(10,2,i+1).set_title('Category: '+ str(labelnb[i]),fontsize=20)
        fig.subplots_adjust(hspace=0.1,wspace=0.1)

        plt.axis('off')
        fig.savefig('plot3.png')

        plt.imshow(wordcloud)

    plt.show()
# *********** END Function definitions ***********


# Read data if needed:
data_all=pd.read_csv('data_all.csv').reset_index(drop=True)
data_all=data_all.drop('Unnamed: 0',1)

# Filter out non english:
data_all = data_all[data_all.lang=='en']

# Vectorize:
vectorizer=TfidfVectorizer(min_df=20, max_df=0.6,ngram_range=(1,2))
document_term_matrix = vectorizer.fit_transform(data_all.text1)

svd = TruncatedSVD(n_components=200)
svd.fit(document_term_matrix)
dtm_svd = svd.transform(document_term_matrix)

data_normalized = Normalizer().fit_transform(dtm_svd)

clustering = KMeans(n_clusters=10, n_jobs=3)
clustering.fit(data_normalized)

labels = clustering.predict(data_normalized)

data_all['Labels']=labels.tolist()
labelnb = np.unique(labels).tolist()


# *** Write the model ***
# pickle.dump(clustering, open('kmeans_model15.sav', 'wb'))

# *** Load the model from disk ***
#loaded_model = pickle.load(open('kmeans_model.sav', 'rb'))

# *** Visualize with wordcloud ***
# data_gr=data_all.groupby('Labels')['text1'].apply(' '.join).reset_index()
# show_wordcloud()

# *** Show statistics with common words ***
data_statistics = common_words.common_words(data_all)
print(data_statistics)


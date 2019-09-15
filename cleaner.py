# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 14:23:23 2019

@author: daan9001
"""
import pandas as pd
import langdetect
import nltk
import string


def read_data(file):
    file_name = file.split('\\')[-1]

    file_text = open(file, 'r', encoding='utf-8').read()

    data = pd.DataFrame({'id':[file_name,], 'text':[file_text,]})

    return data

def categorize_language(df):

    try:
        df['lang']=langdetect.detect(df['text'][0])
        return df

    except Exception as e:
        df['lang']='numbers'
        return df

def shit_remover(df):
    def preprocess_text(text):
        text_lower = text.lower()
        words = nltk.word_tokenize(text_lower)
        words_no_punc = [w for w in words if not w in string.punctuation]
        words_no_stopwords = [
            w
            for w in words_no_punc
            if not w in nltk.corpus.stopwords.words('english')
        ]
        return ' '.join(words_no_stopwords)

    df['text1'] = df.text.apply(lambda x: preprocess_text(x))

    df['text1'] = df['text1'].apply(lambda x: x.replace('*', '').replace('=', ''))

    df['text1'] = df['text1'].apply(lambda x: x.replace('/', '').replace('-', ''))
    return df

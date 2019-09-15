#############################################################################################
# Authors: xyz
# Code modified on 09-Oct-2019
# Python version 3.7
#############################################################################################

import os
import sys
import pandas as pd
import numpy as np
import multiprocessing
import tkinter as tk
from tkinter import filedialog
import cleaner as cleaner
from glob import glob
import multiprocessing
from multiprocessing import Pool, cpu_count
import use_existing_model as labeler
import time


def main(file):
    # reading data insides
    file = cleaner.read_data(file)
    # categorizing data with language
    df_lang = cleaner.categorize_language(file)
    df_lang = df_lang[df_lang.lang=='en']
    if df_lang.empty:
        return df_lang
    else:
        #clearing and removing stopwords
        df_cleaned = cleaner.shit_remover(df_lang)

        return df_cleaned

def appender(lst):
    whole = pd.DataFrame()
    for result in lst:
        whole = whole.append(result, ignore_index=True).reset_index(drop=True)
    return whole

if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askdirectory(title='Choose directory with files to categorise:')

    files = glob(file_path+'\*')
    p = Pool(cpu_count()-1)
    multiprocessing.freeze_support()
    dfs = p.map(main,files)
    p.close()
    p.join()
    df = appender(dfs)
    out_file_path = filedialog.askdirectory(title='Choose output file directory:')
    df.to_csv(out_file_path+r'\data_cleaned.csv', index=False)

    df_labeled = labeler.apply_model(df)


    labelki = pd.DataFrame({'label_name': ['Minecraft errors', 'Natural language', 'C++',
                                           'Internet downloads', 'Web development', 'Roblox',
                                           'Java', 'Code various', 'Movie downloads', 'Java script'],
                            'Labels': list(range(0, 10))})

    final_label = pd.merge(df_labeled, labelki, how='left')

    final_label[['label_name','id','text']].to_csv(out_file_path + r'\data_labeled.csv', index=False)
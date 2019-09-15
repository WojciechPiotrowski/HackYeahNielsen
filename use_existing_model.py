import pandas as pd
import pickle
import time
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def apply_model(data_sample):
    vectorizer = CountVectorizer(min_df=20, max_df=0.6)
    document_term_matrix = vectorizer.fit_transform(data_sample.text1)
    svd = TruncatedSVD(n_components=200)
    svd.fit(document_term_matrix)
    dtm_svd = svd.transform(document_term_matrix)
    data_normalized = Normalizer().fit_transform(dtm_svd)
    with open('kmeans_model10.sav', 'rb') as handle:
        clustering = pickle.load(handle)
    labels = clustering.predict(data_normalized)
    data_sample['Labels'] = labels.tolist()
    # data_sample.to_csv(r'C:\Users\piwo8001\PycharmProjects\hackyeah\data_7_labeled.csv', sep=';')

    # =============================================================================
    # def get_top_features_cluster(document_term_matrix, labels, n_feats):
    #
    #     dfs=pd.DataFrame()
    #     labelnb = np.unique(labels).tolist()
    #     for label in labelnb:
    #         id_temp = data_sample[data_sample.Labels==0].index.tolist() # indices for each cluster
    #         x_means = np.mean(document_term_matrix[id_temp], axis = 0) # returns average score across cluster
    #         x_means=np.array(x_means).flatten().tolist()
    #         sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top scores
    #         features = vectorizer.get_feature_names()
    #         best_features = [(features[i], x_means[i]) for i in sorted_means]
    #         df = pd.DataFrame(best_features, columns = ['features', 'score'])
    #         df['Label']=label
    #         dfs=dfs.append(df)
    #
    #     return dfs
    #
    # df_top_words_per_category = get_top_features_cluster(document_term_matrix, labels, 15)
    #
    #
    # data_gr=data_sample.groupby('Labels')['text'].apply(' '.join).reset_index()
    #
    # =============================================================================

    # def show_wordcloud():
    #     stopwords = set(STOPWORDS)
    #
    #     for i in range(len(labelnb)):
    #         label = data_gr.Labels.loc[i]
    #         text = data_gr['text'].loc[i]
    #
    #         wordcloud = WordCloud(
    #             background_color='white',
    #             stopwords=stopwords,
    #             max_words=70,
    #             max_font_size=40,
    #             scale=3,
    #             random_state=1,
    #             colormap='Dark2'
    #
    #         ).generate(str(text))
    #
    #         fig = plt.figure(1, figsize=(25, 15))
    #         fig.add_subplot(4, 3, i + 1).set_title('Category: ' + str(labelnb[i]), fontsize=20)
    #         fig.subplots_adjust(hspace=0.1, wspace=0.1)
    #
    #         plt.axis('off')
    #
    #         plt.imshow(wordcloud)
    #
    #     plt.show()

    # show_wordcloud()
    return data_sample



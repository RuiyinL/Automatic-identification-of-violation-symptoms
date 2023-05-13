import nltk
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import fasttext
from preprocessing.w2vemb import EMB
from preprocessing.managedb import ManageDB
from preprocessing.preprocessing import preprocess

# load SO_w2v_200
wv_from_bin = KeyedVectors.load_word2vec_format("D:\data\word_embedding\SO_vectors_200.bin", binary=True)
# load FastText_200
fasttext.FastText.eprint = lambda x: None
# ft = fasttext.load_model('D:\data\word_embedding\cc.en.200.bin')
# ft = fasttext.load_model('D:\data\word_embedding\cc.en.100.bin')
# ft = fasttext.load_model('D:\data\word_embedding\cc.en.300.bin')

# load GloVe_200
# gensim_file = 'D:\data\word_embedding\glove.twitter.27B.200d.txt'
# md = ManageDB()
# md.add_file2db('glove.twitter.27B.200d', gensim_file, 200, 1193513)
# GloVe = EMB(name='glove.twitter.27B.200d', dimensions=200)

read_path1 = r'D:\data\Violation symptoms.xlsx'
data1 = pd.read_excel(read_path1, sheet_name='combination', na_values='n/a')
violation_comment = data1['Comment'].tolist()

read_path2 = r'D:\data\Randomly_selected_comments.xlsx'
data2 = pd.read_excel(read_path2, sheet_name='Comments', na_values='n/a')
non_violation_comment = data2['Comment'].tolist()

word2id = wv_from_bin.key_to_index  # dict: {word, index}; example: {'a': 0, 'b', 1, ...}
# ft_word_dic = ft.words

'''Feature Extraction'''
def get_word_vectors(embeding_model, word):
    if embeding_model == 'SO_w2v':
        if word == '0':
            word = '<UNK>'  # replace '0'
        if word in word2id:
            vector = wv_from_bin.get_vector(word)
        else:
            vector = np.array([0.] * 200, dtype=np.float64)

    if embeding_model == 'FastText':
        if word == '0':
            word = '<UNK>'
        if word in ft_word_dic:
            vector = ft.get_word_vector(word)
        else:
            vector = np.array([0.] * 300, dtype=np.float64)     # adjust the dimension

    if embeding_model == 'GloVe':
        if word == '0':
            word = '<UNK>'
        if word in GloVe:
            vector = np.array(GloVe.get_vector(word))
        else:
            vector = np.array([0.] * 200, dtype=np.float64)
    return vector

def get_sen_vectors(embeding_model, sentence):   # list; example: ['this', 'line', 'violat', 'new', 'hack', 'rule']
    # sentence_vec = np.array([0.] * 200, dtype=np.float64)
    sentence_vec = np.array([0.] * 300, dtype=np.float64)
    for word in sentence:
        sentence_vec += get_word_vectors(embeding_model, word)
    return sentence_vec  # return: <class 'numpy.ndarray'>


if __name__ == '__main__':
    '''Manually choose the feature extraction method (i.e., SO_w2v, FastText, GloVe)'''
    preprocessed_comment = []
    for item in violation_comment:
        # preprocessed_comment.append(get_sen_vectors('SO_w2v_200', preprocess(item)))
        preprocessed_comment.append(get_sen_vectors('FastText', preprocess(item)))
        # preprocessed_comment.append(get_sen_vectors('GloVe_200', preprocess(item)))
    Preprocessed_Data = pd.DataFrame(preprocessed_comment)

    # Preprocessed_Data.to_csv(r'D:\data\extracted_features\SO_w2v_200_violation.csv', index=False)   # SO_w2v_200
    # Preprocessed_Data.to_csv(r'D:\data\extracted_features\FastText_200_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'D:\data\extracted_features\FastText_100_violation.csv', index=False)
    Preprocessed_Data.to_csv(r'D:\data\extracted_features\FastText_300_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'D:\data\extracted_features\GloVe_200_violation.csv', index=False)

    # irrelevant data
    preprocessed_comment = []
    for item in non_violation_comment:
        # preprocessed_comment.append(get_sen_vectors('SO_w2v_200', preprocess(item)))
        preprocessed_comment.append(get_sen_vectors('FastText', preprocess(item)))
        # preprocessed_comment.append(get_sen_vectors('GloVe_200', preprocess(item)))
    Preprocessed_Data = pd.DataFrame(preprocessed_comment)

    # Preprocessed_Data.to_csv(r'D:\data\extracted_features\SO_w2v_200_non_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'D:\data\extracted_features\FastText_200_non_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'D:\data\extracted_features\FastText_100_non_violation.csv', index=False)
    Preprocessed_Data.to_csv(r'D:\data\extracted_features\FastText_300_non_violation.csv', index=False)
    # Preprocessed_Data.to_csv(r'D:\data\extracted_features\GloVe_200_non_violation.csv', index=False)
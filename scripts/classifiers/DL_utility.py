import torch
import gensim
import fasttext
import numpy as np
import pandas as pd
import torch.nn as nn
from torchtext.vocab import vocab
from preprocessing.w2vemb import EMB
from torchtext.data import get_tokenizer
from collections import Counter, OrderedDict
from gensim.models.keyedvectors import KeyedVectors
from torchtext.data.utils import ngrams_iterator
from sklearn.model_selection import train_test_split
from preprocessing.preprocessing import preprocess, pad_or_cut
from w2vembeddings.managedb import ManageDB
from w2vembeddings.w2vemb import EMB

path0 = r'D:\data\Randomly_selected_comments.xlsx'
path1 = r'D:\data\Violation symptoms.xlsx'
path_w2v0 = r'D:\data\extracted_features\SO_w2v_200_non_violation.csv'  # negative data (for TextCNN_w2v)
path_w2v1 = r'D:\data\extracted_features\SO_w2v_200_violation.csv'      # positive data (for TextCNN_w2v)
# path_ft0 = r'D:\data\extracted_features\FastText_200_non_violation.csv'  # negative data (for TextCNN_fastText)
# path_ft1 = r'D:\data\extracted_features\FastText_200_violation.csv'      # positive data (for TextCNN_fastText)
path_ft0 = r'D:\data\extracted_features\FastText_100_non_violation.csv'  # negative data (for TextCNN_fastText)
path_ft1 = r'D:\data\extracted_features\FastText_100_violation.csv'      # positive data (for TextCNN_fastText)

label0 = pd.read_excel(path0, sheet_name='Comments', na_values='n/a')
label1 = pd.read_excel(path1, sheet_name='combination', na_values='n/a')


wv_from_bin = KeyedVectors.load_word2vec_format("D:\data\word_embedding\SO_vectors_200.bin", binary=True)
fasttext.FastText.eprint = lambda x: None
ft = fasttext.load_model('D:\data\word_embedding\cc.en.100.bin')
# ft = fasttext.load_model('D:\data\word_embedding\cc.en.200.bin')
# ft = fasttext.load_model('D:\data\word_embedding\cc.en.300.bin')
word2id = wv_from_bin.key_to_index  # dict: {word, index}; like this: {'a': 0, 'b', 1, ...}
ft_word_dic = ft.words              # vocabulary list; like this: ['the', 'design', ..., 'Zwicke']

# gensim_file = 'D:\data\word_embedding\glove.twitter.27B.200d.txt'
# md = ManageDB()
# md.add_file2db('glove.twitter.27B.200d', gensim_file, 200, 1193513)    # write it into database (only need to run in the first time)
GloVe_200 = EMB(name='glove.twitter.27B.200d', dimensions=200)

def dataset_split(args):
    '''
    split training set, validation set, test set,比例是6：2：2
    '''
    sentences0 = pd.read_excel(path0, sheet_name='Comments', na_values='n/a')
    sentences1 = pd.read_excel(path1, sheet_name='combination', na_values='n/a')
    x0 = sentences0['Comment'].tolist()
    x1 = sentences1['Comment'].tolist()

    y0 = label0['Label'].tolist()
    y1 = label1['Label'].tolist()
    percentage1 = 1/5    # test set: 1/5; train + validation set: 4/5
    percentage2 = 1/4    # train set: 3/5; validation set: 1/5
    seed = 6
    X_train_valid0, X_test0, Y_train_valid0, Y_test0 = train_test_split(x0, y0, test_size=percentage1, random_state=seed)
    X_train_valid1, X_test1, Y_train_valid1, Y_test1 = train_test_split(x1, y1, test_size=percentage1, random_state=seed)
    X_train0, X_valid0, Y_train0, Y_valid0 = train_test_split(X_train_valid0, Y_train_valid0, test_size=percentage2, random_state=seed)
    X_train1, X_valid1, Y_train1, Y_valid1 = train_test_split(X_train_valid1, Y_train_valid1, test_size=percentage2, random_state=seed)

    X_train = X_train0 + X_train1  # list
    X_valid = X_valid0 + X_valid1
    X_test = X_test0 + X_test1
    Y_train = Y_train0 + Y_train1
    Y_valid = Y_valid0 + Y_valid1
    Y_test = Y_test0 + Y_test1
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def pad_or_cut(value: np.array, target_length: int):  # value: np.ndarray, target_length: int
    # value = np.array(value)
    data_row = None
    if len(value) < target_length:
        data_row = np.pad(value, (0, target_length - len(value)), 'constant', constant_values=int(0))
    elif len(value) > target_length:
        data_row = value[:target_length]
    return data_row


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        '''
        :param patience (int): How long to wait after last time validation loss improved. Default: 6
        :param verbose (bool): If True, prints a message for each validation loss improvement. Default: False
        :param delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        :param path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
        :param trace_func (function): trace print function. Default: print
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def generate_tensor(embeding_model, sentence, _len, args):
    """transform to FloatTensor
    :param embeding_model: 'SO_w2v_200' or 'FastText_200'
    :param sentence: word list
    :param _len: sentence length
    :return: tensor
    """
    tensor = torch.zeros([args.max_length, args.embedding_dim])
    for index in range(0, args.max_length):
        if index >= len(sentence):
            break
        else:
            word = sentence[index]
            if embeding_model == 'SO_w2v':
                if word == '0':
                    word = '<UNK>'  # replace '0'
                if word in word2id:
                    # tensor[index] = wv_from_bin.get_vector(word)  # vector, <class 'numpy.ndarray'>
                    tensor[index] = torch.FloatTensor(wv_from_bin.get_vector(word))

            if embeding_model == 'FastText':
                if word == '0':
                    word = '<UNK>'  # replace '0'
                if word in ft_word_dic:
                    # tensor[index] = ft.get_word_vector(word)
                    tensor[index] = torch.FloatTensor(ft.get_word_vector(word))

            if embeding_model == 'GloVe':
                if word == '0':
                    word = '<UNK>'  # replace '0'
                if word in GloVe_200:
                    tensor[index] = torch.FloatTensor(np.array(GloVe_200.get_vector(word)))
    return tensor.unsqueeze(0)


def Save_Checkpoint(epoch, epochs_since_improvement, model, optimizer, loss):#, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    torch.save(state, filename)

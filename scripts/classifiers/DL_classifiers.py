import os
import torch
import time
import numpy as np
import torch.nn as nn
from tqdm import trange
from sklearn import metrics
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchtext.transforms import VocabTransform
from torchtext.data.utils import ngrams_iterator
import matplotlib.pyplot as plt
from preprocessing.preprocessing import preprocess
from classifiers.DL_models import parse_args
from classifiers.DL_models import TextCNN_voc, TextCNN_w2v, TextCNN_fastText, TextCNN_GloVe
from classifiers.DL_utility import EarlyStopping, dataset_split, pad_or_cut, generate_tensor

# torch.cuda.empty_cache()        # release memory
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, text_list, text_label, args):
        '''
        :param text_list: sentence_list
        :param text_label: sentence_label
        :param args: load max_length
        '''
        super(MyDataset, self).__init__()
        text_vocab, vocab_transform = self.build_vocab(text_list)
        self.text_list = text_list
        self.text_label = text_label
        self.text_vocab = text_vocab
        self.vocab_transform = vocab_transform
        self.data = self.generate_data()
        self.size = self.get_vocab_size()
        self.max_length = args.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id_index):
        token = self.data[id_index]
        label = self.text_label[id_index]
        return token, label

    # Build a vocabulary
    def build_vocab(self, text_list):   # sentence_list = ['xx',..., 'xxx']
        total_word_list = []
        for sentence in text_list:
            sentence = preprocess(sentence)
            total_word_list += list(ngrams_iterator(sentence, 2))  # n-gram
        counter = Counter(total_word_list)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        special_token = ['<UNK>', '<SEP>']
        text_vocab = vocab(ordered_dict, specials=special_token)
        text_vocab.set_default_index(0)
        vocab_transform = VocabTransform(text_vocab)
        return text_vocab, vocab_transform

    def generate_data(self):
        token_id_list = []
        for sentence in self.text_list:
            sentence_words = preprocess(sentence)
            sentence_id_list = np.array(self.vocab_transform(sentence_words))
            sentence_id_list = pad_or_cut(sentence_id_list, args.max_length)
            token_id_list.append(sentence_id_list)
        return token_id_list

    def get_vocab_size(self):
        return len(self.text_vocab)


class MyDataset_pre_trained(Dataset):
    def __init__(self, text_list, text_label):
        super(MyDataset_pre_trained, self).__init__()
        self.x = text_list
        self.y = text_label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, id_index):
        label = self.y[id_index]
        sentence = self.x[id_index]
        word_list = preprocess(sentence)
        word_list = np.array(word_list)
        word_list = pad_or_cut(word_list, args.max_length)
        if args.TextCNN_w2v:
            sentence_vector = generate_tensor('SO_w2v', word_list, len(word_list), args)
        if args.TextCNN_fastText:
            # sentence_vector = generate_tensor('FastText_200', word_list, len(word_list), args)  # return <class 'torch.Tensor'>
            sentence_vector = generate_tensor('FastText', word_list, len(word_list), args)
        if args.TextCNN_GloVe:
            sentence_vector = generate_tensor('GloVe', word_list, len(word_list), args)
        return sentence_vector, label


def train(args, model, train_iter, valid_iter):
    time_start = time.time()
    if torch.cuda.is_available():
        model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # ============= optimizer =============
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    # ============= Loss function =============
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    # ============= Early Stopping =============
    early_stopping = EarlyStopping(patience=8, verbose=True)

    steps = 0
    total_loss = 0.
    global_step = 0
    train_epoch = trange(args.epoch, colour='blue', desc='train_epoch')
    train_losses = []
    valid_losses = []

    # ============= Training =============
    for epoch in train_epoch:
        model.train()

        for text_token, text_label in train_iter:
            global_step += 1
            optimizer.zero_grad()
            # model_out = model(text_token.to(DEVICE, non_blocking=True))
            # loss = criterion(model_out, text_label.to(DEVICE, non_blocking=True))
            model_out = model(text_token.to(DEVICE))
            loss = criterion(model_out, text_label.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        # print("Training:", epoch_i + 1, "Loss:", np.sum(loss_list))
        training_loss = total_loss / global_step
        print("Epoch:", epoch + 1, "Training loss:", training_loss)
        train_losses.append(training_loss)
        # scheduler.step()

        '''checkpoint'''
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "steps": steps,
                          # "training_loss": training_loss,
                          "global_step": global_step,
                          "total_loss": total_loss}
            path_checkpoint = "D:\scripts\classifiers\models\checkpoint_{}_epoch.pkl".format(epoch + 1)
            torch.save(checkpoint, path_checkpoint)


        '''Validation'''
        valid_loss, valid_result = evaluate(criterion, model, valid_iter)
        valid_losses.append(valid_loss)
        # Early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping!")
            break

    torch.cuda.empty_cache()    # dump GPU cache
    time_end = time.time()
    train_time = time_end - time_start

    # torch.save(model.state_dict(), 'D:\scripts\classifiers\models\My_model.pth')
    torch.save(model, 'D:\scripts\classifiers\models\My_model.pth')

    '''loss vasualization'''
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.ylim(ymin=0, ymax=1.01)
    plt.title("The loss of current model")
    plt.legend(["train loss", 'validation loss'])
    plt.show()
    print('Train time:', train_time, 's')


def evaluate(criterion, model, valid_test_iter):
    model.eval()
    total_loss = 0.
    total_step = 0.
    preds = None
    true_label = None
    with torch.no_grad():
        for text_token, text_label in valid_test_iter:
            model_out = model(text_token.to(DEVICE))
            example_losses = criterion(model_out, text_label.to(DEVICE))
            total_loss += example_losses.item()
            total_step += 1

            if preds is None:
                preds = model_out.detach().cpu().numpy()
                true_label = text_label.detach().cpu().numpy()
            else:
                preds = np.append(preds, model_out.detach().cpu().numpy(), axis=0)
                true_label = np.append(true_label, text_label.detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    print("Predicted labels:", preds)
    result = acc_and_f1(preds, true_label)

    return total_loss / total_step, result


'''Performance'''
def acc_and_f1(preds, Y_test):
    # acc = (preds == Y_test).mean()
    acc = metrics.accuracy_score(y_true=Y_test, y_pred=preds)
    f1 = metrics.f1_score(y_true=Y_test, y_pred=preds, average='weighted')
    precision = metrics.precision_score(y_true=Y_test, y_pred=preds)
    recall = metrics.recall_score(y_true=Y_test, y_pred=preds)
    return [
        "Precision: %0.4f" % precision,
        "Recall: %0.4f" % recall,
        "F1: %0.4f" % f1,
        "Accuracy: %0.4f" % acc
        # "acc_and_f1:": (acc + f1) / 2,
    ]


def load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    if args.TextCNN_voc or args.TextRNN_voc:
        train_data = MyDataset(X_train, Y_train, args)
        # vocab_size = train_data.size
        valid_data = MyDataset(X_valid, Y_valid, args)
        test_data = MyDataset(X_test, Y_test, args)

    if args.TextCNN_w2v or args.TextCNN_fastText or args.TextCNN_GloVe:
        train_data = MyDataset_pre_trained(X_train, Y_train)
        valid_data = MyDataset_pre_trained(X_valid, Y_valid)
        test_data = MyDataset_pre_trained(X_test, Y_test)

    train_iter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_iter = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    return train_iter, valid_iter, test_iter


if __name__ == '__main__':
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        DEVICE = torch.device("cpu")
    global args
    args = parse_args()
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = dataset_split(args)


    '''Training'''
    if args.do_train:
        if args.TextCNN_voc:
            print('=====Train TextCNN_voc=====')
            train_data = MyDataset(X_train, Y_train, args)
            vocab_size = train_data.size
            train_iter, valid_iter, test_iter = load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
            TextCNN_voc_model = TextCNN_voc(args, vocab_size).to(DEVICE)
            train(args, TextCNN_voc_model, train_iter, valid_iter)

        if args.TextCNN_w2v:
            print('=====Train TextCNN_w2v=====')
            train_iter, valid_iter, test_iter = load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
            TextCNN_w2v_model = TextCNN_w2v(args).to(DEVICE)
            train(args, TextCNN_w2v_model, train_iter, valid_iter)

        if args.TextCNN_fastText:
            print('=====Train TextCNN_fastText=====')
            train_iter, valid_iter, test_iter = load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
            TextCNN_fastText_model = TextCNN_fastText(args).to(DEVICE)
            train(args, TextCNN_fastText_model, train_iter, valid_iter)

        if args.TextCNN_GloVe:
            print('=====Train TextCNN_GloVe=====')
            train_iter, valid_iter, test_iter = load_dataset(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
            TextCNN_GloVe_model = TextCNN_GloVe(args).to(DEVICE)
            train(args, TextCNN_GloVe_model, train_iter, valid_iter)


    '''Testing'''
    if args.do_test:
        print("===== Start testing =====")
        criterion = nn.CrossEntropyLoss()
        if args.TextCNN_voc:
            print('=====Test TextCNN_voc=====')
            TextCNN_voc_model = TextCNN_voc(args, vocab_size).to(DEVICE)
            TextCNN_voc_test_loss, TextCNN_voc_result = evaluate(criterion, TextCNN_voc_model, test_iter)
            print("TextCNN_voc_test_loss", TextCNN_voc_test_loss)
            print("TextCNN_voc_result", TextCNN_voc_result)

        if args.TextCNN_w2v:
            print('=====Test TextCNN_w2v=====')
            TextCNN_w2v_model = TextCNN_w2v(args).to(DEVICE)
            TextCNN_w2v_test_loss, TextCNN_w2v_result = evaluate(criterion, TextCNN_w2v_model, test_iter)
            print("TextCNN_w2v_test_loss", TextCNN_w2v_test_loss)
            print("TextCNN_w2v_result", TextCNN_w2v_result)

        if args.TextCNN_fastText:
            print('=====Test TextCNN_fastText=====')
            TextCNN_fastText_model = TextCNN_fastText(args).to(DEVICE)
            TextCNN_fastText_test_loss, TextCNN_fastText_result = evaluate(criterion, TextCNN_fastText_model, test_iter)
            print("TextCNN_fastText_test_loss", TextCNN_fastText_test_loss)
            print("TextCNN_fastText_result", TextCNN_fastText_result)

        if args.TextCNN_GloVe:
            print('=====Test TextCNN_GloVe=====')
            TextCNN_GloVe_model = TextCNN_GloVe(args).to(DEVICE)
            TextCNN_GloVe_test_loss, TextCNN_GloVe_result = evaluate(criterion, TextCNN_GloVe_model, test_iter)
            print("TextCNN_GloVe_test_loss", TextCNN_GloVe_test_loss)
            print("TextCNN_GloVe_result", TextCNN_GloVe_result)

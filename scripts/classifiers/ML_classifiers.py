import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

'''Loading dataset'''
path0 = r'D:\data\extracted_features\SO_w2v_200_non_violation.csv'       # negative data
path1 = r'D:\data\extracted_features\SO_w2v_200_violation.csv'           # positive data
# path0 = r'D:\data\extracted_features\FastText_200_non_violation.csv'    # negative data
# path1 = r'D:\data\extracted_features\FastText_200_violation.csv'        # positive data
# path0 = r'D:\data\extracted_features\GloVe_200_non_violation.csv'      # negative data
# path1 = r'D:\data\extracted_features\GloVe_200_violation.csv'          # positive data
# path0 = r'D:\data\extracted_features\FastText_100_non_violation.csv'    # negative data
# path1 = r'D:\data\extracted_features\FastText_100_violation.csv'        # positive data
# path0 = r'D:\data\extracted_features\FastText_300_non_violation.csv'    # negative data
# path1 = r'D:\data\extracted_features\FastText_300_violation.csv'        # positive data

label0_path = r'D:\data\Randomly_selected_comments.xlsx'
label1_path = r'D:\data\Violation symptoms.xlsx'
label0 = pd.read_excel(label0_path, sheet_name='Comments', na_values='n/a')     # label as '0'
label1 = pd.read_excel(label1_path, sheet_name='combination', na_values='n/a')  # label as '1'

percentage = 1/5    # test set: 1/5，training set: 4/5
seed = 5            # int or None
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)  # 10-fold cross validation

def dataset_split():
    x0 = pd.read_csv(path0)
    x1 = pd.read_csv(path1)
    y0 = label0['Label'].tolist()
    y1 = label1['Label'].tolist()
    X_train0, X_test0, Y_train0, Y_test0 = train_test_split(x0, y0, test_size=percentage, random_state=seed)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(x1, y1, test_size=percentage, random_state=seed)
    # X_train = X_train0 + X_train1
    # X_test = X_test0 + X_test1
    X_train = pd.concat([X_train0, X_train1])
    X_test = pd.concat([X_test0, X_test1])
    Y_train = Y_train0 + Y_train1       # labels of the training dataset; list type
    Y_test = Y_test0 + Y_test1          # labels of the test dataset; list type
    return X_train, Y_train, X_test, Y_test

'''Train classifiers: Build ML classifiers with best parameters'''

def SVM(X, Y):
    clf = svm.SVC(kernel='rbf')  # , probability=True)     # Default: kernel='rbf', probability=False
    param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
    grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    # print(best_parameters)
    clf = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    clf.fit(X, Y)
    print('[SVM] best_parameters', best_parameters)
    return clf

def NB(X, Y):
    # clf = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
    clf = BernoulliNB()
    alpha_can = np.logspace(-2, 1, 10)
    param_grid = {'alpha': alpha_can}
    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=10)
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    clf = BernoulliNB(alpha=best_parameters['alpha'], binarize=0.0, fit_prior=True, class_prior=None)
    clf.fit(X, Y)
    print('[NB] best_parameters:', best_parameters)
    return clf

def LR(X, Y):
    clf = LogisticRegression()
    # clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'max_iter': [10, 100, 250, 500],
                  'class_weight': ['balanced', None],
                  'solver': ['liblinear', 'sag', 'lbfgs', 'newton-cg']}
    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=10, scoring='accuracy')
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    clf = LogisticRegression(random_state=seed, C=best_parameters['C'], max_iter=best_parameters['max_iter'], class_weight=best_parameters['class_weight'], solver=best_parameters['solver'])
    clf.fit(X, Y)
    print('[LR] best_parameters:', best_parameters)
    return clf


def KNN(X, Y):
    clf = KNeighborsClassifier()
    param_grid = {'n_neighbors': [i for i in range(1, 31)],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=10, scoring='accuracy')
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    # print(best_parameters)
    clf = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'], weights=best_parameters['weights'], algorithm=best_parameters['algorithm'])
    clf.fit(X, Y)
    print('[KNN] best_parameters:', best_parameters)
    return clf

def DT(X, Y):
    clf = DecisionTreeClassifier()
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': [i for i in range(2, 15, 1)],
                  'min_samples_leaf': [i for i in range(1, 10, 2)],
                  'min_samples_split': [i for i in range(2, 10, 1)]}
    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=10, scoring='accuracy')
    grid_search.fit(X, Y)
    best_parameters = grid_search.best_estimator_.get_params()
    # print(best_parameters)
    clf = DecisionTreeClassifier(random_state=seed, criterion=best_parameters['criterion'],
                                 max_depth=best_parameters['max_depth'],
                                 min_samples_leaf=best_parameters['min_samples_leaf'],
                                 min_samples_split=best_parameters['min_samples_split'])
    clf.fit(X, Y)
    print('[DT] best_parameters', best_parameters)
    return clf

if __name__ == '__main__':
    # t0 = time()
    X_train_val, Y_train_val, X_test, Y_test = dataset_split()
    X_train_val_std = StandardScaler().fit_transform(X_train_val)
    X_test_std = StandardScaler().fit_transform(X_test)

    ''' ======== ML classifiers ========='''
    # Run the classifier one by one
    # clf = SVM(X_train_val_std, Y_train_val)
    # clf = LR(X_train_val_std, Y_train_val)
    # clf = NB(X_train_val_std, Y_train_val)
    # clf = DT(X_train_val_std, Y_train_val)
    clf = KNN(X_train_val_std, Y_train_val)
    # print("Running Time in %0.3fs" % (time() - t0))  # Running time

    Y_pred = cross_val_predict(clf, X_test_std, Y_test, cv=10)  # list type
    print('Y_pred:', Y_pred.tolist())

    print("The numbers of the testing set: %d" % X_test_std.shape[0])
    print('Wrong prediction: %d' % (Y_test != Y_pred).sum())
    print("-------------------------")
    print('Precision: %0.3f' % metrics.precision_score(Y_test, Y_pred.tolist()))
    print('Recall: %0.3f' % metrics.recall_score(Y_test, Y_pred.tolist()))
    print('F1: %0.3f' % metrics.f1_score(Y_test, Y_pred.tolist()))
    print('Accuracy: %0.3f' % metrics.accuracy_score(Y_test, Y_pred.tolist()))
    print("-------------------------")
    # Classification Report
    print('Classification Report:')
    print(metrics.classification_report(Y_test, Y_pred.tolist()))

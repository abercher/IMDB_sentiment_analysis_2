"""
Train and evaluate model based on TF-idf data representation with Logistic Regression on top.
This model was the baseline model performing the best in the first iteration of this project, as can be seen in
the original repo:
https://github.com/abercher/IMDB_sentiment_analysis/blob/master/IMDB_sent_an_baseline_models/IMDB_sent_an_baseline_models.pdf
"""
import os
import numpy as np
from scipy import sparse
import pickle
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


def print_evaluation_scores(y_true, y_pred, y_prob):
    acc_sc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='binary')
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision = average_precision_score(y_true=y_true, y_score=list(y_prob[:, 1]))
    roc = roc_auc_score(y_true=y_true, y_score=list(y_prob[:, 1]))
    print(f'Accuracy = {acc_sc:.4f}')
    print(f'F1 score binary = {f1:.4f}')
    print(f'Recall score = {recall:.4f}')
    print(f'Average precision score = {precision:.4f}')
    print(f'Air under the ROC = {roc:.4f}')


def main():
    logreg_clf = LogisticRegression()

    ## Load TF-IDF representation of the data
    X_tfidf_train_fn = os.path.join(os.getcwd(), 'Transformed_data/X_tfidf_train.npz')
    X_tfidf_train = sparse.load_npz(X_tfidf_train_fn)
    X_tfidf_valid_fn = os.path.join(os.getcwd(), 'Transformed_data/X_tfidf_valid.npz')
    X_tfidf_valid = sparse.load_npz(X_tfidf_valid_fn)
    X_tfidf_test_fn = os.path.join(os.getcwd(), 'Transformed_data/X_tfidf_test.npz')
    X_tfidf_test = sparse.load_npz(X_tfidf_test_fn)

    ## Load labels
    y_binary_train_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_binary_train.pkl')
    with open(y_binary_train_fn, mode='rb') as f:
        y_binary_train = pickle.load(f)
    y_binary_valid_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_binary_valid.pkl')
    with open(y_binary_valid_fn, mode='rb') as f:
        y_binary_valid = pickle.load(f)
    y_binary_test_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_binary_test.pkl')
    with open(y_binary_test_fn, mode='rb') as f:
        y_binary_test = pickle.load(f)

    ## Train model
    # n_toy = 1000
    # X_tfidf_train = X_tfidf_train[:n_toy]
    # y_binary_train = y_binary_train[:n_toy]

    start = time.time()
    logreg_clf.fit(X_tfidf_train, y_binary_train)
    stop = time.time()
    print(f"Training of Logistic Regression took: {stop - start}")

    ## Test model on Validation set

    y_binary_valid_pred = logreg_clf.predict(X_tfidf_valid)
    y_binary_valid_proba = logreg_clf.predict_proba(X_tfidf_valid)

    start = time.time()
    print_evaluation_scores(y_binary_valid, y_binary_valid_pred, y_binary_valid_proba)
    stop = time.time()

    print(f"Evaluation on validation set took: {stop - start}")

    ## Test model on Test set

    y_binary_test_pred = logreg_clf.predict(X_tfidf_test)
    y_binary_test_proba = logreg_clf.predict_proba(X_tfidf_test)

    start = time.time()
    print_evaluation_scores(y_binary_test, y_binary_test_pred, y_binary_test_proba)
    stop = time.time()

    print(f"Evaluation on test set took: {stop - start}")


if __name__ == "__main__":
    main()

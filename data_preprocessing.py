"""
Prepare the data to have it ready to be fed to a classifier.
Note that the first version of this IMDB sentiment analysis was heavily based on this tutorial:
https://www.oreilly.com/content/perform-sentiment-analysis-with-lstms-using-tensorflow/
I copy it here to give credit to the guy, even if I will modify many aspects.
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import csv
import numpy as np
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer

## Load data
fn = os.path.join(os.getcwd(), os.path.pardir, 'Dataset/IMDB Dataset.csv')

df = pd.read_csv(fn)

#df = df.head(1000)

print("Original Columns are:")
print(df.columns)
print()
print(f"Total number of samples: {len(df)}")
print()
print("First row:")
print(df.loc[0])

# At first, I'm going to replicate the data preprocessing that I had done in 2017, see
# https://github.com/abercher/IMDB_sentiment_analysis/blob/master/IMDB_sent_an_data_preprocessing/IMDB_sent_an_data_preprocessing.pdf

## Clean strings (punctuation and special caracter removal)

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    # The next line is a small improvement over the function of the 2017 notebook
    # Unlike it, it turn "you'll" into "you ll" and not "youll".
    return " ".join(re.sub(strip_special_chars, " ", string.lower()).split())

df['clean_text'] = df['review'].apply(clean_sentences)

print("First review before cleaning:")
print(df.loc[0, 'review'])
print()
print("First review after cleaning")
print(df.loc[0, 'clean_text'])
print()

## Count word per review to determine input length of the LSTM
df['n_words'] = df['clean_text'].str.split().str.len()

#df = df.head(200)

df['n_words'].hist(bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
#plt.show()

# Originally I had used 250 but I would go for 300 now that I look at it again.
input_seq_len = 300

## Turn reviews text into list of word indices (using GloVe list of words) and create look-up table for GloVe embeddings

word_embedding_size = 100
glove_fn = os.path.join(os.getcwd(), 'glove.6B/glove.6B.' + str(word_embedding_size) + 'd.txt')
emb_df = pd.read_table(glove_fn, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')

glove_words_li = list(emb_df.index)
word_vectors = emb_df.values

print("The first 10 words of the GloVe embeddings are:")
print(glove_words_li[:10])

with open("words_list.txt", "wb") as f:
    pickle.dump(glove_words_li, f)
np.save('word_vectors.npy', word_vectors)


def text_2_indices_list(text):
    """
    Turn a text into a sequence of indices of GloVe embedings. We proceed as in
    https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
    :param text: str containing text
    :return: list of integers
    """
    # 0 is used for padding even if it's the index of the token "the"
    indices_li = [0]*input_seq_len
    words_li = text.split(" ")
    for i, word in enumerate(words_li[:min(len(words_li), input_seq_len)]):
        if word in glove_words_li:
            indices_li[i] = glove_words_li.index(word)
        else:
            # An unknown token is assigned the index 399'999.
            # But this corresponds to a word in the GloVe embedding data frame (\"sandberger\").
            # guess that as this word is uncommun, it might serve the purpose of designating unknown words.
            indices_li[i] = 399999

    return indices_li


start = time.time()
df['words_indices_list'] = df['clean_text'].apply(text_2_indices_list)
stop = time.time()

print(f"Turning strings into list of indices took {stop - start} seconds.")

## Label One-hot-encoding

def encode_one_hot(text):
    text = text.lower().strip()
    if text == 'positive':
        return [1, 0]
    elif text == 'negative':
        return[0, 1]
    else:
        print('Problem with this label:')
        print(repr(text))

df['one_hot_label'] = df['sentiment'].apply(encode_one_hot)


## Binary encoding
def encode_binary(text):
    text = text.lower().strip()
    if text == 'positive':
        return 1
    elif text == 'negative':
        return 0
    else:
        print('Problem with this label:')
        print(repr(text))

df['binary_label'] = df['sentiment'].apply(encode_binary)


## Split data into train/valid/test

# The only difference that I will introduce from the start is the train/valid/test split. In 2017, I split the data
# in two equal parts (12.5K each) using the first to train and the second to validate. Now, I recognize that it was
# a mistake and that my assessment of the result was over optimistic (since I used the results on the validation set
# as final results). I will use a 70/15/15 split over the total 50K samples. NN need a lot of data to train and 35K
# is already quite low, but since NN have many parameters, I need a large validation set and 7.5K is also small. At last,
# I need a reasonably big test set in order to have a decent appreciation of the final result.

n_total_samples = len(df)
n_train = int(0.7*n_total_samples)
n_valid = int(0.15*n_total_samples)
n_test = n_total_samples - n_train - n_valid

df = df.sample(frac=1).reset_index(drop=True)

df_train = df.iloc[0: n_train, :]
df_valid = df.iloc[n_train: n_train+n_valid, :]
df_test = df.iloc[n_train+n_valid:, :]


## Save labels

y_one_hot_train = df_train["one_hot_label"].to_list()
y_one_hot_train_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_one_hot_train.pkl')
with open(y_one_hot_train_fn, mode='wb+') as f:
    pickle.dump(y_one_hot_train, f)
y_one_hot_valid = df_valid["one_hot_label"].to_list()
y_one_hot_valid_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_one_hot_valid.pkl')
with open(y_one_hot_valid_fn, mode='wb+') as f:
    pickle.dump(y_one_hot_valid, f)
y_one_hot_test = df_test["one_hot_label"].to_list()
y_one_hot_test_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_one_hot_test.pkl')
with open(y_one_hot_test_fn, mode='wb+') as f:
    pickle.dump(y_one_hot_test, f)

y_binary_train = df_train["binary_label"].to_list()
y_binary_train_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_binary_train.pkl')
with open(y_binary_train_fn, mode='wb+') as f:
    pickle.dump(y_binary_train, f)
y_binary_valid = df_valid["binary_label"].to_list()
y_binary_valid_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_binary_valid.pkl')
with open(y_binary_valid_fn, mode='wb+') as f:
    pickle.dump(y_binary_valid, f)
y_binary_test = df_test["binary_label"].to_list()
y_binary_test_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_binary_test.pkl')
with open(y_binary_test_fn, mode='wb+') as f:
    pickle.dump(y_binary_test, f)


## Save indices lists representation of the reviews

indices_li_train_fn = os.path.join(os.getcwd(), 'Transformed_data/ind_li_train' + str(word_embedding_size) + '.npy')
np.save(indices_li_train_fn, np.array(df_train['words_indices_list'].to_list()))
indices_li_valid_fn = os.path.join(os.getcwd(), 'Transformed_data/ind_li_valid' + str(word_embedding_size) + '.npy')
np.save(indices_li_valid_fn, np.array(df_valid['words_indices_list'].to_list()))
indices_li_test_fn = os.path.join(os.getcwd(), 'Transformed_data/ind_li_test' + str(word_embedding_size) + '.npy')
np.save(indices_li_test_fn, np.array(df_test['words_indices_list'].to_list()))


## Turn text reviews into TF-IDF representations

# I will reuse the parameters that I used in the first version of this project (simplifying a few things)
texts_train = df_train['clean_text'].to_list()
texts_valid = df_valid['clean_text'].to_list()
texts_test = df_test['clean_text'].to_list()

VOCAB_SIZE = 200000

tfidf_transformer = TfidfVectorizer(ngram_range=(1, 3), min_df=0.001, max_features=VOCAB_SIZE)

X_tfidf_train = tfidf_transformer.fit_transform(texts_train)
X_tfidf_train_fn = os.path.join(os.getcwd(), 'Transformed_data/X_tfidf_train.npy')
np.save(indices_li_train_fn, np.array(X_tfidf_train))
X_tfidf_valid = tfidf_transformer.transform(texts_valid)
X_tfidf_valid_fn = os.path.join(os.getcwd(), 'Transformed_data/X_tfidf_valid.npy')
np.save(indices_li_valid_fn, np.array(X_tfidf_valid))
X_tfidf_test = tfidf_transformer.transform(texts_test)
X_tfidf_test_fn = os.path.join(os.getcwd(), 'Transformed_data/X_tfidf_test.npy')
np.save(indices_li_train_fn, np.array(X_tfidf_test))



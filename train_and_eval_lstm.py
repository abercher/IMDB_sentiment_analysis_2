import os
import numpy as np
import pickle
import pandas as pd
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import time


def print_evaluation_scores(y_true, y_pred, y_prob):
    acc_sc = accuracy_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='binary')
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='binary')
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='binary')
    roc = roc_auc_score(y_true=y_true, y_score=list(y_prob))
    print(f'Accuracy = {acc_sc:.4f}')
    print(f'F1 score binary = {f1:.4f}')
    print(f'Recall score = {recall:.4f}')
    print(f'Average precision score = {precision:.4f}')
    print(f'Air under the ROC = {roc:.4f}')




def main():
    ## Load list of indices
    word_embedding_size = 100
    indices_li_train_fn = os.path.join(os.getcwd(), 'Transformed_data/ind_li_train' + str(word_embedding_size) + '.npy')
    indices_li_train = np.load(indices_li_train_fn)
    indices_li_valid_fn = os.path.join(os.getcwd(), 'Transformed_data/ind_li_valid' + str(word_embedding_size) + '.npy')
    indices_li_valid = np.load(indices_li_valid_fn)
    indices_li_test_fn = os.path.join(os.getcwd(), 'Transformed_data/ind_li_test' + str(word_embedding_size) + '.npy')
    indices_li_test = np.load(indices_li_test_fn)

    ## Load binary labels
    y_binary_train_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_binary_train.pkl')
    with open(y_binary_train_fn, mode='rb') as f:
        y_binary_train = pickle.load(f)
    y_binary_valid_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_binary_valid.pkl')
    with open(y_binary_valid_fn, mode='rb') as f:
        y_binary_valid = pickle.load(f)
    y_binary_test_fn = os.path.join(os.getcwd(), 'Transformed_data/Labels/y_binary_test.pkl')
    with open(y_binary_test_fn, mode='rb') as f:
        y_binary_test = pickle.load(f)


    ## Create custom Dataset
    # This class and its custom subclasses allow to loop over the data during training.
    # Whatever form the data is (in one numpy array or spread accross different files),
    # this class puts it in a uniform format which can then be fed to the model (via the Dataloader).
    # It can also be used to add some preprocessing steps to the data
    # The current implementation is basic. For bigger data, one can read the data progressively
    # in the __getitem__ method. But this is not needed here.
    class CustomIMDBDataset(Dataset):
        def __init__(self, indices_matrix, y_binary):
            self.indices_matrix = indices_matrix
            self.y_binary = y_binary

        def __len__(self):
            return self.indices_matrix.shape[0]

        def __getitem__(self, item):
            return self.indices_matrix[item], self.y_binary[item]


    n_toy = 500
    dataset_toy = CustomIMDBDataset(indices_matrix=indices_li_train[:n_toy], y_binary=y_binary_train[:n_toy])
    dataset_train = CustomIMDBDataset(indices_matrix=indices_li_train, y_binary=y_binary_train)
    dataset_test = CustomIMDBDataset(indices_matrix=indices_li_test, y_binary=y_binary_test)
    dataset_valid = CustomIMDBDataset(indices_matrix=indices_li_valid, y_binary=y_binary_valid)

    ind_li, label = dataset_train[0]

    #print(f"Type of first element returned by CustomIMDBDataset: {type(ind_li)}")
    #print(f"Shape of first element: {ind_li.shape}")
    #print(f"Type of items inside first element: {type(ind_li[0])}")
    #print(f"Type of second element returned by CustomIMDBDataset: {type(label)}")
    #print()

    ## Create DataLoader
    # This class allows to batch elements coming from a Dataset instance, and feed them
    # progressively to a Neural Network during its training.
    batch_size = 100
    data_loader_toy = DataLoader(dataset_toy, batch_size=batch_size, shuffle=True)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


    #for i, data in enumerate(data_loader_train):
    #    if i > 0:
    #        break
    #    ind_li_batch, label_batch = data
    #    print(f"Type of first element of Datalader first iteration output: {type(ind_li_batch)}")
    #    print("First element of the output:")
    #    print(ind_li_batch)
    #    print(f"Shape of first element of the output: {ind_li_batch.shape}")
    #    print(f"Type of second element of Dataloader first iteration output: {type(label_batch)}")
    #    print("Second element of the output:")
    #    print(label_batch)
    #    print(f"Shape of second element of the output: {label_batch.shape}")
    #print()



    ## Load GloVe embeddings
    glove_fn = os.path.join(os.getcwd(), 'glove.6B/glove.6B.' + str(word_embedding_size) + 'd.txt')
    emb_df = pd.read_table(glove_fn, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')

    glove_words_li = list(emb_df.index)
    glove_word_vectors = emb_df.values # numpy array of shape (vocab_size, emb_dim)

    ## Define the neural network architecture

    class GloveLSTMClassifier(nn.Module):
        """
        LSTM on top of frozen embeddings initialized with GloVe vectors.
        """
        def __init__(self, glove_embeddings, lstm_hidden_dim, keep_prob):
            super(GloveLSTMClassifier, self).__init__()
            _, self.emb_dim = glove_embeddings.shape
            glove_embeddings = torch.FloatTensor(glove_embeddings)
            self.embeddings = nn.Embedding.from_pretrained(embeddings=glove_embeddings)
            self.embeddings.weight.requires_grad_(False) #Embeddings are frozen
            self.lstm_hidden_dim = lstm_hidden_dim
            self.lstm = nn.LSTM(input_size=self.emb_dim,
                                hidden_size=lstm_hidden_dim,
                                batch_first=True)
            self.dropout = nn.Dropout(1-keep_prob)
            self.hidden2bin = nn.Linear(lstm_hidden_dim, 2)

        def forward(self, input):
            batch_size = input.shape[0]
            # Sequence of (batches of) indices turned into sequence of (batches of) embeddings
            # Input dimensions: (batch size (e.g. 100), input sequence length (e.g. 300))
            # Output dimensions: (batch size, input sequence length, embedding dimension (e.g. 100)
            emb_vect_seq = self.embeddings(input)
            # Sequence of (batches of) embeddings turned into one hidden state (and one cell state which isn't used)
            # As explained in this page:
            # https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
            # there is no need to explicitly loop over the word embeddings.
            # Since we only care about the hidden output for the last word of the input sequence
            # Input dimensions: (batch size, input sequence length, embedding dimension)
            # Output dimensions: (batch size, 1, LSTM hidden dimension (e.g. 64))
            _, hidden = self.lstm(emb_vect_seq)
            hidden = hidden[0]
            # Drop-out: Element randomly assigned to zero with probability p
            hidden = self.dropout(hidden)
            # Hidden state turned into logits
            # Input dimensions: (batch size, LSTM hidden dimension)
            # Output (batch size, 2)
            logits = self.hidden2bin(hidden.view(batch_size, self.lstm_hidden_dim))

            return logits


    classifier = GloveLSTMClassifier(glove_embeddings=glove_word_vectors,
                                     keep_prob=0.7,
                                     lstm_hidden_dim=64)

    ## Loss function definition and Optimization algorithm choice
    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.Adagrad(params=classifier.parameters(), lr=learning_rate)

    ## Training of the model

    # Sanity check: run the model on a single sample
    #classifier.zero_grad()
    #x = torch.from_numpy(np.int64(indices_li_train[0]))
    #y = [y_binary_train[0]]
    #logits = classifier(x.view(1, -1))
    #loss = loss_function(logits, torch.tensor(y))
    #loss.backward()
    #optimizer.step()

    # Real training
    n_epochs = 10

    training_time = 0
    for epoch in range(n_epochs):
        running_loss = 0
        start = time.time()
        for i, data_batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            ind_li_batch, label_batch = data_batch
            x = torch.from_numpy(np.int64(ind_li_batch))
            logits = classifier(x)
            loss = loss_function(logits, torch.tensor(label_batch))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        stop = time.time()
        training_epoch_time = stop - start
        training_time += training_epoch_time
        print()
        print("##############")
        print(f"Epoch {epoch}")
        print("##############")
        print()
        print(f"Loss: {running_loss}")
        print(f"Time to train epoch {epoch}: {training_epoch_time:.1f}")

        # Evaluate model on validation set
        correct = 0
        total = 0

        evaluation_time = 0
        with torch.no_grad():
            y_true = []
            y_pred = []
            y_prob = []

            for data_batch in data_loader_valid:
                ind_li_batch, label_batch = data_batch
                y_true.extend(list(label_batch.numpy()))
                x = torch.from_numpy(np.int64(ind_li_batch))
                start = time.time()
                logits = classifier(x)
                stop = time.time()
                evaluation_time += stop - start
                # In order to use sklearn, we put tensors into lists and turn logits into proba
                unnormalized_prob = np.exp(logits.numpy())
                denominator = unnormalized_prob[:, 0] + unnormalized_prob[:, 1]
                proba_positive = list(unnormalized_prob[:, 1]/denominator)
                y_prob.extend(proba_positive)

                _, predicted = torch.max(logits.data, 1)
                y_pred.extend(list(predicted.numpy()))
                total += label_batch.size(0)
                correct += (predicted == label_batch).sum().item()

        print(f"Accuracy on the test set: {100 * correct/total}")
        print_evaluation_scores(y_true, y_pred, y_prob)
        print(f"Evaluation time for epoch {epoch}: {evaluation_time}")


    print()
    print(f"Total training time on {n_epochs} epochs: {training_time:.1f} seconds")


if __name__ == "__main__":
    main()




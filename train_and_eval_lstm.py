import os
import numpy as np
import pickle
import pandas as pd
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import time
from datetime import date
import json


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
    learning_rate = 1
    optimizer = torch.optim.Adadelta(params=classifier.parameters(), lr=learning_rate)
    optimizer_name = "Adadelta"
    #optimizer = torch.optim.Adagrad(params=classifier.parameters(), lr=learning_rate)

    ## Training of the model

    # Sanity check: run the model on a single sample
    #classifier.zero_grad()
    #x = torch.from_numpy(np.int64(indices_li_train[0]))
    #y = [y_binary_train[0]]
    #logits = classifier(x.view(1, -1))
    #loss = loss_function(logits, torch.tensor(y))
    #loss.backward()
    #optimizer.step()

    # For TensorBoard
    writer = SummaryWriter()

    # Real training
    training_time = 0
    # First train the model for a fixed number of epoch with the same learning rate
    n_epochs = 12
    acc_max = 0
    acc_max_epoch = 1
    acc_seq = []
    patience = 3
    learning_rate_factor = 10
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")

    model_fn = os.path.join(os.getcwd(), 'Saved_models/lstm_model_state_dict_' + d1 + '.pt')
    model_description_fn = os.path.join(os.getcwd(), 'Saved_models/lstm_model_description_' + d1 + '.json')
    training_description = f"Train model for {n_epochs} epochs with fixed learning rate {learning_rate}." \
                           f" Then train model with early stopping (patience = {patience})" \
                           f" and devide learning rate by {learning_rate_factor} when accuracy decreases." \
                           f" Save model at epoch having highest accuracy on validation set."

    def update_max_acc_and_save_best_model(accuracy,
                                           epoch,
                                           acc_max_old,
                                           acc_max_epoch_old,
                                           save_model=True):
        """
        Update maximum accuracy (and epoch where this max occured) and save model when a max is reached
        :param accuracy:
        :param epoch:
        :param acc_max_old:
        :param acc_max_epoch_old:
        :param model_description_fn:
        :param model_fn:
        :param save_model: bool indicates if one wants to save model (when a max accuracy is reached) or not
        :return:
        """
        if accuracy > acc_max_old:
            acc_max_old = accuracy
            acc_max_epoch_old = epoch
            with open(model_description_fn, encoding='utf-8', mode='w+') as f:
                json.dump({"epoch": epoch,
                           "optimizer_name": optimizer_name,
                           "accuracy": accuracy,
                           "loss_on_training_set": training_loss,
                           "training_description": training_description
                           }, f, indent=4)

            if save_model:
                torch.save({"epoch": epoch,
                            "optimizer_name": optimizer_name,
                            "accuracy": accuracy,
                            "loss_on_training_set": training_loss,
                            "training_description": training_description,
                            "model_state_dict": classifier.state_dict()}, model_fn)
        return acc_max_old, acc_max_epoch_old

    # Train for a fixed number of epoch
    for epoch in range(1, n_epochs + 1):
        print()
        print("##############")
        print(f"Epoch {epoch}")
        print("##############")
        training_loss = 0
        start = time.time()
        for i, data_batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            ind_li_batch, label_batch = data_batch
            x = torch.from_numpy(np.int64(ind_li_batch))
            logits = classifier(x)
            loss = loss_function(logits, torch.tensor(label_batch))
            loss.backward()
            # For Gradient clipping (in case NaN appear)
            #torch.nn.utils.clip_grad_norm_(classifier.parameters(), 0.5)
            optimizer.step()
            training_loss += loss.item()

        stop = time.time()
        training_epoch_time = stop - start
        training_time += training_epoch_time

        writer.add_scalar("Loss/train_" + optimizer_name, training_loss, epoch)  # for tensorboard

        print()
        print(f"Loss on training set: {training_loss}")
        print(f"Time to train epoch {epoch}: {training_epoch_time:.1f}")

        # Evaluate model on validation set
        evaluation_time = 0
        validation_loss = 0

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
                loss = loss_function(logits, torch.tensor(label_batch))
                validation_loss += loss
                evaluation_time += stop - start
                # In order to use sklearn, we put tensors into lists and turn logits into proba
                unnormalized_prob = np.exp(logits.numpy())
                denominator = unnormalized_prob[:, 0] + unnormalized_prob[:, 1]
                proba_positive = list(unnormalized_prob[:, 1]/denominator)
                y_prob.extend(proba_positive)

                _, predicted = torch.max(logits.data, 1)
                y_pred.extend(list(predicted.numpy()))

        writer.add_scalar("Loss/valid_" + optimizer_name, validation_loss, epoch)  # for tensorboard

        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        writer.add_scalar("Accuracy/valid_" + optimizer_name, accuracy, epoch)  # for tensorboard

        acc_seq.append(accuracy)
        acc_max, acc_max_epoch = update_max_acc_and_save_best_model(accuracy,
                                                                    epoch,
                                                                    acc_max,
                                                                    acc_max_epoch)
        print()
        print_evaluation_scores(y_true, y_pred, y_prob)
        print()
        print(f"Loss on validation set: {validation_loss}")
        print(f"Evaluation time for epoch {epoch}: {evaluation_time:.1f} seconds")

    # Train model a few more epochs using early stopping
    # and decreasing learning rate by a factor if model doesn't improve
    early_stopping = False # set it to True to have only a fixed number of epoch (first part of the training implemented above)
    epoch = n_epochs
    max_epoch = 40

    def has_to_stop(acc_seq, patience):
        """
        Helper function to determine if training should stop because accuracy didn't improve over the
        last epochs.
        :param acc_seq: list containing accuracy on validation for each epoch
        :param patience: int indicating patience level before early stopping
        :return:
        """
        if patience >= len(acc_seq):
            return False
        for i in range(patience):
            if acc_seq[-(i+1)] > acc_seq[-(i+2)]:
                return False
        return True

    decrease_lr_li = [False for i in range(n_epochs)]

    def has_to_decrease_lr(acc_seq, decrease_lr_li, learning_rate):
        """
        Helper function to determine if learning rate should be divided by a factor.
        :param acc_seq:
        :param decrease_lr_li:
        :param learning_rate:
        :return:
        """
        if len(decrease_lr_li) < 3:
            return False
        if acc_seq[-1] < acc_seq[-2]:
            if decrease_lr_li[-1] or decrease_lr_li[-2] or learning_rate <= 0.0001:
                return False
            else:
                return True

    while not early_stopping:
        epoch += 1

        print()
        print("##############")
        print(f"Epoch {epoch}")
        print("##############")

        early_stopping = has_to_stop(acc_seq, patience)

        if early_stopping:
            print()
            print(f"Early stopping at epoch {epoch}")
            print()
            break
        if epoch >= max_epoch:
            print()
            print(f"Maximum number of epochs ({max_epoch}) reached. Stopping training.")
            print()
            break
        decrease_lr = has_to_decrease_lr(acc_seq, decrease_lr_li, learning_rate)
        decrease_lr_li.append(decrease_lr)
        if decrease_lr:
            print()
            print(f"Change or learning rate")
            print(f"Previous learning rate: {learning_rate}")
            learning_rate = learning_rate / learning_rate_factor
            print(f"New learning rate: {learning_rate}")
        training_loss = 0
        start = time.time()
        for i, data_batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            ind_li_batch, label_batch = data_batch
            x = torch.from_numpy(np.int64(ind_li_batch))
            logits = classifier(x)
            loss = loss_function(logits, torch.tensor(label_batch))
            loss.backward()
            # For Gradient clipping (in case NaN appear)
            #torch.nn.utils.clip_grad_norm_(classifier.parameters(), 0.5)
            optimizer.step()
            training_loss += loss.item()

        stop = time.time()
        training_epoch_time = stop - start
        training_time += training_epoch_time

        writer.add_scalar("Loss/train_" + optimizer_name, training_loss, epoch)  # for tensorboard

        print()
        print(f"Loss on training set: {training_loss}")
        print(f"Time to train epoch {epoch}: {training_epoch_time:.1f}")

        # Evaluate model on validation set
        evaluation_time = 0
        validation_loss = 0
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
                loss = loss_function(logits, torch.tensor(label_batch))
                validation_loss += loss
                evaluation_time += stop - start
                # In order to use sklearn, we put tensors into lists and turn logits into proba
                unnormalized_prob = np.exp(logits.numpy())
                denominator = unnormalized_prob[:, 0] + unnormalized_prob[:, 1]
                proba_positive = list(unnormalized_prob[:, 1] / denominator)
                y_prob.extend(proba_positive)

                _, predicted = torch.max(logits.data, 1)
                y_pred.extend(list(predicted.numpy()))

        writer.add_scalar("Loss/valid_" + optimizer_name, validation_loss, epoch)  # for tensorboard

        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        writer.add_scalar("Accuracy/valid_" + optimizer_name, accuracy, epoch)  # for tensorboard

        acc_seq.append(accuracy)
        acc_max, acc_max_epoch = update_max_acc_and_save_best_model(accuracy,
                                                                    epoch,
                                                                    acc_max,
                                                                    acc_max_epoch)

        print()
        print_evaluation_scores(y_true, y_pred, y_prob)
        print()
        print(f"Evaluation time for epoch {epoch}: {evaluation_time:.1f} seconds")

    print()
    print(f"Maximum accuracy on validation set was reached at epoch {acc_max_epoch} with value: {acc_max}")

    print()
    print(f"Total training time on {epoch} epochs: {training_time:.1f} seconds")

    writer.flush()# for tensorboard
    writer.close()

if __name__ == "__main__":
    main()

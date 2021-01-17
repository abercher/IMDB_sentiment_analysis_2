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
import sys
import warnings


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


class LearningRateUpdater:
    """
    Implement a cyclical update of learning rate inspired by this paper:
    https://arxiv.org/abs/1506.01186
    It can also be used for annealing learning rate.
    """
    update_direction = "forward"
    lr_position = 0
    update_history = []

    def __init__(self, lr_li, update_type="annealing", update_history=None):
        """
        Initialization
        :param lr_li: list of float indicating learning rate values
        :param update_type: type of learning rate update protocol. Possible values are:
        "annealing": when the lr updater hits the last entry of lr_li, it stops modifying the learning rate
        "loop": when the lr updater hits the last entry of lr_li, it goes to the first entry at the next update
        "cyclical": the lr updater goes back and forth the lr_li
        :param update_history: list of boolean indicating for each epoch if the lr was updated
        """
        if not isinstance(lr_li, list) or len(lr_li)==0:
            sys.exit("Error: incorrect lr_li type or length")
        lr_good_type_li = [isinstance(lr, float) or isinstance(lr, int) for lr in lr_li]
        if not all(lr_good_type_li):
            sys.exit("Error: incorrect learning rate type inside lr_li")
        self.lr_li = lr_li
        # indicates if we want to go back to the first learning rate when the last one is reached
        # or if we change the learning rate going backward with the list
        if update_type not in {"annealing", "loop", "cyclical"}:
            sys.exit("Error: incorrect update_type value")
        self.update_type = update_type
        if update_history is None:
            self.update_history = []
        else:
            self.update_history = update_history

    def update_lr(self, acc_seq):
        if len(self.update_history) < 3:
            self.update_history.append(False)
            return self.lr_li[0]
        # lr not modified if it was modified in one of the two previous epoch
        elif self.update_history[-1] or self.update_history[-2]:
            self.update_history.append(False)
            return self.lr_li[self.lr_position]
        # lr not modified if accuracy on validation set improved in the last epoch
        elif acc_seq[-1] >= acc_seq[-2]:
            self.update_history.append(False)
            return self.lr_li[self.lr_position]
        elif self.update_direction == "forward":
            self.update_history.append(True)
            if self.lr_position == len(self.lr_li) - 1:
                if self.update_type == "loop":
                    self.lr_position = 0
                elif self.update_type == "cyclical":
                    self.lr_position = max(0, len(self.lr_li) - 2)
                    self.update_direction = "backward"
                elif self.update_type == "annealing":
                    # Nothing to do in this case
                    pass
                else:
                    # If we arrive here, there is a problem...
                    warnings.warn("A problem occurred in the update of the learning rate.")
            else:
                self.lr_position += 1
            return self.lr_li[self.lr_position]
        elif self.update_direction == "backward":
            self.update_history.append(True)
            if self.lr_position == 0:
                self.update_direction = "forward"
                self.lr_position = 1
            else:
                self.lr_position -= 1
            return self.lr_li[self.lr_position]
        else:
            warnings.warn("A problem occurred in the update of the learning rate.")
            return self.lr_li[self.lr_position]


def train_and_evaluate_pytorch_model(classifier,
                                     data_loader_train,
                                     data_loader_valid,
                                     optimizer,
                                     optimizer_name,
                                     model_name,
                                     learning_rate=0.1,
                                     max_epoch=40,
                                     use_early_stopping=False,
                                     use_lr_updater=True,
                                     lr_schedule_type="annealing",
                                     n_fixed_epochs=10,
                                     patience=3,
                                     lr_li=None
                                     ):

    if not lr_li:
        lr_li = [1, 0.1, 0.01]

    loss_function = nn.CrossEntropyLoss()

    today = date.today()
    d1 = today.strftime("%d_%m_%Y")

    model_fn = os.path.join(os.getcwd(), 'Saved_models/lstm_model_state_dict_' + d1 + '.pt')
    model_description_fn = os.path.join(os.getcwd(), 'Saved_models/lstm_model_description_' + d1 + '.json')
    training_description = f"Train model for {n_fixed_epochs} epochs with fixed learning rate {learning_rate}." \
                           f" Then train model with early stopping (patience = {patience})" \
                           f" and {lr_schedule_type} learning rate schedule when accuracy decreases." \
                           f" Save model at epoch having highest accuracy on validation set."

    training_time = 0

    acc_max = 0
    acc_max_epoch = 1
    acc_seq = []


    # For TensorBoard
    writer = SummaryWriter()

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
    for epoch in range(1, n_fixed_epochs + 1):
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
            loss = loss_function(logits, label_batch)
            loss.backward()
            # For Gradient clipping (in case NaN appear)
            #torch.nn.utils.clip_grad_norm_(classifier.parameters(), 0.5)
            optimizer.step()
            training_loss += loss.item()

        stop = time.time()
        training_epoch_time = stop - start
        training_time += training_epoch_time

        writer.add_scalar(model_name + "/" + "Learning_rate_" + optimizer_name, learning_rate, epoch)  # for tensorboard

        writer.add_scalar(model_name + "/" + "Loss/train_" + optimizer_name + "_" + lr_schedule_type, training_loss, epoch)  # for tensorboard

        print()
        print(f"Loss on training set: {training_loss}")
        print(f"Time to train epoch {epoch}: {training_epoch_time:.1f}")

        writer.add_scalar(model_name + "/" + "Epoch_training_time_" + optimizer_name + "_" + lr_schedule_type, training_epoch_time, epoch)  # for tensorboard

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
                loss = loss_function(logits, label_batch)
                validation_loss += loss
                evaluation_time += stop - start
                # In order to use sklearn, we put tensors into lists and turn logits into proba
                unnormalized_prob = np.exp(logits.numpy())
                denominator = unnormalized_prob[:, 0] + unnormalized_prob[:, 1]
                proba_positive = list(unnormalized_prob[:, 1]/denominator)
                y_prob.extend(proba_positive)

                _, predicted = torch.max(logits.data, 1)
                y_pred.extend(list(predicted.numpy()))

        writer.add_scalar(model_name + "/" + "Loss/valid_" + optimizer_name + "_" + lr_schedule_type, validation_loss, epoch)  # for tensorboard

        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        writer.add_scalar(model_name + "/" + "Accuracy/valid_" + optimizer_name + "_" + lr_schedule_type, accuracy, epoch)  # for tensorboard

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

    print()
    print('###################################')
    print('Finished first part of the training')
    print('###################################')

    # Train model a few more epochs using early stopping
    # and modifying learning rate if model doesn't improve on validation set
    epoch = n_fixed_epochs

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

    update_history = [False for _ in range(n_fixed_epochs)]

    if lr_schedule_type != "fixed_learning_rate":
        if learning_rate != lr_li[0]:
            warnings.warn(f"Initial learning rate is {learning_rate} but first value of lr_updater is {lr_li[0]}")
        lr_updater = LearningRateUpdater(lr_li=lr_li,
                                         update_type=lr_schedule_type,
                                         update_history=update_history)

    early_stopping = False

    while not early_stopping:
        epoch += 1

        # Early stopping
        early_stopping_ = has_to_stop(acc_seq, patience)
        if early_stopping_ and use_early_stopping:
            print()
            print(f"Early stopping at epoch {epoch}")
            print()
            break
        if epoch > max_epoch:
            print()
            print(f"Maximum number of epochs ({max_epoch}) reached. Stopping training.")
            print()
            break

        print()
        print("##############")
        print(f"Epoch {epoch}")
        print("##############")

        # Learning rate update
        if use_lr_updater:
            old_lr = learning_rate
            learning_rate = lr_updater.update_lr(acc_seq)
            if old_lr != learning_rate:
                print()
                print(f"Change of learning rate")
                print(f"Previous learning rate: {old_lr}")
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate
                print(f"New learning rate: {learning_rate}")

        writer.add_scalar(model_name + "/" + "Learning_rate_" + optimizer_name, learning_rate, epoch)  # for tensorboard

        training_loss = 0
        start = time.time()
        for i, data_batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            ind_li_batch, label_batch = data_batch
            x = torch.from_numpy(np.int64(ind_li_batch))
            logits = classifier(x)
            loss = loss_function(logits, label_batch)
            loss.backward()
            # For Gradient clipping (in case NaN appear)
            #torch.nn.utils.clip_grad_norm_(classifier.parameters(), 0.5)
            optimizer.step()
            training_loss += loss.item()

        stop = time.time()
        training_epoch_time = stop - start
        training_time += training_epoch_time

        writer.add_scalar(model_name + "/" + "Loss/train_" + optimizer_name + "_" + lr_schedule_type, training_loss, epoch)  # for tensorboard

        print()
        print(f"Loss on training set: {training_loss}")
        print(f"Time to train epoch {epoch}: {training_epoch_time:.1f}")

        writer.add_scalar(model_name + "/" + "Epoch_training_time_" + optimizer_name + "_" + lr_schedule_type, training_epoch_time, epoch)  # for tensorboard

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
                loss = loss_function(logits, label_batch)
                validation_loss += loss
                evaluation_time += stop - start
                # In order to use sklearn, we put tensors into lists and turn logits into proba
                unnormalized_prob = np.exp(logits.numpy())
                denominator = unnormalized_prob[:, 0] + unnormalized_prob[:, 1]
                proba_positive = list(unnormalized_prob[:, 1] / denominator)
                y_prob.extend(proba_positive)

                _, predicted = torch.max(logits.data, 1)
                y_pred.extend(list(predicted.numpy()))

        writer.add_scalar(model_name + "/" + "Loss/valid_" + optimizer_name + "_" + lr_schedule_type, validation_loss, epoch)  # for tensorboard

        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        writer.add_scalar(model_name + "/" + "Accuracy/valid_" + optimizer_name + "_" + lr_schedule_type, accuracy, epoch)  # for tensorboard

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
    print(f"Total training time on {epoch-1} epochs: {training_time:.1f} seconds")

    writer.flush()# for tensorboard
    writer.close()


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
        __name__ = "GloveLSTMBasic"

        def __init__(self, glove_embeddings, lstm_hidden_dim, keep_prob):
            super(GloveLSTMClassifier, self).__init__()
            _, self.emb_dim = glove_embeddings.shape
            glove_embeddings = torch.FloatTensor(glove_embeddings)
            self.embeddings = nn.Embedding.from_pretrained(embeddings=glove_embeddings)
            self.embeddings.weight.requires_grad_(False) #Embeddings are frozen
            self.lstm_hidden_dim = lstm_hidden_dim
            self.lstm = nn.LSTM(input_size=self.emb_dim,
                                hidden_size=lstm_hidden_dim,
                                batch_first=True,
                                bidirectional=False)
            self.dropout = nn.Dropout(1-keep_prob)
            self.hidden2bin = nn.Linear(lstm_hidden_dim, 2)# For simple LSTM
            #self.hidden2bin = nn.Linear(lstm_hidden_dim*2, 2)# For Bi-LSTM

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
            logits = self.hidden2bin(hidden.view(batch_size, self.lstm_hidden_dim))# For simple LSTM
            #logits = self.hidden2bin(hidden.view(batch_size, self.lstm_hidden_dim*2))# For Bi-LSTM

            return logits


    classifier = GloveLSTMClassifier(glove_embeddings=glove_word_vectors,
                                     keep_prob=0.7,
                                     lstm_hidden_dim=64)

    ## Loss function definition and Optimization algorithm choice
    learning_rate = 1
    optimizer = torch.optim.Adadelta(params=classifier.parameters(), lr=learning_rate)
    optimizer_name = "Adadelta"

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

    use_early_stopping = False# decide if early stopping should be used or not

    lr_schedule_type = "annealing"
    if lr_schedule_type == "fixed_learning_rate":
        use_lr_updater = False
    else:
        use_lr_updater = True

    lr_li = [1, 0.1, 0.01]

    n_fixed_epochs = 12# Minimum number of epochs. Learning rate is fixed during these epochs.

    patience = 3

    max_epoch = 40

    model_name = classifier.__name__

    train_and_evaluate_pytorch_model(classifier=classifier,
                                     data_loader_train=data_loader_train,
                                     data_loader_valid=data_loader_valid,
                                     optimizer=optimizer,
                                     optimizer_name=optimizer_name,
                                     model_name=model_name,
                                     learning_rate=learning_rate,
                                     max_epoch=max_epoch,
                                     use_early_stopping=use_early_stopping,
                                     use_lr_updater=use_lr_updater,
                                     lr_schedule_type=lr_schedule_type,
                                     n_fixed_epochs=n_fixed_epochs,
                                     patience=patience,
                                     lr_li=lr_li)


if __name__ == "__main__":
    main()

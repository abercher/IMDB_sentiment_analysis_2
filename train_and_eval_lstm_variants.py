"""
Similar to train_and_eval_lstm.py but with variants of the original model
"""
import os
import numpy as np
import pickle
import pandas as pd
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_and_eval_lstm import train_and_evaluate_pytorch_model



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

    ## Create DataLoader
    # This class allows to batch elements coming from a Dataset instance, and feed them
    # progressively to a Neural Network during its training.
    batch_size = 100
    data_loader_toy = DataLoader(dataset_toy, batch_size=batch_size, shuffle=True)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    ## Load GloVe embeddings
    glove_fn = os.path.join(os.getcwd(), 'glove.6B/glove.6B.' + str(word_embedding_size) + 'd.txt')
    emb_df = pd.read_table(glove_fn, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')

    glove_words_li = list(emb_df.index)
    glove_word_vectors = emb_df.values  # numpy array of shape (vocab_size, emb_dim)

    ###############################################################################################
    ## First model: Bi-LSTM + 1D Maxpooling
    ###############################################################################################

    ## Define the neural network architecture

    class GloveBiLSTMClassifier(nn.Module):
        """
        Bi-LSTM on top of frozen embeddings initialized with GloVe vectors, followed by 1D max pooling
        on all the outputs of the Bi-LSTM layer.
        """
        __name__ = "GloVeBiLSTM"

        def __init__(self, glove_embeddings, lstm_hidden_dim, keep_prob, seq_length=300):
            super(GloveBiLSTMClassifier, self).__init__()
            _, self.emb_dim = glove_embeddings.shape
            glove_embeddings = torch.FloatTensor(glove_embeddings)
            self.embeddings = nn.Embedding.from_pretrained(embeddings=glove_embeddings)
            self.embeddings.weight.requires_grad_(False)  # Embeddings are frozen
            self.lstm_hidden_dim = lstm_hidden_dim
            self.lstm = nn.LSTM(input_size=self.emb_dim,
                                hidden_size=lstm_hidden_dim,
                                batch_first=True,
                                bidirectional=True)# For Bi-LSTM
            self.maxpool = nn.MaxPool1d(seq_length)
            self.dropout = nn.Dropout(1 - keep_prob)
            #self.hidden2bin = nn.Linear(lstm_hidden_dim, 2)# For simple LSTM
            self.hidden2bin = nn.Linear(lstm_hidden_dim * 2, 2)  # For Bi-LSTM

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
            # Output dimensions: (batch size, sequence length (e.g. 300), LSTM hidden dimension (e.g. 64) * number directions)
            output, _ = self.lstm(emb_vect_seq)
            # Swap axis
            # Input dimensions: (batch size, sequence length (e.g. 300), LSTM hidden dimension (e.g. 64) * number directions)
            # Output dimensions: (batch size, LSTM hidden dimension (e.g. 64) * number directions, sequence length (e.g. 300))
            output = output.permute(0, 2, 1)
            # Sequence of (batches of) lstm outputs turned into one hidden state using 1D Max pooling
            # Input dimensions: (batch size, sequence length, LSTM hidden dimension)
            # Output dimensions: (batch size, LSTM hidden dimension)
            hidden = self.maxpool(output)
            # Drop-out: Element randomly assigned to zero with probability p
            hidden = self.dropout(hidden)
            # Hidden state turned into logits
            # Input dimensions: (batch size, LSTM hidden dimension)
            # Output (batch size, 2)
            #logits = self.hidden2bin(hidden.view(batch_size, self.lstm_hidden_dim))# For simple LSTM
            logits = self.hidden2bin(hidden.view(batch_size, self.lstm_hidden_dim * 2))  # For Bi-LSTM

            return logits

    classifier = GloveBiLSTMClassifier(glove_embeddings=glove_word_vectors,
                                     keep_prob=0.7,
                                     lstm_hidden_dim=64)

    ## Loss function definition and Optimization algorithm choice
    learning_rate = 1
    optimizer = torch.optim.Adadelta(params=classifier.parameters(), lr=learning_rate)
    optimizer_name = "Adadelta"

    ## Training of the model

    use_early_stopping = False  # decide if early stopping should be used or not

    lr_schedule_type = "cyclical"
    if lr_schedule_type == "fixed_learning_rate":
        use_lr_updater = False
    else:
        use_lr_updater = True

    lr_li = [1, 0.1, 0.01]

    n_fixed_epochs = 12  # Minimum number of epochs. Learning rate is fixed during these epochs.

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

    ###############################################################################################
    ## Second model: Custom Embeddings
    ###############################################################################################

    ## Define the neural network architecture

    class CustomEmbeddingsLSTMClassifier(nn.Module):
        """
        LSTM on top of custom embeddings.
        """

        __name__ = "CustomEmbeddingsLSTM"

        def __init__(self, vocab_size, emb_dim, lstm_hidden_dim, keep_prob):
            super(CustomEmbeddingsLSTMClassifier, self).__init__()
            self.emb_dim = emb_dim
            self.embeddings = nn.Embedding(vocab_size, emb_dim)
            self.embeddings.weight.requires_grad_(True)  # Embeddings are frozen
            self.lstm_hidden_dim = lstm_hidden_dim
            self.lstm = nn.LSTM(input_size=self.emb_dim,
                                hidden_size=lstm_hidden_dim,
                                batch_first=True,
                                bidirectional=False)
            self.dropout = nn.Dropout(1 - keep_prob)
            self.hidden2bin = nn.Linear(lstm_hidden_dim, 2)# For simple LSTM
            #self.hidden2bin = nn.Linear(lstm_hidden_dim * 2, 2)  # For Bi-LSTM

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
            #logits = self.hidden2bin(hidden.view(batch_size, self.lstm_hidden_dim * 2))  # For Bi-LSTM

            return logits

    vocab_size, emb_dim_glove = glove_word_vectors.shape
    emb_dim = 50

    classifier = CustomEmbeddingsLSTMClassifier(vocab_size=vocab_size,
                                                emb_dim=emb_dim,
                                                keep_prob=0.7,
                                                lstm_hidden_dim=64)

    ## Loss function definition and Optimization algorithm choice
    learning_rate = 1
    optimizer = torch.optim.Adadelta(params=classifier.parameters(), lr=learning_rate)
    optimizer_name = "Adadelta"

    ## Training of the model

    use_early_stopping = False  # decide if early stopping should be used or not

    lr_schedule_type = "cyclical"
    if lr_schedule_type == "fixed_learning_rate":
        use_lr_updater = False
    else:
        use_lr_updater = True

    lr_li = [1, 0.1, 0.01]

    n_fixed_epochs = 12  # Minimum number of epochs. Learning rate is fixed during these epochs.

    patience = 3

    max_epoch = 50

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

    ###############################################################################################
    ## Third model: 1D Max Pooling
    ###############################################################################################

    ## Define the neural network architecture

    class GloveLSTMMaxPoolClassifier(nn.Module):
        """
        LSTM on top of frozen embeddings initialized with GloVe vectors, followed by 1D max pooling
        over the sequence of all the outputs of the LSTM layer.
        """

        __name__ = "GloveLSTMMaxPool"

        def __init__(self, glove_embeddings, lstm_hidden_dim, keep_prob, seq_length=300):
            super(GloveLSTMMaxPoolClassifier, self).__init__()
            _, self.emb_dim = glove_embeddings.shape
            glove_embeddings = torch.FloatTensor(glove_embeddings)
            self.embeddings = nn.Embedding.from_pretrained(embeddings=glove_embeddings)
            self.embeddings.weight.requires_grad_(False)  # Embeddings are frozen
            self.lstm_hidden_dim = lstm_hidden_dim
            self.lstm = nn.LSTM(input_size=self.emb_dim,
                                hidden_size=lstm_hidden_dim,
                                batch_first=True,
                                bidirectional=False)  # For Bi-LSTM
            self.maxpool = nn.MaxPool1d(seq_length)
            self.dropout = nn.Dropout(1 - keep_prob)
            self.hidden2bin = nn.Linear(lstm_hidden_dim, 2)# For simple LSTM
            #self.hidden2bin = nn.Linear(lstm_hidden_dim * 2, 2)  # For Bi-LSTM

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
            # Output dimensions: (batch size, sequence length (e.g. 300), LSTM hidden dimension (e.g. 64) * number directions)
            output, _ = self.lstm(emb_vect_seq)
            # Swap axis
            # Input dimensions: (batch size, sequence length (e.g. 300), LSTM hidden dimension (e.g. 64) * number directions)
            # Output dimensions: (batch size, LSTM hidden dimension (e.g. 64) * number directions, sequence length (e.g. 300))
            output = output.permute(0, 2, 1)
            # Sequence of (batches of) lstm outputs turned into one hidden state using 1D Max pooling
            # Input dimensions: (batch size, sequence length, LSTM hidden dimension)
            # Output dimensions: (batch size, LSTM hidden dimension)
            hidden = self.maxpool(output)
            # Drop-out: Element randomly assigned to zero with probability p
            hidden = self.dropout(hidden)
            # Hidden state turned into logits
            # Input dimensions: (batch size, LSTM hidden dimension)
            # Output (batch size, 2)
            logits = self.hidden2bin(hidden.view(batch_size, self.lstm_hidden_dim))# For simple LSTM
            #logits = self.hidden2bin(hidden.view(batch_size, self.lstm_hidden_dim * 2))  # For Bi-LSTM

            return logits

    classifier = GloveLSTMMaxPoolClassifier(glove_embeddings=glove_word_vectors,
                                       keep_prob=0.7,
                                       lstm_hidden_dim=64)

    ## Loss function definition and Optimization algorithm choice
    learning_rate = 1
    optimizer = torch.optim.Adadelta(params=classifier.parameters(), lr=learning_rate)
    optimizer_name = "Adadelta"

    ## Training of the model

    use_early_stopping = False  # decide if early stopping should be used or not

    lr_schedule_type = "cyclical"
    if lr_schedule_type == "fixed_learning_rate":
        use_lr_updater = False
    else:
        use_lr_updater = True

    lr_li = [1, 0.1, 0.01]

    n_fixed_epochs = 12  # Minimum number of epochs. Learning rate is fixed during these epochs.

    patience = 3

    max_epoch = 40

    model_name = classifier.__name__

    #train_and_evaluate_pytorch_model(classifier=classifier,
    #                                 data_loader_train=data_loader_train,
    #                                 data_loader_valid=data_loader_valid,
    #                                 optimizer=optimizer,
    #                                 optimizer_name=optimizer_name,
    #                                 model_name=model_name,
    #                                 learning_rate=learning_rate,
    #                                 max_epoch=max_epoch,
    #                                 use_early_stopping=use_early_stopping,
    #                                 use_lr_updater=use_lr_updater,
    #                                 lr_schedule_type=lr_schedule_type,
    #                                 n_fixed_epochs=n_fixed_epochs,
    #                                 patience=patience,
    #                                 lr_li=lr_li)




if __name__ == "__main__":
    main()

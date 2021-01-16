"""
Train and evaluate a model using pretrained BERT embeddings.
"""
import os
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertTokenizer, BertModel, BertConfig
import json
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import time
from datetime import date
import warnings
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

from train_and_eval_lstm import print_evaluation_scores, LearningRateUpdater


def train_and_evaluate_transformer_model(classifier,
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

    classifier.to(device)

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
            ids = data_batch['ids'].to(device, dtype=torch.long)
            mask = data_batch['mask'].to(device, dtype=torch.long)
            token_type_ids = data_batch['token_type_ids'].to(device, dtype=torch.long)
            targets = data_batch['targets']
            logits = classifier(ids, mask, token_type_ids)
            loss = loss_function(logits, targets)
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
                optimizer.zero_grad()
                ids = data_batch['ids'].to(device, dtype=torch.long)
                mask = data_batch['mask'].to(device, dtype=torch.long)
                token_type_ids = data_batch['token_type_ids'].to(device, dtype=torch.long)
                targets = data_batch['targets']
                y_true.extend(list(targets.numpy()))
                start = time.time()
                logits = classifier(ids, mask, token_type_ids)
                stop = time.time()
                loss = loss_function(logits, targets)
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
            ids = data_batch['ids'].to(device, dtype=torch.long)
            mask = data_batch['mask'].to(device, dtype=torch.long)
            token_type_ids = data_batch['token_type_ids'].to(device, dtype=torch.long)
            targets = data_batch['targets']
            logits = classifier(ids, mask, token_type_ids)
            loss = loss_function(logits, targets)
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
                optimizer.zero_grad()
                ids = data_batch['ids'].to(device, dtype=torch.long)
                mask = data_batch['mask'].to(device, dtype=torch.long)
                token_type_ids = data_batch['token_type_ids'].to(device, dtype=torch.long)
                targets = data_batch['targets']
                y_true.extend(list(targets.numpy()))
                start = time.time()
                logits = classifier(ids, mask, token_type_ids)
                stop = time.time()
                loss = loss_function(logits, targets)
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
    ## Load (cleaned) reviews
    clean_text_train_fn = os.path.join(os.getcwd(), "Transformed_data/clean_text_train.csv")
    df_clean_text_train = pd.read_csv(clean_text_train_fn)
    clean_text_valid_fn = os.path.join(os.getcwd(), "Transformed_data/clean_text_valid.csv")
    df_clean_text_valid = pd.read_csv(clean_text_valid_fn)
    clean_text_test_fn = os.path.join(os.getcwd(), "Transformed_data/clean_text_test.csv")
    df_clean_text_test = pd.read_csv(clean_text_test_fn)


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

    ## Using pretrained Tokenizer (whatever "training" means for a tokenizer...)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ## Create custom Dataset
    # This class and its custom subclasses allow to loop over the data during training.
    # Whatever form the data is (in one numpy array or spread accross different files),
    # this class puts it in a uniform format which can then be fed to the model (via the Dataloader).
    # It can also be used to add some preprocessing steps to the data
    # The current implementation is basic. For bigger data, one can read the data progressively
    # in the __getitem__ method. But this is not needed here.
    class IMDBBertDataset(Dataset):
        def __init__(self, cleaned_reviews, y_binary, tokenizer, max_len):
            self.tokenizer = tokenizer
            self.cleaned_reviews = cleaned_reviews
            self.y_binary = y_binary
            self.max_len = max_len

        def __len__(self):
            return len(self.y_binary)

        def __getitem__(self, index):
            text = str(self.cleaned_reviews[index])
            text = " ".join(text.split())
            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_token_type_ids=True,
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            token_type_ids = inputs["token_type_ids"]

            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': self.y_binary[index]
            }

    n_toy = 500
    max_len = 300
    dataset_toy = IMDBBertDataset(cleaned_reviews=df_clean_text_train["clean_text"][:n_toy],
                                  y_binary=y_binary_train[:n_toy],
                                  tokenizer=tokenizer,
                                  max_len=max_len)
    dataset_train = IMDBBertDataset(cleaned_reviews=df_clean_text_train["clean_text"],
                                    y_binary=y_binary_train,
                                    tokenizer=tokenizer,
                                    max_len=max_len)
    dataset_valid = IMDBBertDataset(cleaned_reviews=df_clean_text_valid["clean_text"],
                                    y_binary=y_binary_valid,
                                    tokenizer=tokenizer,
                                    max_len=max_len)
    dataset_test = IMDBBertDataset(cleaned_reviews=df_clean_text_test["clean_text"],
                                   y_binary=y_binary_test,
                                   tokenizer=tokenizer,
                                   max_len=max_len)

    ids, mask, token_type_ids, target = dataset_train[0]



    ## Create DataLoader
    # This class allows to batch elements coming from a Dataset instance, and feed them
    # progressively to a Neural Network during its training.
    batch_size = 10
    data_loader_toy = DataLoader(dataset_toy, batch_size=batch_size, shuffle=True)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    ## Define the neural network architecture

    class BERTBaseClassifier(nn.Module):
        """
        Bi-LSTM on top of frozen embeddings initialized with GloVe vectors, followed by 1D max pooling
        on all the outputs of the Bi-LSTM layer.
        """
        __name__ = "BERTbase"

        def __init__(self, keep_prob):
            super(BERTBaseClassifier, self).__init__()
            self.BERT = transformers.BertModel.from_pretrained("bert-base-uncased")
            self.BERT.requires_grad_(False)  # Embeddings are frozen
            self.dropout = nn.Dropout(1 - keep_prob)
            self.hidden2bin = nn.Linear(768, 2)  # For Bi-LSTM

        def forward(self, ids, mask, token_type_ids):
            batch_size = ids.shape[0]
            _, hidden = self.BERT(ids, attention_mask=mask, token_type_ids=token_type_ids)
            hidden = self.dropout(hidden)
            logits = self.hidden2bin(hidden.view(batch_size, 768))
            return logits

    classifier = BERTBaseClassifier(keep_prob=0.7)

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

    train_and_evaluate_transformer_model(classifier=classifier,
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

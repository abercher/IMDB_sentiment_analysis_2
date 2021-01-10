# Sentitment Analysis on IMDB dataset with Pytorch: Training neural networks

## Introduction
This project builds upon a similar project I did in 2018. Back then
My goal was mainly to get familiar with the 3 main python programming libraries
used in deep learning (Keras, TensorFlow, and Pytorch). I used the 
IMDB movie reviews data set and implemented a model based on LSTM and
GloVes embedding to perform sentiment analysis (binary classification).
In December 2021, I decided to start again this project but put the focus
on the optimization of the Neural Networks used. Unlike in my previous
project, I will use only Pytorch for deep learning. My goals are:
* Get more familiar with the process of tuning the parameters of a neural
network and training it.
  
* Experimenting with several deep learning architectures and try to beat
* Modifying the initial architecture and measure the impact on the performance.
a simple but efficient baseline (the best one obtained in my previous project).

## Data
I downloaded the IMDB movie reviews data from Kaggle at:
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

In order to compare different models I split the data into a 70/15/15
train/valid/test format. The validation set was use to compare neural networks
with different architecture and/or optimization methods. The test set is used
only to compare (at the end of the project) the performance of the best neural
network with the baseline.

## Baseline
In order to assess if the NN based model obtained, performed well, I used a strong
baseline based on a simple model: TF-IDF vectorization followed by logistic regression,
achieving 0.9025 accuracy on the validation set and 0.8981 accuracy on the test set.

## Models tested
The primary goal being to get familiar with the optimization process, I decided first
compare many training set-ups (choice of optimizer + choice of learning rate + number of epochs)
on the same model. I reused the same model I had used before. Here is its pipeline
* Data cleaning (removal of punctuation and HTML tags)
* Keep only 300 first tokens and use padding
* frozen pretrained GloVe
* basic LSTM (only the output for the last token is used further)
* drop-out layer (probability to drop = 0.3)
* fully connected layer with output having only 2 dimensions

## Results and observations

### Metrics

#### Baseline
For the baseline model, the metrics on validation were

Accuracy = 0.9025  
F1 score binary = 0.9028  
Recall score = 0.9026  
Average precision score = 0.9636  
Air under the ROC = 0.9643  

and on the test set

Accuracy = 0.8981  
F1 score binary = 0.9003  
Recall score = 0.8980  
Average precision score = 0.9627  
Air under the ROC = 0.9643  

#### Initial GloVe + LSTM model

For the initial LSTM model taken from the previous iteration of this project
(GloVe Frozen Embedding, LSTM, 10 epochs, learning rate 0.1, Adagrad)
the metrics on the validation set were

Accuracy = 0.8676  
F1 score binary = 0.8692  
Recall score = 0.8833  
Average precision score = 0.8556  
Air under the ROC = 0.9325  

#### Best performing model when changing only optimization

Here I present the best performing model (so far)
when taking the initial model above and changing only the optimization method
(meaning the optimizer, learning rate schedule, and number of epochs). I use
as metric the metrics of the model for the epoch which gave the highest
accuracy.

As of the 10.01.2021, the best results were obtained with Adadelta, 40 epochs, fixed
fixed learning rate 1. Metrics on validation set (at epoch 20) were

Accuracy = 0.8783  
F1 score binary = 0.8799  
Recall score = 0.8956  
Average precision score = 0.8648  
Air under the ROC = 0.9466  



### Observations
* The choice of the optimizer can have a big impact on the performance of the model.
While vanilla SGD, Adagrad, and Adadelta gave similar performances (between 0.84 and 0.875
  accuracy on validation set), Adam didn't manage to properly tune the parameters:
  performances from one epoch to the next fluctuated a lot from one epoch to
  the next, and achieved only 0.6852 as maximum accuracy on the validation set
  in its best setting (fixed learning rate of 0.1, 40 epochs). It performed
  particularly poorly with fixed learning rate 1, with the model clearly not learning
  anything even after 40 epochs.
  
* Adadelta seemed to consistently provide the best results for the initial model (0.8763 ac8uracy
  on validation set), accross the different learning rate schedules.
  
* Using annealing or cyclical learning rate seemed to give slightly better results and a faster convergence than fixed learning rate.

* Trying out different optimizer, learning rates schedules, and number of epochs
  allowed the accuracy to go from to go from 0.8676 (Adagrad, 10 epochs, fixed learning rate 0.1)
  to 0.8783 (Adadelta, 30 epochs, fixed learning rate 1), so 1% accuracy increase.
  
* Simply replacing the LSTM layer by a Bi-LSTM resulted in a model enable to learn
  (accuracy on validation set staying close to 0.5). I interpret this result in the
  following way: since the next layer (fully connected layer) uses only the output
  of the (Bi-)LSTM layer for the last token, which is the concatenation of the output
  of the forward layer and the output of the backward layer, adding the backward
  layer is equivalent to doubling the output dimension but the entries added carry
  close to no information. Indeed, if the forward layer iterates through the whole
  sequence before finally taking the last token as input (which means at this point
  that the cell state carries information about the whole review), the backward
  layer sees the last token as its first input which means that its cell state is
  randomly picked (no information) and hence meaningless. Add to this the fact
  that for most reviews, the last token will be a padding token, you can conclude
  that most of the entries added by the backward layer are meaningless and that
  the number of parameters of the forward layer is unnecessarily doubled. This
  interpretation is validated by the fact that if instead of using only the output
  of the Bi-LSTM layer for the last token in the forward layer, we pass the output
  of the Bi-LSTM layer for all tokens through a layer of 1D MaxPool layer (hence
  giving a chance to the backward layer to output information regarding the whole
  sequence) and then to the fully connected layer, then performance are even
  better than the simple LSTM.
  
* The time needed to train one epoch seems inversely correlated with the accuracy
  obtained.


  


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 12:23:10 2021

@author: shahan
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Model
from keras.models import Sequential
from keras.layers import Lambda, GlobalAveragePooling1D, Dense, Embedding, concatenate, Bidirectional, Flatten
from keras.layers import Dropout, Input, LeakyReLU, Conv1D, GlobalMaxPooling1D,InputLayer, ReLU, LSTM

from sklearn.manifold import TSNE

#metrics
from sklearn.metrics import f1_score , recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import roc_curve, auc


import warnings
warnings.filterwarnings('ignore')

from nltk.tokenize import word_tokenize

#Mark Down print
from IPython.display import Markdown, display

def printmd(string):
    # Print with Markdowns    
    display(Markdown(string))
    
"""
Helper function to show Confusion diagram and ROC-AUC 

ROC-AUC : Compute Area Under the Curve (AUC) using the trapezoidal rule. This is a general function, given points on a curve. 
For computing the area under the ROC-curve.

AUC - ROC curve is a performance measurement for the classification problems at various threshold settings. 
ROC is a probability curve and AUC represents the degree or measure of separability. It tells how much the 
model is capable of distinguishing between classes. Higher the AUC, the better the model is at predicting 0 
classes as 0 and 1 classes as 1. By analogy, the Higher the AUC, the better the model is at distinguishing between patients with the disease and no disease.
"""
def get_confusion_matrix_heatmap(y_test,y_pred,fName):
    # Confusion Matrix
    # sklearn builtin function to calculate confusion matrix values using true labels and predictions
    CF = confusion_matrix(y_test,y_pred.round())
    # list of labels that will be displayed on the image boxes
    labels = ['True Neg','False Pos','False Neg','True Pos']
    # list of all possible label values
    categories = ['Spam', 'Ham']
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    # count total values present in each cell of the matrix
    group_counts = ["{0:0.0f}".format(value) for value in CF.flatten()]
    # count percentage of total values present in each cell of the matrix
    group_percentages = ["{0:.2%}".format(value) for value in CF.flatten()/np.sum(CF)]
    # group the labels to plot in graph
    labels = [f"{v1}\n{v2}\n{v3}"for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    # reshape true label values according to the requirement
    labels = np.asarray(labels).reshape(2,2)
    # declare graph using heatmap function
    heatmap=sns.heatmap(CF, annot=labels, fmt='', cmap='Blues')
    # plot confusion matrix
    fig = heatmap.get_figure()
    # save confusion matrix as image in results folder
    # fig.savefig('drive/MyDrive/SPAM classification deep learning/heatmaps/'+fName)
    fig.savefig(f"heatmaps/{fName}")
    # display confusion matrix as numeric values
    print(CF)

def ROC_AUC(y_test, y_pred, fname):
    # evluate true positive rate and false positive rate using sklearn builtin function
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
    # find area under curve score
    lr_auc = auc(lr_fpr, lr_tpr)

    # display auc score
    print("AUC:", lr_auc)
    # plot linear line with no learning
    plt.plot([0, 1], [0, 1], 'k--')
    # plot tpr and fpr ratio
    plt.plot(lr_fpr, lr_tpr, marker='.', label='lr (auc = %0.3f)' % lr_auc)
    # assign labels
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Characterisics')
    plt.legend(loc='lower right')
    # plt.savefig(f"drive/MyDrive/SPAM classification deep learning/Visuals/{fname}")
    plt.savefig(f"Visuals/{fname}")
    return lr_auc
  
# ANN - TSNE - word Embedding visualisations
def Word_Embeddings_visualise_TSNE(model, X, idx_to_word, fname, lname, lim1, lim2, lim3, lim4, Flag=False):
    np.random.seed(1)
    print(f"First sample of the training data set vectorised \n{X[0]}\n")
    text = [idx_to_word[idx] if idx != 0 else "<UNK>" for idx in X[0]]
    print(f"First sample of the training data set using index to word : \n{' '.join(text)}\n")
    
    ## Extraction of word Embeddings
    word_embeddings = model.get_layer(lname).get_weights()[0]
    print('Shape of word_embeddings:', word_embeddings.shape)
    
    # Ploting the word embeddings using TSNE
    tsne = TSNE(perplexity=3, n_components=2, init='pca', n_iter=500, method='exact')
    np.set_printoptions(suppress=True)
    plot_only = 60
    
    T = tsne.fit_transform(word_embeddings[:plot_only, :])
    labels = [idx_to_word[i+1] for i in range(plot_only)]
    plt.figure(figsize=(14, 8))
    if(Flag):
      plt.ylim(lim1, lim2)
      plt.xlim(lim3, lim4)
    plt.scatter(T[:, 0], T[:, 1])
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points', ha='right',
    va='bottom')
    # plt.savefig(f"drive/MyDrive/SPAM classification deep learning/Visuals/{fname}.jpeg")
    plt.savefig(f"Visuals/{fname}.jpeg")




"""
GLOVE : Global vectors for word representation

Definition : 
GloVe is an unsupervised learning algorithm for obtaining vector representations 
for words. Training is performed on aggregated global word-word co-occurrence statistics from a 
corpus, and the resulting representations showcase interesting linear substructures of the word 
vector space.

Reference : Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: 
            Global Vectors for Word Representation.

Usage : Using Glove Embedding matrix to convert input data/sample to embedding vectors
"""
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def X_to_index(text, word_to_index, max_length):
  
    vectorised = word_tokenize(text)
    vectorised_index = np.zeros((max_length))
    # vectorised_index = [word_to_index[w] if w in word_to_index.keys() else  0 for w in vectorised]
    i = 0
    for w in vectorised:
      if w in word_to_index.keys():
        vectorised_index[i] = word_to_index[w]
      else:
        vectorised_index[i] = 0
      i += 1
      if i >= 2400:
        break
    return vectorised_index

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. Get a list of words.
        sentence_words =X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            if w in word_to_index.keys():
              X_indices[i, j] = word_to_index[w]
            else:
              X_indices[i, j] = 0
            
            
            # Increment j to j + 1
            j = j + 1
            if j >= 2400:
              break
    return X_indices

"""
Model comparison table with metrics : 
1. Accuracy
2. Loss
3. Error
4. Precision, Recall, F1 score
5. ROC AUC
"""
def get_Metrics(y_test, y_pred, average="macro"):
    
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
    # find area under curve score
    lr_auc = auc(lr_fpr, lr_tpr)
    precision = precision_score(y_test, y_pred, average = average)
    recall = recall_score(y_test, y_pred, average = average)
    f1_score_ = f1_score(y_test, y_pred, average = average)
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"precision : {precision} recall : {recall} f1_score : {f1_score_} accuracy : {accuracy}")
    return precision, recall, f1_score_, accuracy, lr_auc

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding 
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of GloVe word vectors
    
    # Initialize the embedding matrix as a numpy array of zeros.
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,)) 
    
    # Set the weights of the embedding layer to the embedding matrix. The layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def ANN_with_glove(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the ANN_with_glove model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    
    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32'.
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through a max pooling layer with default kernal
    X = GlobalMaxPooling1D()(embeddings)
    
    # Propagate X through a Dense layer with relu activation to get back activation of next layer
    X = Dense(20, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(10, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back y_pred
    X = Dense(1, activation='sigmoid')(X)
        
    # # Add a sigmoid activation
    # X = Activation('sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model

def ANN_with_glove_architecture_2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the ANN_with_glove model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    
    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32'.
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through a max pooling layer with default kernal
    X = GlobalMaxPooling1D()(embeddings)
    
    # Propagate X through a Dense layer with relu activation to get back activation of next layer
    X = Dense(512, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(256, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with relu activation to get back activation of next layer
    X = Dense(128, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(64, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    #X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with relu activation to get back activation of next layer
    X = Dense(32, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    #X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(16, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    #X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back y_pred
    X = Dense(1, activation='sigmoid')(X)
        
    # # Add a sigmoid activation
    # X = Activation('sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model  
    
def RNN_with_glove_architecture_2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the ANN_with_glove model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    
    Returns:
    model -- a model instance in Keras
    """
    
    
    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32'.
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)    
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=True)(embeddings)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=False)(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    
    # Propagate X through a Dense layer with relu activation to get back activation of next layer
    X = Dense(32, activation='relu')(X)
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(16, activation='relu')(X)
    # Propagate X through a Dense layer with sigmoid activation to get back y_pred
    X = Dense(1, activation='sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model

def RNN_with_glove(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the ANN_with_glove model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    
    Returns:
    model -- a model instance in Keras
    """
    
    
    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32'.
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)    
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=True)(embeddings)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=False)(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    
    # Propagate X through a Dense layer with relu activation to get back activation
    X = Dense(5, activation='relu')(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back y_pred
    X = Dense(1, activation='sigmoid')(X)
    
    # # Add a sigmoid activation
    # X = Activation('sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model

def CNN_with_glove(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the ANN_with_glove model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    
    Returns:
    model -- a model instance in Keras
    """
    
    
    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32'.
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)    
    
    ''' --------------------
    - initialize second hidden layer as convolutional layer
    - number of filters that cnn will return is set to 20
    - kernel size is set to 3
    - padding is set to valid, which means no padding will be applied
    - number of strides is set to default that is 1
    -------------------- '''
    X = Conv1D(activation="relu",filters=20, kernel_size=3, padding="valid")(embeddings)
    
    
    # Propagate the embeddings through a max pooling layer with default kernal
    X = GlobalMaxPooling1D()(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(units = 20)(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(10, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back y_pred
    X = Dense(1, activation='sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model

def CNN_with_glove_architecture_2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the ANN_with_glove model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    
    Returns:
    model -- a model instance in Keras
    """
    
    
    # Define sentence_indices as the input of the graph
    # It should be of shape input_shape and dtype 'int32'.
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)    
    
    ''' --------------------
    - initialize second hidden layer as convolutional layer
    - number of filters that cnn will return is set to 20
    - kernel size is set to 3
    - padding is set to valid, which means no padding will be applied
    - number of strides is set to default that is 1
    -------------------- '''
    X = Conv1D(activation="relu",filters=100, kernel_size=5, padding="valid")(embeddings)
    
    # Propagate the embeddings through a max pooling layer with default kernal
    X = GlobalMaxPooling1D()(X)
    # Add dropout with a probability of 0.2
    X = Dropout(0.2)(X)
    
    X = Conv1D(activation="relu",filters=100, kernel_size=4, padding="valid")(X)
    
    # Propagate the embeddings through a max pooling layer with default kernal
    X = GlobalMaxPooling1D()(X)
    
    X = Flatten(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(units = 64, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(units = 32, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(units = 16, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back activation of next layer
    X = Dense(units = 8, activation='relu')(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X through a Dense layer with sigmoid activation to get back y_pred
    X = Dense(1, activation='sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    return model

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
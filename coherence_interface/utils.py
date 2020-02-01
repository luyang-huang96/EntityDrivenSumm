# File:
# -*- coding: utf-8 -*-
# @Time    : 4/13/2019 3:57 PM
# @Author  : Derek Hu
import numpy as np
import json


glove_300d = "/data2/zhe/glove/glove.42B.300d.txt"

def to_categorical(label):
    if label == 0:
        return np.array([1, 0])
    if label == 1:
        return np.array([0, 1])


def load_embeddings(path, word_dict, embedding_dim):
    print("Loading Glove vectors...")
    word_vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = embedding
    #word_embedding_matrix = np.zeros((len(word_dict), embedding_dim))
    word_embedding_matrix = np.random.normal(0, 1, (len(word_dict), embedding_dim))
    for word, i in word_dict.items():
        embedding_vector = word_vectors.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector
    return np.array(word_embedding_matrix)


def batch_iter_json(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    num_batches_per_epoch = (len(data) - 1) // batch_size + 1

    for epoch in range(num_epochs):
        # print("-------"*10, "epoch: ", epoch, "-------"*10)
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(len(data)))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(data))
            yield shuffled_data[start_index:end_index], \
               epoch

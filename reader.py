"""Utilities for parsing who_did_what text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import numpy as np
import tensorflow as tf


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        fi = f.read().decode("utf-8").strip().split("\n")
    return [re.split('[$$$\s]', x) for x in fi]


def build_vocab(questions_file, context_file, choices_file):
  # right now counts across all fns
    data = []
    for filename in [questions_file, context_file]:
        data_curr = _read_words(filename)
        data.extend(data_curr)
    all_words = [item for sublist in data for item in sublist]

    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # Get top-50000 words
    count_pairs = count_pairs[:50000]

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    # Add choices not in the top-50000 to word_to_id
    choices_file = open(choices_file, "r")
    for line in choices_file.readlines():
        for choice in line.strip().split("$$$"):
            if choice not in word_to_id:
                word_to_id[choice] = 1
    choices_file.close()
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    for i, l in enumerate(data):
        data[i] = [word_to_id[word] for word in l if word in word_to_id]
    return data

def pad(m):
    max_size = max([len(s) for s in m])
    new = 100000 * np.ones((len(m), max_size))
    for sentence_ind, word_indices in enumerate(m):
        new[sentence_ind, :len(word_indices)] = word_indices
    return new

def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

    Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
    """
    qu_fn = 'qu.txt'
    context_fn = 'context.txt'
    choice_fn = 'choices.txt'
    labels_fn = 'labels.txt'

    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "val")
    test_path = os.path.join(data_path, "test")

    tr_qu = os.path.join(train_path, qu_fn)
    tr_cont = os.path.join(train_path, context_fn)
    tr_ch = os.path.join(train_path, choice_fn)
    tr_lab = os.path.join(train_path, labels_fn)

    val_qu = os.path.join(valid_path, qu_fn)
    val_cont = os.path.join(valid_path, context_fn)
    val_ch = os.path.join(valid_path, choice_fn)
    val_lab = os.path.join(valid_path, labels_fn)

    te_qu = os.path.join(test_path, qu_fn)
    te_cont = os.path.join(test_path, context_fn)
    te_ch = os.path.join(test_path, choice_fn)
    te_lab = os.path.join(test_path, labels_fn)

    word_to_id = _build_vocab([tr_qu, tr_cont])
    train_data_cont = pad(_file_to_word_ids(tr_cont, word_to_id))
    train_data_qu = pad(_file_to_word_ids(tr_qu, word_to_id))
    train_data_ch = _file_to_word_ids((tr_ch), word_to_id)
    train_data_lab = _file_to_word_ids(tr_lab, word_to_id)

    valid_data_cont = pad(_file_to_word_ids(val_cont, word_to_id))
    valid_data_qu = pad(_file_to_word_ids(val_qu, word_to_id))
    valid_data_ch = _file_to_word_ids((val_ch), word_to_id)
    valid_data_lab = _file_to_word_ids(val_lab, word_to_id)

    test_data_cont = pad(_file_to_word_ids(te_cont, word_to_id))
    test_data_qu = pad(_file_to_word_ids(te_qu, word_to_id))
    test_data_ch = _file_to_word_ids((te_ch), word_to_id)
    test_data_lab = _file_to_word_ids(te_lab, word_to_id)

    vocabulary = len(word_to_id)

    return (train_data_cont, train_data_qu, train_data_ch, train_data_lab),\
          (valid_data_cont, valid_data_qu), \
          (test_data_cont, test_data_qu), vocabulary

def batch_iter(data_path=None, batch_size=32, num_epochs=5, random_state=0):
    """
    Generates a batch iterator for a dataset.
    """
    rng = np.random.RandomState(0)
    qu_fn = 'qu.txt'
    context_fn = 'context.txt'
    choice_fn = 'choices.txt'
    labels_fn = 'labels.txt'

    if data_path is None:
        data_path = os.getcwd()

    train_path = os.path.join(data_path, "train")
    tr_qu = os.path.join(train_path, qu_fn)
    tr_cont = os.path.join(train_path, context_fn)
    tr_ch = os.path.join(train_path, choice_fn)
    tr_lab = os.path.join(train_path, labels_fn)

    questions_file = open(tr_qu, "r")
    questions = questions_file.readlines()
    questions_file.close()

    context_file = open(tr_cont, "r")
    context = context_file.readlines()
    context_file.close()

    choices_file = open(tr_ch, "r")
    choices = choices_file.readlines()
    choices_file.close()

    labels_file = open(tr_lab, "r")
    labels = labels_file.readlines()
    labels_file.close()

    word_to_id = build_vocab(tr_qu, tr_cont, tr_ch)
    return word_to_id

    # data_size = len(questions)
    # data_indices = np.arange(data_size)
    # num_batches_per_epoch = int(data_size / batch_size) + 1
    # for epoch in range(num_epochs):
    #     # Shuffle the data at each epoch
    #     if shuffle:
    #         shuffle_indices = rng.permutation(data_indices)
    #         shuffled_qs = data[shuffle_indices]
    #         shuffled_cont = context[shuffle_indices]
    #         shuffled_choices = choices[shuffle_indices]
    #         shuffled_labels = labels[shuffle_indices]
    #
    #     for batch_num in range(num_batches_per_epoch):
    #         start_index = batch_num * batch_size
    #         end_index = min((batch_num + 1) * batch_size, data_size)
    #         yield shuffled_data[start_index:end_index]

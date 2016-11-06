"""Utilities for parsing who_did_what text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from string import punctuation
import re
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        fi = f.read().decode("utf-8").strip().split("\n")
    return [re.split('[$$$\s]', x) for x in fi]


def build_vocab(questions, encoded_context, word_cutoff=50000):
    data = [questions, encoded_context]
    all_words = [token for sublist in data for item in sublist for token in item.split()]

    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    # Get top-50000 words
    count_pairs = count_pairs[:word_cutoff]

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _word_to_word_ids(words, word_to_id):
    data = []
    for l in words:
        data.append([word_to_id.get(word, 50000) for word in l.split()])
    return data


def pad(m, max_size):
    new = 100000 * np.ones((len(m), max_size))
    for sentence_ind, word_indices in enumerate(m):
        new[sentence_ind, :len(word_indices)] = word_indices
    return new


def strip_punctuation(lines):
    """
    Strip punctuation from a list of lines
    """
    stripped_lines = []
    for line in lines:
        stripped_lines.append(
            ' '.join([token for token in clean_str(line).split(' ') if token not in punctuation])
        )
    return stripped_lines


def encode_context_with_entities(contexts, choices, choice_to_id):
    """
    Replace all entities in the questions with their corresponding id's
    as given by choice_to_id
    """
    encoded_context = []
    for qs_choice, context in zip(choices, contexts):
        for choice in qs_choice:
            context = context.replace(choice, "entity_" + str(choice_to_id[choice]))
        encoded_context.append(context)
    return encoded_context


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
    questions = strip_punctuation(questions_file.readlines())[:2000]
    questions_file.close()

    context_file = open(tr_cont, "r")
    context = strip_punctuation(context_file.readlines())[:2000]
    context_file.close()

    choices_file = open(tr_ch, "r")
    choices = choices_file.readlines()[:2000]
    choices_file.close()
    all_choices = []
    list_of_choices = []

    for choice in choices:
        all_choices.extend([t for t in choice.strip().split("$$$")])
        list_of_choices.append([t for t in choice.strip().split("$$$")])
    le = LabelEncoder()
    le.fit(all_choices)
    choice_to_id = dict(zip(le.classes_, range(len(le.classes_))))
    id_to_choice = dict(zip(range(len(le.classes_)), le.classes_))

    encoded_choices = []
    for choice in choices:
        encoded_choices.append([choice_to_id[t] for t in choice.strip().split("$$$")])

    labels_file = open(tr_lab, "r")
    labels = le.transform(
        [label.strip() for label in labels_file.readlines()[:2000]]
    )
    labels_file.close()
    encoded_context = encode_context_with_entities(
        context, list_of_choices, choice_to_id)

    vocab = build_vocab(questions, encoded_context, word_cutoff=50000)
    data_size = len(questions)
    data_indices = np.arange(data_size)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    questions = np.asarray(questions)
    encoded_context = np.asarray(encoded_context)
    encoded_choices = np.asarray(encoded_choices)
    max_con_len = max([len(context.split(" ")) for context in encoded_context])
    max_qs_len = max([len(question.split(" ")) for question in questions])
    labels = np.asarray(labels)

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = rng.permutation(data_indices)
        shuffled_qs = questions[shuffle_indices]
        shuffled_cont = encoded_context[shuffle_indices]
        shuffled_choices = encoded_choices[shuffle_indices]
        shuffled_labels = labels[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            padded_qs = pad(
                _word_to_word_ids(shuffled_qs[start_index:end_index], vocab),
                max_qs_len)
            padded_cont = pad(
                _word_to_word_ids(shuffled_cont[start_index:end_index], vocab),
                max_con_len)

            yield (
                padded_qs, padded_cont, shuffled_choices[start_index: end_index],
                shuffled_labels[start_index: end_index]
            )

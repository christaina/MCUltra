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


def clean_str(string, choice=False):
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
    string = re.sub(r"`", "'", string)
    string = string.replace("\\)", "rrb")
    string = string.replace("\\(", "lrb")
    string = string.replace("''", " ")
    string = string.replace("' ", " ")
    return string.strip().lower()


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        fi = f.read().decode("utf-8").strip().split("\n")
    return [re.split('[$$$\s]', x) for x in fi]

def build_choices(choices):
    all_choices = [item for sublist in choices for item in sublist]
    le = LabelEncoder()
    le.fit(all_choices)
    choice_to_id = dict(zip(le.classes_, range(len(le.classes_))))
    return choice_to_id


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


def encode_choices(context, question, choices, label, i):
    """
    Assign numbers to entities based on occurence in document;
    encode that choice in the document
    """
    entity_num = 0
    choices_map = {}
    new_context = context.split(" ")
    new_question = question.split(" ")
    word_ent_map = {}
    new_word = None
    new_label = None

    for choice in choices:
        if choice not in choices_map:
            choices_map[choice] = "@entity%s" % entity_num
            entity_num += 1
        if choice not in context:
            print("choice does notexist in context: %s, id %d" % (choice, i))
        context = context.replace(choice, choices_map[choice])
        question = question.replace(choice, choices_map[choice])
        label = label.replace(choice, choices_map[choice])
    new_choices = [choices_map[x] for x in choices]
    return context, question, new_choices, label

def load_data(data_path=None):
    """
    Return a tuple of a

    1. List of contexts.
    2. List of questions.
    3. List of choices.
    4. List of labels.
    """
    qu_fn = 'qu.txt'
    context_fn = 'context.txt'
    choice_fn = 'choices.txt'
    labels_fn = 'labels.txt'
    if data_path is None:
        data_path = os.getcwd()
    qu_p = os.path.join(data_path, qu_fn)
    cont_p = os.path.join(data_path, context_fn)
    ch_p = os.path.join(data_path, choice_fn)
    lab_p = os.path.join(data_path, labels_fn)

    questions_file = open(qu_p, "r")
    questions = strip_punctuation(questions_file.readlines())
    questions_file.close()

    context_file = open(cont_p, "r")
    context = strip_punctuation(context_file.readlines())
    context_file.close()

    choices_file = open(ch_p, "r")
    choices = choices_file.read().strip().split("\n")
    choices = [x.strip().split("$$$") for x in choices]
    for i, lines in enumerate(choices):
        choices[i] = strip_punctuation(lines)
    choices_file.close()

    # Remove duplicate choices and replace the longest string
    # first.
    new_choices = []
    for i, choice in enumerate(choices):
        dup_choices = list(set(choice))
        longest_first = sorted(dup_choices, key=lambda x: -len(x))
        new_choices.append(longest_first)

    data_size = len(context)
    labels = [clean_str(l) for l in open(lab_p).read().strip().split("\n")]

    for i in range(data_size):
        context[i], questions[i], new_choices[i], labels[i] = \
                encode_choices(
                    context[i],
                    questions[i],
                    new_choices[i],
                    labels[i], i)
    return context, questions, new_choices, labels


def batch_iter(context, questions, choices, labels, vocab,
               batch_size=32, num_epochs=5, random_state=0):
    """
    Generates a batch iterator for a dataset.
    """
    rng = np.random.RandomState(0)

    data_size = len(context)
    data_indices = np.arange(data_size)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    max_con_len = max([len(cont.split(" ")) for cont in context])
    max_qs_len = max([len(question.split(" ")) for question in questions])

    questions = np.asarray(questions)
    context = np.asarray(context)
    choices = np.asarray(choices)

    labels = np.asarray(labels)

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = rng.permutation(data_indices)
        shuffled_qs = questions[shuffle_indices]
        shuffled_cont = context[shuffle_indices]
        shuffled_choices = choices[shuffle_indices]
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

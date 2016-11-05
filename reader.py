# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
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
    #return f.read().encode("utf-8").replace("\n", "<eos>").split()
    fi = f.read().encode("utf-8").strip().split("\n")
    #print(fi[0])
    #print(len(fi[0]))
    return [re.split('[$$$\s]',x) for x in fi] 


def _build_vocab(filenames):
  # right now counts across all fns
  data = []
  for filename in filenames:
    data_curr = _read_words(filename)
    data.extend(data_curr)
  all_words = [item for sublist in data for item in sublist]

  counter = collections.Counter(all_words)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  for i,l in enumerate(data):
      data[i] = [word_to_id[word] for word in l if word in word_to_id]
  return data

def pad(m):
    max_size = max([len(s) for s in m])
    print(max_size)
    print(len(m))
    new = 100000 * np.ones((len(m),max_size))
    for sentence_ind, word_indices in enumerate(m):
        new[sentence_ind, : len(word_indices)] = word_indices
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
  
  tr_qu = os.path.join(train_path,qu_fn)
  tr_cont = os.path.join(train_path,context_fn)
  tr_ch = os.path.join(train_path,choice_fn)
  tr_lab = os.path.join(train_path,labels_fn)

  val_qu = os.path.join(valid_path,qu_fn)
  val_cont = os.path.join(valid_path,context_fn)
  val_ch = os.path.join(valid_path,choice_fn)
  val_lab = os.path.join(valid_path,labels_fn)

  te_qu = os.path.join(test_path,qu_fn)
  te_cont = os.path.join(test_path,context_fn)
  te_ch = os.path.join(test_path,choice_fn)
  te_lab = os.path.join(test_path,labels_fn)

  word_to_id = _build_vocab([tr_qu,tr_cont])
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

  return (train_data_cont,train_data_qu,train_data_ch,train_data_lab),\
          (valid_data_cont,valid_data_qu), \
          (test_data_cont,test_data_qu), vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
    y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])
    return x, y

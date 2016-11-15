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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import reader as rn

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", '/scratch/ceb545/nlp/A3/simple-examples/data/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("data_wdw", '/scratch/ceb545/nlp/project/who_did_what/Strict/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("checkpoint_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

"""
class PTBInput(object):

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)

class GenBatch(object):
  def __init__(self, data, name=None):
    self.batch_size = data.batch_size
    self.num_steps = data.num_steps
    self.epoch_size = data.epoch_size
    self.input_data,self.targets = read.ptb_producer(data.context,\
            data.questions,\
            data.choices,\
            data.labels,\
            batch_size=self.batch_size,\
            num_steps=self.num_steps)
    self.vocabulary = data.vocabulary
    self.all_choices = data.all_choices


class GenInput(object):

  def __init__(self, config, data_path=None, vocabulary=None,name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    raw_context,raw_questions,raw_choices,raw_labels,self.choices_map = \
            read.load_data(data_path)
    all_choices = read.build_choices(raw_choices)
    self.epoch_size = ((len(raw_context) // batch_size) - 1) // num_steps
    # build vocab for train data
    if not vocabulary:
        self.vocabulary = read.get_vocab(raw_questions,\
                raw_context,min_frequency=500)
    else:
        self.vocabulary=vocabulary

    raw_choices = [" ".join(x) for x in raw_choices]
    self.all_choices = read.vocab_transform(all_choices,self.vocabulary)
    self.questions = read.vocab_transform(raw_questions,self.vocabulary)
    self.context = read.vocab_transform(raw_context,self.vocabulary)
    self.labels = read.vocab_transform(raw_labels,self.vocabulary)
    self.choices = read.vocab_transform([" ".join(x) for x in raw_choices],self.vocabulary)
"""
class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):

    print("input data shape:")
    choices_size = 5
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    self.size = config.hidden_size
    self.vocab_size = config.vocab_size

    self.input_x = tf.placeholder(tf.int32,[self.batch_size,self.num_steps])
    self.input_y = tf.placeholder(tf.int32,[self.batch_size,self.num_steps])
    self.sequence_lengths = tf.placeholder(tf.int32,[self.batch_size])

    #choices_size=vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.

    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size, forget_bias=0.0, state_is_tuple=False)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size, forget_bias=0.0, state_is_tuple=False)
    if is_training and config.keep_prob < 1:
      lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_fw_cell, output_keep_prob=config.keep_prob)
      lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_bw_cell, output_keep_prob=config.keep_prob)
    #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=False)
    self._initial_state_fw = lstm_fw_cell.zero_state(self.batch_size, data_type())
    self._initial_state_bw = lstm_bw_cell.zero_state(self.batch_size, data_type())
    self._initial_state = (self._initial_state_fw,self._initial_state_bw)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [self.vocab_size, self.size], dtype=data_type())
      #inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
      inputs = tf.nn.embedding_lookup(embedding, self.input_x)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    inputs = [tf.squeeze(input_step, [1])
              for input_step in tf.split(1, self.num_steps, inputs)]
    print(len(inputs))
    outputs, state_fw, state_bw = tf.nn.bidirectional_rnn(
            lstm_fw_cell, \
                    lstm_bw_cell,\
                    inputs, \
                    initial_state_fw=self._initial_state[0],\
                    initial_state_bw=self._initial_state[1])

    print("Recieved output tensor %s long"%len(outputs))
    print("Each element has shape %s"%outputs[0].get_shape())
    concat_outputs = tf.concat(1,outputs)
    print("Shape after concatting: %s"%concat_outputs.get_shape())
    # concat list of outputs
    output = tf.reshape(concat_outputs, [-1, self.size])
    print("Shape after concatting + reshaping: %s"%output.get_shape())
    softmax_w = tf.get_variable(
        "softmax_w", [self.size, choices_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [choices_size], dtype=data_type())
    print("Softmax shape:%s,%s"%(softmax_w.get_shape(),softmax_b.get_shape()))
    logits = tf.matmul(output, softmax_w) + softmax_b
    print("Logits shape : %s"%logits.get_shape())

    print("original y shape: %s"%self.input_y.get_shape())
    y_ext = tf.expand_dims(self.input_y,2)
    y_doubles = tf.concat(2,[y_ext,y_ext])
    print("new y shape: %s"%y_doubles.get_shape())
    y_grp = tf.reshape(y_doubles,[self.batch_size,-1])
    loss_weights = tf.ones([self.batch_size * self.num_steps * 2],dtype=data_type())
    print("y shape: %s"%y_grp.get_shape())
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [y_grp],
        [loss_weights])
    self._cost = cost = tf.reduce_sum(loss) / self.batch_size
    self._final_state = (state_fw,state_bw)

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
  """
  @property
  def input(self):
    return self._input
  """
  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 21
  vocab_size = 13065


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, input, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  for j, batch in enumerate(input):

    questions, context, choices, labels, map, all_choices, vocabulary = batch
    m_vocab_size = str(len(vocabulary.vocabulary_))
    print("vocab_size %s" % m_vocab_size)
    mapped_labels = ([all_choices[x] for x in labels])

    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for i,step in enumerate(context):
        print("step %s, batch %s"%(i,j))
        feed_dict = {}
    #input_x = input.input_data.eval(session=session)
    #input_y = input.input_data.eval(session=session)
        feed_dict[model.initial_state] = state
        feed_dict[model.input_x]=step
        reshape_labels = np.array([x*np.ones(len(step[1])) for x in mapped_labels])
        feed_dict[model.input_y]=reshape_labels
        feed_dict[model.sequence_lengths] = rn.get_seq_length(step)

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.num_steps
        """
        if verbose and j % (model.epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))
        """
  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  train_path = os.path.join(FLAGS.data_wdw, 'test')
  #raw_data = reader.ptb_raw_data(FLAGS.data_path)
  #train_data, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  print("Loading WDW Data..")
  #train_data_wdw = GenInput(config,data_path=train_path)
  print("loading iter data..")
  train_data_wdw_iter = rn.batch_iter(
      train_path, batch_size=config.batch_size,
      num_epochs=config.max_epoch, context_num_steps=config.num_steps)

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      #train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config)
      tf.scalar_summary("Training Loss", m.cost)
      tf.scalar_summary("Learning Rate", m.lr)
    """
    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.scalar_summary("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_input)
    """
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, train_data_wdw_iter,
                                     eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        """
        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session, mtest)
      print("Test Perplexity: %.3f" % test_perplexity)
      """
      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()

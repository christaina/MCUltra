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

class RawInput(object):
    def __init__(self, data_bundle):
        (self.contexts, self.questions, self.choices, self.labels,
            self.choices_map, self.context_lens, self.qs_lens) = data_bundle
        self.vocab = rn.get_vocab(
            self.questions, self.contexts, min_frequency=10)
        self.vocab_size = len(self.vocab.vocabulary_)
        self.labels_idx = list(set([choice for choices in self.choices for choice in choices]))


class BiLSTM(object):
    """
    Bidirectional LSTM
    """
    def __init__(self, input_x, sequence_lengths,
                 is_training, config, num_steps, embedding,name):
        self.batch_size=config.batch_size
        self.size = config.hidden_size
        self.num_steps = num_steps
        self.input_x = input_x
        self.sequence_lengths = sequence_lengths
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.size, forget_bias=0.0, state_is_tuple=False)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
          lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_fw_cell, output_keep_prob=config.keep_prob)
          lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_bw_cell, output_keep_prob=config.keep_prob)
        self._initial_state_fw = lstm_fw_cell.zero_state(
            self.batch_size, data_type())
        self._initial_state_bw = lstm_bw_cell.zero_state(
            self.batch_size, data_type())
        self._initial_state = (self._initial_state_fw, self._initial_state_bw)
        inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_step, [1])
                  for input_step in tf.split(1, self.num_steps, inputs)]
        self.outputs, self.state_fw, self.state_bw = tf.nn.bidirectional_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                inputs,
                initial_state_fw=self._initial_state[0],
                initial_state_bw=self._initial_state[1],\
                        sequence_length=self.sequence_lengths,scope="BiRNN_%s"%name)



class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, vocab_size, labels_idx, context_steps, question_steps):

        print("input data shape:")
        self.num_steps =context_steps
        choices_size = len(labels_idx)
        self.labels_idx = labels_idx
        self.batch_size = config.batch_size
        self.size = config.hidden_size
        self.vocab_size = vocab_size
        self.context_steps = context_steps
        self.question_steps = question_steps

        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.context_steps])
        self.input_y = tf.placeholder(tf.int32, [self.batch_size, self.context_steps])
        self.sequence_lengths = tf.placeholder(tf.int32, [self.batch_size])
        with tf.device("/cpu:0"):
          embedding = tf.get_variable(
              "embedding", [self.vocab_size, self.size], dtype=data_type())

        # bidirectional lstm
        qlstm = BiLSTM(self.input_x, self.sequence_lengths,
                       is_training, config, context_steps, embedding, name='context')
        self._initial_state = qlstm._initial_state
        print("Shape of last output: %s" % qlstm.outputs[-1].get_shape())

        concat_outputs = qlstm.outputs[-1]
        state_fw = qlstm.state_fw
        state_bw = qlstm.state_bw
        output = tf.reshape(concat_outputs, [-1, self.size])
        print("Shape after concatting + reshaping: %s" % output.get_shape())
        softmax_w = tf.get_variable(
            "softmax_w", [self.size, choices_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [choices_size], dtype=data_type())
        print("Softmax shape:%s,%s"%(softmax_w.get_shape(),softmax_b.get_shape()))
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._logits = tf.reshape(logits,[self.batch_size,-1,2,choices_size])
        self._logits = tf.reduce_max(self.logits,reduction_indices=2)
        logits = tf.reshape(self._logits,[-1,choices_size])
        self._predictions = tf.argmax(self._logits,2)
        print("Logits shape : %s" % logits.get_shape())

        print("original y shape: %s"%self.input_y.get_shape())
        y_ext = tf.expand_dims(self.input_y,2)
        y_doubles = tf.concat(2,[y_ext,y_ext])
        print("new y shape: %s"%y_doubles.get_shape())
        y_grp = tf.reshape(y_ext,[self.batch_size,-1])
        y_grp = tf.reduce_max(y_grp,reduction_indices=1)
        loss_weights = tf.ones([self.batch_size ],dtype=data_type())
        print("y shape: %s"%y_grp.get_shape())
        print("loss shape: %s"%loss_weights.get_shape())
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [y_grp],
            [loss_weights])
        correct_preds = tf.equal(tf.to_int32(self._predictions),y_grp)
        self._acc = tf.reduce_mean(tf.cast(correct_preds,"float"))
        self._cost = cost = tf.reduce_sum(loss) / self.batch_size
        self._final_state = state_fw, state_bw

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

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def acc(self):
        return self._acc

    @property
    def c_initial_state(self):
        return self._c_initial_state

    @property
    def predictions(self):
        return self._predictions

    @property
    def logits(self):
        return self._logits

    @property
    def cost(self):
        return self._cost

    @property
    def targets(self):
        return self.input_y

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
    learning_rate = 0.01
    max_grad_norm = 5
    num_layers = 2
    num_steps = 300
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 32
    vocab_size = 13065

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
    context_steps = model.context_steps

    for j, batch in enumerate(input):

        questions, context, choices, labels, map, context_lens, qs_lens = batch
        mapped_labels = ([model.labels_idx.index(x) for x in labels])

        state = session.run(model.initial_state)

        fetches = {
          "cost": model.cost,
          "final_state": model.final_state,
          "predictions":model.predictions,
          "logits": model.logits,
          "targets": model.targets,
          "acc": model.acc
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        for step_ind, step in enumerate(context):
            feed_dict = {}
            feed_dict[model.initial_state] = state
            feed_dict[model.input_x] = step
            reshape_labels = np.array([x*np.ones(len(step[1])) for x in mapped_labels])
            feed_dict[model.input_y] = reshape_labels
            seq_len = rn.get_seq_length(
                context_lens, step_ind, context_steps)
            feed_dict[model.sequence_lengths] = seq_len
            print(seq_len)
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]
            if step_ind==0:
                print("batch %s; accuracy: %s"%(j, vals["acc"]))

                print("predictions: %s"%vals["predictions"].T)

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

    train_path = os.path.join(FLAGS.data_wdw, 'test')
    val_path = os.path.join(FLAGS.data_wdw, 'val')
    test_path = os.path.join(FLAGS.data_wdw, 'test')

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    q_steps = 20
    c_steps = 300
    config.num_steps = c_steps

    print("Loading WDW Data..")
    train = RawInput(rn.load_data(train_path))

    print("loading iter data..")
    train_iter = rn.batch_iter(
        train.contexts, train.questions,
        train.choices, train.labels, train.choices_map, train.context_lens,
        train.qs_lens, batch_size=config.batch_size,
        num_epochs=config.max_epoch, context_num_steps=c_steps,
        question_num_steps=q_steps)

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config, vocab_size=train.vocab_size,
                             labels_idx=train.labels_idx, context_steps=c_steps,
                             question_steps = q_steps)
        tf.scalar_summary("Training Loss", m.cost)
        tf.scalar_summary("Learning Rate", m.lr)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                run_epoch(session, m, train_iter, eval_op=m.train_op,
                          verbose=True)
            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()

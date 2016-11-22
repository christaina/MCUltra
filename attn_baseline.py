from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.preprocessing import LabelBinarizer
import reader as rn

flags = tf.flags
logging = tf.logging

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
    def __init__(self, data_bundle,vocabulary=None,c_len=None,q_len=None):
        (self.contexts, self.questions, self.choices, self.labels,
            self.choices_map, self.context_lens, self.qs_lens) = data_bundle
        if vocabulary:
            self.vocab = vocabulary
        else:
            self.vocab = rn.get_vocab(
                self.questions, self.contexts, min_frequency=10)
        self.vocab_size = len(self.vocab.vocabulary_)

        self.labels_idx = sorted(
            list(set([choice for choices in self.choices for choice in choices]))
        )

        self.contexts = rn.vocab_transform(self.contexts,self.vocab)
        self.questions = rn.vocab_transform(self.questions,self.vocab)
        if c_len:
            self.contexts = rn.pad_eval(self.contexts,c_len)
        if q_len:
            self.questions = rn.pad_eval(self.questions,q_len)
        else:
            self.c_len = len(self.contexts[0])
            self.q_len = len(self.questions[0])
            

class BiLSTM(object):
    """
    Bidirectional LSTM
    """
    def __init__(self, input_x, sequence_lengths,
                 is_training, config, num_steps, embedding,name):
        self.batch_size = config.batch_size
        self.size = config.hidden_size
        self.num_steps = num_steps
        self.input_x = input_x
        self.sequence_lengths = sequence_lengths
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.size, forget_bias=0.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.size, forget_bias=0.0, state_is_tuple=True)
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
                initial_state_bw=self._initial_state[1],
                sequence_length=self.sequence_lengths,scope="BiRNN_%s"%name)


class Model(object):
    """The PTB model."""

    def __init__(self, is_training, config, vocab_size, labels_idx,
                 context_steps, question_steps):
        
        self.num_steps = context_steps
        labels_size = len(labels_idx)
        self.labels_idx = labels_idx
        self.batch_size = config.batch_size
        self.size = config.hidden_size
        self.vocab_size = vocab_size
        self.context_steps = context_steps
        self.question_steps = question_steps

        self.q_x = tf.placeholder(tf.int32,[self.batch_size,self.question_steps])
        self.q_lengths = tf.placeholder(tf.int32, [self.batch_size])
        self.q_steps = self.question_steps

        self.input_x = tf.placeholder(
            tf.int32, [self.batch_size, self.context_steps])
        self.input_y = tf.placeholder(tf.int32, [self.batch_size])
        self.encoded_y = tf.placeholder(
            tf.int32, [self.batch_size, labels_size])

        self.sequence_lengths = tf.placeholder(tf.int32, [self.batch_size])
        with tf.device("/cpu:0"):
          embedding = tf.get_variable(
              "embedding", [self.vocab_size, self.size], dtype=data_type())

        # bidirectional lstm
        context_lstm = BiLSTM(self.input_x, self.sequence_lengths,
                              is_training, config, context_steps, embedding,
                              name='context')
        concat_outputs = context_lstm.outputs[-1]
        self._initial_state = context_lstm._initial_state

        # Use the concatenated hidden states of the final and initial LSTM cells
        # for prediction.
        state_fw = context_lstm.state_fw
        state_bw = context_lstm.state_bw
        hidden_state_fw = context_lstm.state_fw.h
        hidden_state_bw = context_lstm.state_bw.h
        hidden_state = tf.concat(1, (hidden_state_fw, hidden_state_bw))
        print("Shape of the hidden state %s." % hidden_state.get_shape())

        # Transform from hidden size to labels size.
        softmax_w = tf.get_variable(
            "softmax_w", [2*self.size, labels_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [labels_size], dtype=data_type())
        self._logits = tf.matmul(hidden_state, softmax_w) + softmax_b
        print("Shape of the logits %s." % self._logits.get_shape())

        # Cross-entropy loss over final output.
        self._cost = cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self._logits, self.encoded_y))
        self._final_state = state_fw, state_bw

        self._predictions = tf.argmax(self._logits, 1)
        correct_preds = tf.equal(tf.to_int32(self._predictions), self.input_y)
        self._acc = tf.reduce_mean(tf.cast(correct_preds, "float"))

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
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


class Config(object):
    init_scale = 0.1
    learning_rate = 0.01
    max_grad_norm = 5
    hidden_size = 200
    max_epoch = 50
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 32

def run_epoch(session, model, input, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    context_steps = model.context_steps
    all_accs = []
    for j, batch in enumerate(input):

        lc = 30
        lq = 70

        questions, context, choices, labels, map, context_lens, qs_lens = batch

        #context_lens = lc * np.ones(32)
        qs_lens = lq * np.ones(32)

        questions = questions[0].T[:lq].T
        #context = context[0].T[:lc].T
        context = context[0]
        lb = LabelBinarizer()
        lb.fit(model.labels_idx)
        mapped_labels = lb.transform(labels)
        actual_labels = [int(lab[-1]) for lab in labels]

        state = session.run(model.initial_state)

        fetches = {
          "cost": model.cost,
          "final_state": model.final_state,
          "predictions": model.predictions,
          "logits": model.logits,
          "targets": model.targets,
          "acc": model.acc
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        feed_dict = {}
        feed_dict[model.initial_state] = state
        feed_dict[model.input_x] = context
        feed_dict[model.input_y] = actual_labels
        feed_dict[model.q_x] = questions
        feed_dict[model.q_lengths] = qs_lens
        feed_dict[model.encoded_y] = mapped_labels
        feed_dict[model.sequence_lengths] = context_lens

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]
        all_accs.append(vals["acc"])
        if ((verbose) & (j%10==0)):
            print("batch %s; accuracy: %s" % (j, vals["acc"]))
            print("predictions: %s" %vals["predictions"].T)
    return np.mean(all_accs)

def main(_):

    train_path = os.path.join(FLAGS.data_wdw, 'train')
    val_path = os.path.join(FLAGS.data_wdw, 'val')
    test_path = os.path.join(FLAGS.data_wdw, 'test')

    config = Config() 

    print("Loading train data from %s"%train_path)
    train = RawInput(rn.load_data(train_path))

    print("Loading val data from %s"%val_path)
    val = RawInput(rn.load_data(val_path),vocabulary=train.vocab,c_len=train.c_len,\
            q_len=train.q_len)
    if len(train.labels_idx) < len(val.labels_idx):
        print("More validation choices than train")

    print("Loading test data from %s"%test_path)
    test = RawInput(rn.load_data(test_path),vocabulary=train.vocab,c_len=train.c_len,\
            q_len=train.q_len)
    if len(train.labels_idx) < len(test.labels_idx):
        print("More test choices than train")

    q_steps = train.q_len
    c_steps = train.c_len

    
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        print("Loading model..")
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Model(is_training=True, config=config, vocab_size=train.vocab_size,
                             labels_idx=train.labels_idx, context_steps=c_steps,
                             question_steps = 70)
            tf.scalar_summary("Training Loss", m.cost)
            tf.scalar_summary("Learning Rate", m.lr)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_epoch):
                train_iter = rn.batch_iter(
                    train.contexts, train.questions,
                    train.choices, train.labels, train.choices_map, train.context_lens,
                    train.qs_lens, batch_size=config.batch_size,
                    context_num_steps=c_steps,
                    question_num_steps=q_steps)
                lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                val_iter = rn.batch_iter(
                    val.contexts, val.questions,
                    val.choices, val.labels, val.choices_map, val.context_lens,
                    val.qs_lens, batch_size=config.batch_size,
                    context_num_steps=c_steps,
                    question_num_steps=q_steps)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                run_epoch(session, m, train_iter, eval_op=m.train_op,
                          verbose=True)
                print("\nChecking on validation set.")
                val_acc = run_epoch(session, m, val_iter, eval_op=None,
                          verbose=False)
                print("\nAvg. Val Accuracy: %s\n"%val_acc)
            test_iter = rn.batch_iter(
                test.contexts, test.questions,
                test.choices, test.labels, test.choices_map, test.context_lens,
                test.qs_lens, batch_size=config.batch_size,
                context_num_steps=c_steps,
                question_num_steps=q_steps)
            print("\nChecking on test set.")
            test_acc = run_epoch(session, m, test_iter, eval_op=None,
                          verbose=False)
            print("\nAvg. Test Accuracy: %s\n"%test_acc)
            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()

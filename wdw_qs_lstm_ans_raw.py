import os
import time

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from new_reader import batch_iter
from new_reader import build_choices
from new_reader import load_data
from new_reader import pad_eval
from new_reader import get_vocab
from new_reader import vocab_transform

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("data_wdw", 'who_did_what/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_string("checkpoint_path", None,
                    "Model output directory.")
flags.DEFINE_integer("max_epoch", 5,
    "Number of passes over the training data.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("context_steps", 200,
                     "Clip context lengths below this threshold.")
flags.DEFINE_integer("question_steps", 40,
                     "Clip question lengths below this threshold.")
flags.DEFINE_integer("min_freq", 10,
                     "Words lesser than this frequency are mapped to unk.")
flags.DEFINE_integer("qs_size", 300, "Hidden size of the question.")
flags.DEFINE_integer("ans_size", 300, "Size of the answer embedding.")
flags.DEFINE_float("init_scale", 0.1, "Random initialization.")
flags.DEFINE_float("max_grad_norm", 5, "Clips gradient above this.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_float("keep_prob", 1.0, "Dropout.")


FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class RawInput(object):
    def __init__(self, data_bundle, vocabulary=None):
        (contexts, questions, self.choices, self.labels,
            self.choices_map, self.context_lens, self.qs_lens) = data_bundle

        if vocabulary:
            self.vocab = vocabulary
        else:
            self.vocab = get_vocab(
                questions, contexts, min_frequency=FLAGS.min_freq)
        self.vocab_size = len(self.vocab.vocabulary_)

        self.labels_idx = sorted(
            list(set([choice for choices in self.choices for choice in choices]))
        )

        contexts = vocab_transform(contexts, self.vocab)
        self.contexts = pad_eval(contexts, FLAGS.context_steps)

        questions = vocab_transform(questions, self.vocab)
        self.questions = pad_eval(questions, FLAGS.question_steps)


class BiLSTM(object):
    """
    Bidirectional LSTM
    """
    def __init__(self, input_x, sequence_lengths,
                 is_training, embedding, name):
        self.num_steps = FLAGS.question_steps
        self.input_x = input_x
        self.sequence_lengths = sequence_lengths

        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            FLAGS.qs_size, forget_bias=0.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            FLAGS.qs_size, forget_bias=0.0, state_is_tuple=True)
        if is_training and FLAGS.keep_prob < 1:
          lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_fw_cell, output_keep_prob=FLAGS.keep_prob)
          lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_bw_cell, output_keep_prob=FLAGS.keep_prob)

        self._initial_state_fw = lstm_fw_cell.zero_state(
            FLAGS.batch_size, data_type())
        self._initial_state_bw = lstm_bw_cell.zero_state(
            FLAGS.batch_size, data_type())
        self._initial_state = (self._initial_state_fw, self._initial_state_bw)

        inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        if is_training and FLAGS.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)

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
    """WDW model that uses context-based information from the context
    and an embedding for each token from the passage
    """

    def __init__(self, is_training, vocab_size, labels_idx):

        self.labels_idx = labels_idx
        labels_size = len(labels_idx)

        self.vocab_size = vocab_size
        self.context_steps = FLAGS.context_steps
        self.question_steps = FLAGS.question_steps

        self.questions = tf.placeholder(tf.int32, [None, self.question_steps])
        self.enc_y = tf.placeholder(tf.int32, [None])
        self.bin_y = tf.placeholder(tf.int32, [None, labels_size])
        self.ques_lengths = tf.placeholder(tf.int32, [None])

        ques_embedding = tf.get_variable(
            "ques_embedding", [self.vocab_size, FLAGS.qs_size],
            dtype=data_type())

        # bidirectional lstm
        ques_lstm = BiLSTM(self.questions, self.ques_lengths,
                           is_training, ques_embedding, name='ques')
        ques_outputs = ques_lstm.outputs[-1]
        self._initial_state = ques_lstm._initial_state

        # Use the concatenated hidden states of the final and initial LSTM cells
        # for prediction.
        state_fw = ques_lstm.state_fw
        state_bw = ques_lstm.state_bw
        hidden_state_fw = state_fw.h
        hidden_state_bw = state_bw.h
        hidden_state = tf.concat(1, (hidden_state_fw, hidden_state_bw))
        print("Shape of the hidden state %s." % hidden_state.get_shape())

        self.contexts = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.context_steps])
        self.cont_lengths = tf.placeholder(tf.int32, shape=[None])

        # Shape = (batch_size X context_length X ans_hidden_size)
        context_embedding = tf.get_variable(
            "context_embedding",
            [self.vocab_size, FLAGS.ans_size], dtype=tf.float32)
        context_transformed = tf.nn.embedding_lookup(
            context_embedding, self.contexts)

        bilinear = tf.get_variable(
            "bilinear",
            [2*FLAGS.qs_size, FLAGS.ans_size], dtype=tf.float32)
        softmax_W = tf.get_variable(
            "softmax_W", [FLAGS.ans_size, labels_size])
        softmax_b = tf.get_variable("softmax_b", [labels_size])

        # Shape = (batch_size X ans_hidden_size)
        ques_transform = tf.matmul(hidden_state, bilinear)

        self._logits = []
        self._predictions = []
        ques_batches = tf.unpack(ques_transform, axis=0)
        ans_batches = tf.unpack(context_transformed, axis=0)
        for q_b, a_b in zip(ques_batches, ans_batches):
            tmp = tf.matmul(tf.expand_dims(q_b, dim=0), tf.transpose(a_b))[0]
            att = tf.expand_dims(tf.nn.softmax(tmp), 0)
            cont_final = tf.matmul(att, a_b)
            self._logits.append(
                tf.add(tf.matmul(cont_final, softmax_W), softmax_b)[0])

        self._cost = cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self._logits, self.bin_y))
        self._predictions = tf.argmax(self._logits, 1)
        print(self._predictions.get_shape())
        correct_preds = tf.equal(tf.to_int32(self._predictions), self.enc_y)
        self._acc = tf.reduce_mean(tf.cast(correct_preds, "float"))

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          FLAGS.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))


    @property
    def acc(self):
        return self._acc

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
        return self.enc_y

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, model=None, batches=None, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    all_accs = []
    all_costs = []

    lb = LabelBinarizer()
    lb.fit(model.labels_idx)

    le = LabelEncoder()
    le.fit(model.labels_idx)

    for j, batch in enumerate(batches):

        questions, contexts, choices, labels, _, context_lens, qs_lens = batch

        bin_labels = lb.transform(labels)
        enc_labels = le.transform(labels)

        fetches = {
          "cost": model.cost,
          "predictions": model.predictions,
          "logits": model.logits,
          "targets": model.targets,
          "acc": model.acc
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        feed_dict = {}
        feed_dict[model.questions] = questions
        feed_dict[model.contexts] = contexts
        feed_dict[model.enc_y] = enc_labels
        feed_dict[model.bin_y] = bin_labels
        feed_dict[model.ques_lengths] = np.minimum(qs_lens, FLAGS.question_steps)
        feed_dict[model.cont_lengths] = np.minimum(context_lens, FLAGS.context_steps)

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        all_costs.append(vals["cost"])
        all_accs.append(vals["acc"])
        if ((verbose) & (j%10==0)):
            print("batch %s; accuracy: %s" % (j, vals["acc"]))
            print("cost %s" % vals["cost"])
            print("predictions: %s" % vals["predictions"].T)
            print("targets: %s" % vals["targets"])
    return np.mean(all_costs), np.mean(all_accs)


def main():
    train_path = os.path.join(FLAGS.data_wdw, 'test')
    val_path = os.path.join(FLAGS.data_wdw, 'test')
    # test_path = os.path.join(FLAGS.data_wdw, 'test')

    print("Loading train data from %s" % train_path)
    train = RawInput(load_data(train_path))

    print("Loading val data from %s"%val_path)
    val = RawInput(load_data(val_path), vocabulary=train.vocab)
    if len(train.labels_idx) < len(val.labels_idx):
        print("More validation choices than train")
    #
    # print("Loading test data from %s"%test_path)
    # test = RawInput(load_data(test_path), vocabulary=train.vocab)
    # if len(train.labels_idx) < len(test.labels_idx):
    #     print("More test choices than train")

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                                                    FLAGS.init_scale)
        print("Loading model..")
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Model(is_training=True, vocab_size=train.vocab_size,
                          labels_idx=train.labels_idx)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(FLAGS.max_epoch):
                train_iter = batch_iter(
                    train.contexts, train.questions,
                    train.choices, train.labels, train.choices_map,
                    train.context_lens,
                    train.qs_lens, batch_size=FLAGS.batch_size,
                    context_num_steps=FLAGS.context_steps,
                    question_num_steps=FLAGS.question_steps)

                val_iter = batch_iter(
                    val.contexts, val.questions,
                    val.choices, val.labels, val.choices_map, val.context_lens,
                    val.qs_lens, batch_size=FLAGS.batch_size,
                    context_num_steps=FLAGS.context_steps,
                    question_num_steps=FLAGS.question_steps)

                print("Epoch: %d" % (i + 1))
                run_epoch(session, m, train_iter, eval_op=m.train_op,
                          verbose=True)
                print("Checking on validation set.")
                ave_cost, ave_acc = run_epoch(
                    session, m, val_iter, eval_op=None, verbose=False)
                print("Avg. Val Accuracy: %s" % ave_acc)
                print("Avg. Vac Cost: %s" % ave_cost)


if __name__ == "__main__":
    main()

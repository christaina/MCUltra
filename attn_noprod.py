from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import logging as lg
import reader as rn

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", '../who_did_what/Strict',
                    "Where the training/test data is stored.")
flags.DEFINE_string("glove_path", '/scratch/ceb545/nlp/project/glove.6B.300d.txt',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", './dump',
                    "checkpoints.")
flags.DEFINE_float("init_scale", 0.5, "uniform initialization scale.")
flags.DEFINE_float("learning_rate", 0.001, "Adam optimizer learning rate.")
flags.DEFINE_float("grad_norm", 5.0, "Clip the gradient.")
flags.DEFINE_integer("hidden_size", 150, "Hidden size of qs / ans RNN.")
flags.DEFINE_integer("embed_size", 300, "Embedding size.")
flags.DEFINE_integer("max_epoch", 30, "Max number of epochs.")
flags.DEFINE_float("keep_prob", 1.0, "Dropout probability.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("ckpt_steps", 5, "Checkpoint every ckpt_steps")
flags.DEFINE_bool("testing", False, "Run on test data")
flags.DEFINE_bool("use_glove", False, "Use pre-trained glove embeddings")
flags.DEFINE_bool("clear_save", False, "Delete checkpoints")
flags.DEFINE_bool("use_gru", False, "Use gru cells")
flags.DEFINE_bool("use_sgd", False, "Use SGD instead of Adam optimizer")


FLAGS = flags.FLAGS

class RawInput(object):
    def __init__(self, data_bundle, vocabulary=None):#, c_len=None, q_len=None):
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
        print(self.contexts[3])
        print(self.questions[3])
        print(self.labels[3])
        print(self.choices_map[3])
        self.transformed_labels_idx = [x[0] for x in list(self.vocab.transform(self.labels_idx))]
        print(self.transformed_labels_idx)

        self.contexts = rn.vocab_transform(self.contexts, self.vocab)
        self.questions = rn.vocab_transform(self.questions, self.vocab)


class BiLSTM(object):
    """
    Bidirectional LSTM
    """
    def __init__(self, input_x, sequence_lengths, embedding,size, keep_prob,name):

        self.input_x = input_x
        self.sequence_lengths = sequence_lengths
        if FLAGS.use_gru:
            lstm_fw_cell = tf.nn.rnn_cell.GRUCell(
                size)
            lstm_bw_cell = tf.nn.rnn_cell.GRUCell(
                size)
        else:

            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True)
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True)
        #lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
        #  lstm_fw_cell, output_keep_prob=keep_prob)
        #lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
        #  lstm_bw_cell, output_keep_prob=keep_prob)
        self._initial_state_fw = lstm_fw_cell.zero_state(
            FLAGS.batch_size, tf.float32)
        self._initial_state_bw = lstm_bw_cell.zero_state(
            FLAGS.batch_size, tf.float32)
        self._initial_state = (self._initial_state_fw, self._initial_state_bw)
        inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        (outputs_fw, outputs_bw), (self.state_fw, self.state_bw) = tf.nn.bidirectional_dynamic_rnn(
            lstm_fw_cell,
            lstm_bw_cell,
            inputs,
            initial_state_fw=self._initial_state[0],
            initial_state_bw=self._initial_state[1],
            sequence_length=self.sequence_lengths, scope="BiRNN_%s" % name)
        if FLAGS.use_gru:
            self._final_state = (self.state_fw, self.state_bw)
            self._state = tf.concat(1, (self.state_fw, self.state_bw))
        else:
            self._final_state = (self.state_fw.h, self.state_bw.h)
            self._state = tf.concat(1, (self.state_fw.h, self.state_bw.h))
        self._outputs = tf.concat(2, (outputs_fw, outputs_bw))


class Model(object):
    """The PTB model."""

    def __init__(self, vocab_size, choices_idx,pre_embedding=None):
        """
        choices_idx: Index in the vocabulary corresponding to choices.
        """
        n_choices = len(choices_idx)
        self.choices_idx = choices_idx
        batch_size = FLAGS.batch_size
        size = FLAGS.hidden_size
        embed_size = FLAGS.embed_size
        self.vocab_size = vocab_size
        if FLAGS.use_glove:
            size = pre_embedding.shape[1]/2
            embed_size = pre_embedding.shape[1]
            pre_embedding = tf.convert_to_tensor(pre_embedding,dtype=tf.float32)

        self.keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")
        self.q_lengths = tf.placeholder(tf.int32, [batch_size])
        self.q_x = tf.placeholder(tf.int32, [batch_size, None])
        print("qu shape: %s" % self.q_x.get_shape())

        self.c_x = tf.placeholder(tf.int32, [batch_size, None])
        c_lengths = self.c_lengths = tf.placeholder(tf.int32, [batch_size])

        self.enc_y = tf.placeholder(tf.int64, [batch_size])
        self.bin_y = tf.placeholder(tf.int32, [batch_size, n_choices])

        # the number of choices
        # self.choices = tf.placeholder(tf.bool, [self.batch_size, n_choices])

        if FLAGS.use_glove:
            print("Using glove embedding.")
            train_embedding = tf.get_variable(
                "pre_embedding", [self.vocab_size, embed_size], dtype=tf.float32)

            embedding = tf.select(tf.equal(tf.reduce_sum(tf.abs(pre_embedding),1),0),\
                    train_embedding,pre_embedding)
        else:
            print("Using own embedding.")
            embedding = tf.get_variable(
                "embedding", [self.vocab_size, FLAGS.embed_size], dtype=tf.float32)
        print('embedding shape : %s'%embedding.get_shape())
        embedding_drop = tf.nn.dropout(embedding,self.keep_prob)

        choices_embedding = tf.nn.embedding_lookup(embedding, self.choices_idx)
        print(choices_embedding.get_shape())

        # bidirectional lstm - context
        context_lstm = BiLSTM(self.c_x, self.c_lengths, embedding,\
                size,self.keep_prob,name='context')
        c_outputs = context_lstm._outputs

        # bidirectional lstm - question
        question_lstm = BiLSTM(self.q_x, self.q_lengths, embedding,\
                size,self.keep_prob,name="question")
        q_state = question_lstm._state

       # print("state shape: %s" % question_lstm.state_fw.h.get_shape())

        self._initial_state = context_lstm._initial_state, question_lstm._initial_state
        self._final_state = context_lstm._final_state, question_lstm._final_state

        # Attention calculation.
        bilinear_weights = tf.ones([2*size, 2*size])
        bilinear_weights = tf.get_variable(
            "bilinear_w", [2*size, 2*size], dtype=tf.float32)

        losses = []
        predictions = []
        softmax_weights = tf.get_variable(
            "softmax_w", [size*2,n_choices], dtype=tf.float32)

        for i in range(batch_size):
            curr_c = tf.transpose(c_outputs[i, :c_lengths[i], :])
            curr_q = tf.expand_dims(q_state[i], dim=0)
            att_weights = tf.nn.softmax(
                tf.matmul(curr_q, tf.matmul(bilinear_weights, curr_c)), dim=-1)
            context_vector = tf.matmul(att_weights, tf.transpose(curr_c))
            logits = tf.matmul(context_vector, softmax_weights)[0]
            predictions.append(tf.argmax(logits, 0))
            losses.append(
                tf.nn.softmax_cross_entropy_with_logits(logits, self.bin_y[i]))

        self._predictions = predictions
        self._acc = tf.reduce_mean(
            tf.cast(tf.equal(predictions, self.enc_y), dtype=tf.float32))
        # Cross-entropy loss over final output.
        self._cost = cost = tf.reduce_mean(losses)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          FLAGS.grad_norm)
        if FLAGS.use_sgd:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

    @property
    def acc(self):
        return self._acc

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op


def mask_choices(indices, choices):
    choices_mask = [np.zeros(len(indices)) for x in choices]
    for i, x in enumerate(choices_mask):
        curr = x
        for j, choice in enumerate(choices[i].split(" ")):
            if choice in indices:
                curr[indices.index(choice)] = 1
        choices_mask[i] = curr
    return np.array(choices_mask) > 0


def run_epoch(session, model, input, train_op=None, verbose=False,
              vocab=None,is_testing=False):
    """Runs the model on the given data."""
    start_time = time.time()

    lb = LabelBinarizer()
    entities = ["@entity" + str(i) for i in range(5)]
    lb.fit(entities)

    le = LabelEncoder()
    le.fit(entities)

    choices_idx  = model.choices_idx
    ent_ch_dict = dict(zip(entities, choices_idx))

    all_accs = []
    all_costs = []
    for j, batch in enumerate(input):

        questions, context, choices, labels, choices_map, context_lens, qs_lens = batch

        # TODO: Provide choices.
        enc_labels = le.transform(labels)
        bin_labels = lb.transform(labels)

        fetches = {"cost": model.cost, "acc": model.acc}

        if train_op is not None:
            fetches["train_op"] = train_op

        feed_dict = {}
        feed_dict[model.c_x] = context
        feed_dict[model.c_lengths] = context_lens
        feed_dict[model.q_x] = questions
        feed_dict[model.q_lengths] = qs_lens
        feed_dict[model.enc_y] = enc_labels
        feed_dict[model.bin_y] = bin_labels
        if is_testing:
            feed_dict[model.keep_prob] = 1.0
        else:
            feed_dict[model.keep_prob] = FLAGS.keep_prob
        vals = session.run(fetches, feed_dict)

        all_accs.append(vals["acc"])
        all_costs.append(vals["cost"])
    return np.mean(all_costs), np.mean(all_accs)


def main(_):

    data_path = FLAGS.data_path
    if ((FLAGS.clear_save) & (os.path.exists(FLAGS.save_path))):
        shutil.rmtree(FLAGS.save_path)
    if FLAGS.testing:
        train_path = os.path.join(data_path, 'test')
    else:
        train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    test_path = os.path.join(data_path, 'test')

    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    log_fi = os.path.join(FLAGS.save_path,'output.log')
    lg.basicConfig(filename=log_fi,level=lg.DEBUG,\
    format='%(asctime)s %(message)s')

    # print("Loading train data from %s" % train_path)
    train = RawInput(rn.load_data(train_path))

    print("Loading val data from %s"%val_path)
    val = RawInput(rn.load_data(val_path), vocabulary=train.vocab)

    print("Loading test data from %s" % test_path)
    test = RawInput(rn.load_data(test_path), vocabulary=train.vocab)

    if FLAGS.glove_path:
        embedding = rn.glove_embedding(FLAGS.glove_path,train.vocab)
    else:
        embedding = None

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                                                    FLAGS.init_scale)
        print("Loading model..")
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                if FLAGS.use_glove:
                    m = Model(vocab_size=train.vocab_size,
                          choices_idx=train.transformed_labels_idx,pre_embedding=embedding)
                else:
                    m = Model(vocab_size=train.vocab_size,
                          choices_idx=train.transformed_labels_idx)


        with tf.Session() as session:
            saver = tf.train.Saver(tf.all_variables())
            ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
                print("Loading parameters from %s" % ckpt.model_checkpoint_path)
                lg.info("Loading parameters from %s" % ckpt.model_checkpoint_path)
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("New session.")
                lg.info("New session.")
                session.run(tf.initialize_all_variables())
            all_st = time.time()
            for i in range(FLAGS.max_epoch):
                train_iter = rn.batch_iter(
                    train.contexts, train.questions,
                    train.choices, train.labels, train.choices_map, train.context_lens,
                    train.qs_lens, batch_size=FLAGS.batch_size)
                train_cost, train_acc = run_epoch(
                    session, m, train_iter, train_op=m.train_op, verbose=False,
                    vocab=train.vocab)
                print("Train cost: after " + str(i) + " epoch is " + str(train_cost))
                print("Train acc: after " + str(i) +  " epoch is " + str(train_acc))
                lg.info("Train cost: after " + str(i) + " epoch is " + str(train_cost))
                lg.info("Train acc: after " + str(i) +  "epoch is " + str(train_acc))

                if i % FLAGS.ckpt_steps == 0:
                    checkpoint_path = os.path.join(FLAGS.save_path, "wdw.ckpt")
                    saver.save(session, checkpoint_path, global_step=i)

                val_iter = rn.batch_iter(
                    val.contexts, val.questions,
                    val.choices, val.labels, val.choices_map, val.context_lens,
                    val.qs_lens, batch_size=FLAGS.batch_size)
                val_cost, val_acc = run_epoch(
                    session, m, val_iter, train_op=None, verbose=False,
                    vocab=train.vocab,is_testing=True)
                lg.info("Val cost: after " + str(i) + " epoch is " + str(val_cost))
                lg.info("Val acc: after " + str(i) + " epoch is " + str(val_acc))
                print("Val cost: after " + str(i) + " epoch is " + str(val_cost))
                print("Val acc: after " + str(i) + " epoch is " + str(val_acc))

                test_iter = rn.batch_iter(
                    test.contexts, test.questions,
                    test.choices, test.labels, test.choices_map, test.context_lens,
                    test.qs_lens, batch_size=FLAGS.batch_size)
                print("Checking on test set.")
                test_cost, test_acc = run_epoch(session, m, test_iter, train_op=None,
                                                verbose=False, vocab=train.vocab,is_testing=True)

                test_str = ("Test Accuracy: %s\n" % test_acc)
                print(test_str)
                lg.info(test_str)

if __name__ == "__main__":
    tf.app.run()

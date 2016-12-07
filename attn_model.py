from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import shutil

import numpy as np
#import bidir_rnn
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.preprocessing import LabelBinarizer
import reader as rn

flags = tf.flags
logging = tf.logging

# flags.DEFINE_string("data_wdw", '/home/manoj/oogie-boogie/wdw',
#                     "Where the training/test data is stored.")
# flags.DEFINE_string("save_path", './runs/dump',
#                     "Model output directory.")
# flags.DEFINE_string("checkpoint_path", './runs/dump',
#                     "Model output directory.")
# flags.DEFINE_bool("use_fp16", False,
#                   "Train using 16-bit floats instead of 32bit floats")
#
# FLAGS = flags.FLAGS


def data_type():
  return  tf.float32

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
        self.transformed_labels_idx = [x[0] for x in list(self.vocab.transform(self.labels_idx))]
        print(self.transformed_labels_idx)

        self.contexts = rn.vocab_transform(self.contexts, self.vocab)
        self.questions = rn.vocab_transform(self.questions, self.vocab)

        # TODO: choices embedding
        # if c_len:
        #     self.contexts = rn.pad_eval(self.contexts, c_len)
        # if q_len:
        #     self.questions = rn.pad_eval(self.questions, q_len)
        # else:
        #     self.c_len = len(self.contexts[0])
        #     self.q_len = len(self.questions[0])


class BiLSTM(object):
    """
    Bidirectional LSTM
    """
    def __init__(self, input_x, keep_prob, sequence_lengths,
                 config, embedding, name, is_tuple=False,
                 num_steps=40):

        self.batch_size = config.batch_size
        self.size = config.hidden_size

        self.input_x = input_x
        self.keep_prob = keep_prob
        self.sequence_lengths = sequence_lengths
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.size, forget_bias=0.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.size, forget_bias=0.0, state_is_tuple=True)
        """
        if config.keep_prob < 1:
          lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_fw_cell, output_keep_prob=config.keep_prob)
          lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_bw_cell, output_keep_prob=config.keep_prob)
        """
        self._initial_state_fw = lstm_fw_cell.zero_state(
            self.batch_size, data_type())
        self._initial_state_bw = lstm_bw_cell.zero_state(
            self.batch_size, data_type())
        self._initial_state = (self._initial_state_fw, self._initial_state_bw)
        inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        inputs = tf.nn.dropout(inputs, self.keep_prob)
        print('first shape %s' % inputs.get_shape())

        (outputs_fw, outputs_bw), (self.state_fw, self.state_bw) = tf.nn.bidirectional_dynamic_rnn(
            lstm_fw_cell,
            lstm_bw_cell,
            inputs,
            initial_state_fw=self._initial_state[0],
            initial_state_bw=self._initial_state[1],
            sequence_length=self.sequence_lengths, scope="BiRNN_%s" % name)
        self._final_state = (self.state_fw.h, self.state_bw.h)
        self._state = tf.concat(1, (self.state_fw.h, self.state_bw.h))
        self._outputs = tf.concat(2, (outputs_fw, outputs_bw))


class Model(object):
    """The PTB model."""

    def __init__(self, config, vocab_size, choices_idx, keep_prob):
        """
        choices_idx: Index in the vocabulary corresponding to choices.
        """
        n_choices = len(choices_idx)
        self.choices_idx = choices_idx
        batch_size = self.batch_size = config.batch_size
        size = self.size = config.hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = config.embedding_size
        self.keep_prob = keep_prob

        self.q_x = tf.placeholder(tf.int32, [self.batch_size, None])
        self.q_lengths = tf.placeholder(tf.int32, [self.batch_size])
        print("qu shape: %s" % self.q_x.get_shape())

        self.c_x = tf.placeholder(tf.int32, [self.batch_size, None])
        c_lengths = self.c_lengths = tf.placeholder(tf.int32, [self.batch_size])

        self.enc_y = tf.placeholder(tf.int32, [self.batch_size])
        self.bin_y = tf.placeholder(tf.int32, [self.batch_size, n_choices])

        # the number of choices
        # self.choices = tf.placeholder(tf.bool, [self.batch_size, n_choices])

        embedding = tf.get_variable(
            "embedding", [self.vocab_size, self.embedding_size], dtype=data_type())
        choices_embedding = tf.nn.embedding_lookup(embedding, self.choices_idx)
        print(choices_embedding.get_shape())

        # bidirectional lstm - context
        context_lstm = BiLSTM(
            self.c_x, self.keep_prob, self.c_lengths, config, embedding,
            name='context')
        c_outputs = context_lstm._outputs

        # bidirectional lstm - question
        question_lstm = BiLSTM(
            self.q_x, self.keep_prob, self.q_lengths, config, embedding,
            name="question")
        q_state = question_lstm._state
        print("state shape: %s" % question_lstm.state_fw.h.get_shape())

        self._initial_state = context_lstm._initial_state, question_lstm._initial_state
        self._final_state = context_lstm._final_state, question_lstm._final_state

        # Attention calculation.
        bilinear_weights = tf.ones([2*size, 2*size])
        bilinear_weights = tf.get_variable(
            "bilinear_w", [2*size, 2*size], dtype=data_type())

        losses = []
        for i in range(batch_size):
            curr_c = tf.transpose(c_outputs[i, :c_lengths[i], :])
            curr_q = tf.expand_dims(q_state[i], dim=0)
            att_weights = tf.matmul(curr_q, tf.matmul(bilinear_weights, curr_c))
            context_vector = tf.matmul(att_weights, tf.transpose(curr_c))
            logits = tf.matmul(context_vector, tf.transpose(choices_embedding))[0]
            losses.append(
                tf.nn.softmax_cross_entropy_with_logits(logits, self.bin_y[i]))

        # Cross-entropy loss over final output.
        self._cost = cost = tf.reduce_mean(losses)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def test(self):
        return self._test

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def acc(self):
        return self._acc

    @property
    def predictions(self):
        return self._predictions

    @property
    def cost(self):
        return self._cost

    @property
    def logits(self):
        return self._attn_logits

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
    learning_rate = 0.001
    max_grad_norm = 10
    hidden_size = 150
    embedding_size = 300
    max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 32

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
              vocab=None):
    """Runs the model on the given data."""
    start_time = time.time()

    lb = LabelBinarizer()
    entities = ["@entity" + str(i) for i in range(5)]
    lb.fit(entities)

    choices_idx  = model.choices_idx
    ent_ch_dict = dict(zip(entities, choices_idx))

    all_accs = []
    for j, batch in enumerate(input):

        questions, context, choices, labels, choices_map, context_lens, qs_lens = batch

        # TODO: Provide choices.
        enc_labels = [ent_ch_dict[l] for l in labels]
        bin_labels = lb.transform(labels)

        fetches = {
          "cost": model.cost
          }
        if train_op is not None:
            fetches["train_op"] = train_op

        feed_dict = {}
        feed_dict[model.c_x] = context
        feed_dict[model.c_lengths] = context_lens
        feed_dict[model.q_x] = questions
        feed_dict[model.q_lengths] = qs_lens
        feed_dict[model.enc_y] = enc_labels
        feed_dict[model.bin_y] = bin_labels

        vals = session.run(fetches, feed_dict)
        print(vals["cost"])
    #     all_accs.append(vals["acc"])
    #     if ((verbose) & (j%10==0)):
    #         print("batch %s; accuracy: %s" % (j, vals["acc"]))
    #         #print(vals['logits'])
    #         #print(vals['test'])
    #         print("predictions: %s" %vals["predictions"].T)
    #         print("labels: %s"%actual_labels)
    #         #print("mapped labels: %s"%mapped_labels)
    # return np.mean(all_accs)


# def main(_):
# if (os.path.exists(FLAGS.save_path)):
#     shutil.rmtree(FLAGS.save_path)
# os.makedirs(FLAGS.save_path)
# t_log = open(os.path.join(FLAGS.save_path, 'train.txt'),'w')
# v_log = open(os.path.join(FLAGS.save_path, 'val.txt'),'w')
# te_log = open(os.path.join(FLAGS.save_path, 'test.txt'),'w')

data_path = "/home/manoj/oogie-boogie/wdw"
train_path = os.path.join(data_path, 'test')
val_path = os.path.join(data_path, 'test')
test_path = os.path.join(data_path, 'test')

config = Config()
print("Loading train data from %s" % train_path)
train = RawInput(rn.load_data(train_path))

# print("Loading val data from %s"%val_path)
# val = RawInput(rn.load_data(val_path),vocabulary=train.vocab,c_len=train.c_len,\
#         q_len=train.q_len)
# if len(train.labels_idx) < len(val.labels_idx):
#     print("More validation choices than train")
#
# print("Loading test data from %s"%test_path)
# test = RawInput(rn.load_data(test_path),vocabulary=train.vocab,c_len=train.c_len,\
#         q_len=train.q_len)
# if len(train.labels_idx) < len(test.labels_idx):
#     print("More test choices than train")


with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    print("Loading model..")
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = Model(config=config, vocab_size=train.vocab_size,
                      choices_idx=train.transformed_labels_idx,
                      keep_prob=config.keep_prob)

        #tf.scalar_summary("Training Loss", m.cost)
        #tf.scalar_summary("Accuracy",m.acc)
        #tf.scalar_summary("Learning Rate", m.lr)

    # sv = tf.train.Supervisor(logdir="/home/manoj")
    with tf.Session() as session:
        all_st = time.time()
        for i in range(config.max_epoch):
            train_iter = rn.batch_iter(
                train.contexts, train.questions,
                train.choices, train.labels, train.choices_map, train.context_lens,
                train.qs_lens, batch_size=config.batch_size)
            session.run(tf.initialize_all_variables())
            train_acc = run_epoch(
                session, m, train_iter, train_op=m.train_op, verbose=True,
                vocab=train.vocab)
#
#             val_iter = rn.batch_iter(
#                 val.contexts, val.questions,
#                 val.choices, val.labels, val.choices_map, val.context_lens,
#                 val.qs_lens, batch_size=config.batch_size,
#                 context_num_steps=c_steps,
#                 question_num_steps=q_steps)
#
#             print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
#             st = time.time()
#             print("Epoch time: %s"%(time.time()-st))
#             t_log.write("%s,%s,%s\n"%(i,time.time()-st,train_acc))
#             print("\nChecking on validation set.")
#             st = time.time()
#             val_acc = run_epoch(session, m, val_iter, eval_op=None,
#                       verbose=False)
#             print("\nAvg. Val Accuracy: %s\n"%val_acc)
#             v_log.write("%s,%s,%s\n"%(i,time.time()-st,val_acc))
#             print("Saving model to %s." % FLAGS.save_path)
#             sv.saver.save(session, os.path.join(FLAGS.save_path,'model'),\
#                     global_step=sv.global_step)
#                 #saver.save(session,FLAGS.save_path,global_step=sv.global_step)
#         test_iter = rn.batch_iter(
#             test.contexts, test.questions,
#             test.choices, test.labels, test.choices_map, test.context_lens,
#             test.qs_lens, batch_size=config.batch_size,
#             context_num_steps=c_steps,
#             question_num_steps=q_steps)
#         print("\nChecking on test set.")
#         test_acc = run_epoch(session, m, test_iter, eval_op=None,
#                       verbose=False)
#         te_log.write("%s,%s\n"%(time.time()-all_st,test_acc))
#         print("\nAvg. Test Accuracy: %s\n"%test_acc)
#         te_log.close()
#         v_log.close()
#         t_log.close()

# if __name__ == "__main__":
#   tf.app.run()

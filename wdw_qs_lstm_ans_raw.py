import numpy as np
import tensorflow as tf

from reader import batch_iter
from reader import build_choices
from reader import load_data
from reader import get_vocab

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("data_path", None, "Where the training/test data is stored.")
flags.DEFINE_integer("num_epochs", 5,
    "Number of passes over the training data.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("num_steps", 20, "Number of steps to be unrolled.")
flags.DEFINE_integer("qs_hidden_size", 300, "Hidden size of the question.")

FLAGS = flags.FLAGS


class QuesLSTMAnsEmbedding(object):
    def __init__(self, vocab_size, is_training=True):
        batch_size = FLAGS.batch_size
        num_steps = FLAGS.num_steps
        hidden_size = FLAGS.qs_hidden_size

        self.question = tf.placeholder(
            tf.int32, shape=[batch_size, num_steps])
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # Converts from batch_size X num_steps to
        # num_steps X batch_size X hidden_size
        qs_embedding = tf.get_variable(
            "qs_embedding",
            [vocab_size + 1, hidden_size], dtype=tf.float32)
        inputs = []
        for num_step in range(FLAGS.num_steps):
            inputs.append(tf.nn.embedding_lookup(
                qs_embedding, self.question[:, num_step]))

        self.sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
        outputs, self.final_state = tf.nn.rnn(
            cell, inputs, dtype=tf.float32, sequence_length=self.sequence_length)

    def get_attention(self):
        self.question_embedding = tf.placeholder(
            tf.int32, shape=[FLAGS.qs_hidden_size])


def run_epoch(data_iter, model, session):
    for (qs_batch, ans_batch, choice_batch, lab_batch, map_batch,
         cont_len, qs_len) in data_iter:

        print(qs_len)
        print(cont_len)
        var_qs_len = FLAGS.num_steps * np.ones(FLAGS.batch_size)

        state = session.run(model.initial_state)

        # Obtain question embedding from the last-step of the LSTM.
        for step_epoch, qs_step in enumerate(qs_batch):

            curr_step = step_epoch * var_qs_len
            seq_lengths = np.maximum(0, np.minimum(FLAGS.num_steps, qs_len - curr_step))
            feed_dict = {
                model.question: qs_step,
                model.sequence_length: seq_lengths,
                model.initial_state: state}
            state = session.run(model.final_state, feed_dict)

        break


def main():
    if not FLAGS.data_path:
        raise ValueError("Must set data_path to the data directory.")
    data_path = FLAGS.data_path
    contexts, questions, choices, labels, choices_map, \
        context_lens, qs_lens = load_data(FLAGS.data_path)
    vocabulary = get_vocab(questions, contexts, min_frequency=10)
    vocab_size = len(vocabulary.vocabulary_)
    all_choices = build_choices(choices)

    with tf.variable_scope("rc_model", reuse=None):
        model = QuesLSTMAnsEmbedding(vocab_size=vocab_size, is_training=True)

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    for i in range(FLAGS.num_epochs):
        data_iter = batch_iter(
            contexts, questions, choices, labels, choices_map,
            context_lens, qs_lens,
            batch_size=FLAGS.batch_size,
            num_epochs=FLAGS.num_epochs, random_state=i,
            vocabulary=vocabulary
        )
        run_epoch(data_iter, model, session)

if __name__ == "__main__":
    main()

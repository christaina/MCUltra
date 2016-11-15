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
flags.DEFINE_integer("hidden_size", 300, "Hidden size.")

FLAGS = flags.FLAGS

class QuesLSTMAnsEmbedding(object):
    def __init__(self, is_training=True):
        batch_size = FLAGS.batch_size
        num_steps = FLAGS.num_steps
        hidden_size = FLAGS.hidden_size

        self.question = tf.placeholder(
            tf.int32, shape=[batch_size, num_steps])
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        self.initial_state = cell.zero_state(batch_size, tf.float32)

def run_epoch(data_iter, model, session):
    for (qs_batch, ans_batch, choice_batch, lab_batch, map_batch,
         cont_len, qs_len) in data_iter:

        print(qs_len)
        print(cont_len)
        state = model.initial_state
        for qs_step in qs_batch:
            feed_dict = {
                model.question: qs_step
                }
            question = session.run(model.question, feed_dict)
            print(question)
        break

def main():
    if not FLAGS.data_path:
        raise ValueError("Must set data_path to the data directory.")
    data_path = FLAGS.data_path
    contexts, questions, choices, labels, choices_map, \
        context_lens, qs_lens = load_data(FLAGS.data_path)
    vocabulary = get_vocab(questions, contexts, min_frequency=10)
    all_choices = build_choices(choices)

    with tf.variable_scope("rc_model", reuse=True):
        model = QuesLSTMAnsEmbedding()

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

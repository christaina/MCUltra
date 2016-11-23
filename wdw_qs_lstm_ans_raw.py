import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from reader import batch_iter
from reader import build_choices
from reader import load_data
from reader import get_vocab
from reader import vocab_transform


flags = tf.flags
logging = tf.logging
flags.DEFINE_string("data_path", None, "Where the training/test data is stored.")
flags.DEFINE_integer("num_epochs", 5,
    "Number of passes over the training data.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("num_steps", 20, "Number of steps to be unrolled.")
flags.DEFINE_integer("qs_hidden_size", 300, "Hidden size of the question.")
flags.DEFINE_integer("ans_dim", 300, "Dimensionality of the answer.")
flags.DEFINE_float("init_scale", 0.1, "Random initialization.")
flags.DEFINE_float("max_grad_norm", 5, "Clips gradient above this.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")


FLAGS = flags.FLAGS


class QuesLSTMAnsEmbedding(object):
    def __init__(self, vocab_size, all_choices, batch_size):
        self.len_choices = len(all_choices)
        num_steps = FLAGS.num_steps
        hidden_size = FLAGS.qs_hidden_size
        self.vocab_size = vocab_size

        self.batch_size = batch_size
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
        outputs, hidden_state = tf.nn.rnn(
            cell, inputs, dtype=tf.float32, sequence_length=self.sequence_length)
        hidden_state = hidden_state.h

        self.context = tf.placeholder(tf.int32, shape=[batch_size, None])
        self.context_length = tf.placeholder(tf.int32, shape=[batch_size])
        self.choices = tf.placeholder(tf.int32, shape=[batch_size, self.len_choices])
        self.labels = tf.placeholder(tf.int32, shape=[batch_size, self.len_choices])

        answer_dim = FLAGS.ans_dim
        answer_embedding = tf.get_variable(
            "answer_embedding",
            [self.vocab_size + 1, answer_dim], dtype=tf.float32)
        bilinear = tf.get_variable(
            "bilinear",
            [FLAGS.qs_hidden_size, answer_dim], dtype=tf.float32)
        softmax_W = tf.get_variable("softmax_W", [answer_dim, self.len_choices])
        softmax_b = tf.get_variable("softmax_b", [self.len_choices])
        self.qs_embedding = qs_embedding.get_shape()

        # Calculate attentions.
        losses = []
        correct = []
        for num_batch in range(batch_size):
            curr_qs_emb = qs_embedding[num_batch]
            curr_ans = self.context[num_batch, :self.context_length[num_batch]]
            curr_labels = self.labels[num_batch]

            ans_embed = tf.nn.embedding_lookup(answer_embedding, curr_ans)
            attentions = tf.nn.softmax(tf.matmul(
                tf.expand_dims(curr_qs_emb, dim=0),
                tf.matmul(bilinear, tf.transpose(ans_embed))))
            ans_repres = tf.matmul(attentions, ans_embed)
            raw_preds = tf.add(tf.matmul(ans_repres, softmax_W), softmax_b)
            correct.append(tf.equal(
                tf.argmax(raw_preds, 0),
                tf.argmax(curr_labels, 0)))

            # XXX: Softmax over choices instead of all possible entities.
            losses.append(
                tf.nn.softmax_cross_entropy_with_logits(raw_preds, curr_labels))

        self.loss = tf.reduce_mean(losses)
        self.accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          FLAGS.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.optimizer = optimizer.apply_gradients(zip(grads, tvars))


def run_epoch(data_iter, model, session, choices_length, is_training=True):
    for i, (question, context, choice_batch, lab_batch, map_batch,
            cont_len, qs_len) in enumerate(data_iter):

        lb = LabelBinarizer()
        lb.fit(np.arange(choices_length))
        curr_labels = lb.transform(
            [int(label[-1]) for label in lab_batch])

        curr_choices = np.zeros((question.shape[0], choices_length))
        for ind, choices in enumerate(choice_batch):
            indices = [int(i[-1]) for i in choices.split(" ")]
            curr_choices[ind, indices] = 1

        # Obtain question embedding from the last-step of the LSTM.
        feed_dict = {
            model.batch_size: question.shape[0],
            model.question: question,
            model.sequence_length: qs_len,
            model.context: context,
            model.context_length: cont_len,
            model.choices: curr_choices,
            model.labels: curr_labels}

        if is_training:
            ops = {"loss": model.loss, "accuracy": model.accuracy,
                   "optimizer": model.optimizer}
        else:
            ops = {"loss": model.loss, "accuracy": model.accuracy}
        vals = session.run(ops, feed_dict)
        print("accuracy on batch %d, %0.3f" % (i, vals['accuracy']))
    #print(vals['loss'])


def main():
    if not FLAGS.data_path:
        raise ValueError("Must set data_path to the data directory.")
    data_path = FLAGS.data_path
    contexts, questions, choices, labels, choices_map, \
        context_lens, qs_lens = load_data(FLAGS.data_path)
    vocabulary = get_vocab(questions, contexts, min_frequency=10)
    contexts = vocab_transform(contexts, vocabulary)
    questions = vocab_transform(questions, vocabulary)
    all_choices = build_choices(choices)
    vocab_size = len(vocabulary.vocabulary_)

    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                                                FLAGS.init_scale)

    with tf.variable_scope("rc_model", reuse=None, initializer=initializer):
        model = QuesLSTMAnsEmbedding(
            vocab_size=vocab_size, all_choices=all_choices,
            batch_size=FLAGS.batch_size)

    with tf.variable_scope("rc_model", reuse=True):
        test_model = QuesLSTMAnsEmbedding(
            vocab_size=vocab_size, all_choices=all_choices,
            batch_size=contexts.shape[0])

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    test_iter = batch_iter(
        contexts, questions, choices, labels, choices_map,
        context_lens, qs_lens,
        batch_size=contexts.shape[0],
        question_num_steps=FLAGS.num_steps,
        context_num_steps=contexts.shape[1])

    for i in range(FLAGS.num_epochs):
        data_iter = batch_iter(
            contexts, questions, choices, labels, choices_map,
            context_lens, qs_lens,
            batch_size=FLAGS.batch_size,
            random_state=i, question_num_steps=FLAGS.num_steps,
            context_num_steps=contexts.shape[1])
        print("Running epoch %d" % i)

        run_epoch(data_iter, model, session, len(all_choices))
        run_epoch(test_iter, test_model, session, len(all_choices),
                  is_training=False)


if __name__ == "__main__":
    main()

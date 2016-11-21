import numpy as np
from sklearn.preprocessing import LabelBinarizer
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
flags.DEFINE_integer("ans_dim", 300, "Dimensionality of the answer.")
flags.DEFINE_float("init_scale", 0.1, "Random initialization.")
flags.DEFINE_float("max_grad_norm", 5, "Clips gradient above this.")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")


FLAGS = flags.FLAGS


class QuesLSTMAnsEmbedding(object):
    def __init__(self, vocab_size, all_choices, is_training=True):
        self.len_choices = len(all_choices)
        batch_size = FLAGS.batch_size
        num_steps = FLAGS.num_steps
        hidden_size = FLAGS.qs_hidden_size
        self.vocab_size = vocab_size

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
        self.question_output = outputs[-1]

    def get_attention(self):
        batch_size = FLAGS.batch_size
        # Should be the output of the last-step of the question-LSTM.
        self.question_embedding = tf.placeholder(
            tf.float32, shape=[batch_size, FLAGS.qs_hidden_size])
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

        # Calculate attentions.
        losses = []
        correct = []
        for num_batch in range(batch_size):
            curr_qs_emb = self.question_embedding[num_batch]
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


def run_epoch(data_iter, model, session, choices_length):
    for i, (qs_batch, cont_batch, choice_batch, lab_batch, map_batch,
         cont_len, qs_len) in enumerate(data_iter):

        lb = LabelBinarizer()
        lb.fit(np.arange(choices_length))
        curr_labels = lb.transform(
            [int(label[-1]) for label in lab_batch])

        curr_choices = np.zeros((FLAGS.batch_size, choices_length))
        for ind, choices in enumerate(choice_batch):
            indices = [int(i[-1]) for i in choices.split(" ")]
            curr_choices[ind, indices] = 1

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
            ops = {
                "output": model.question_output,
                "state": model.final_state}
            vals = session.run(ops, feed_dict)
            state = vals["state"]
            qs_embedding = vals['output']

        context = np.hstack((cont_batch))
        feed_dict = {
            model.question_embedding: qs_embedding,
            model.context: context,
            model.context_length: cont_len,
            model.choices: curr_choices,
            model.labels: curr_labels}
        ops = {"loss": model.loss, "accuracy": model.accuracy, "optimizer": model.optimizer}
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
    vocab_size = len(vocabulary.vocabulary_)
    all_choices = build_choices(choices)
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale,
                                                FLAGS.init_scale)

    with tf.variable_scope("rc_model", reuse=None, initializer=initializer):
        model = QuesLSTMAnsEmbedding(
            vocab_size=vocab_size, all_choices=all_choices, is_training=True)
        model.get_attention()

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
        print("Running epoch %d" % i)
        run_epoch(data_iter, model, session, len(all_choices))

if __name__ == "__main__":
    main()

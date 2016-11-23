from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import shutil

import numpy as np
import bidir_rnn
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.preprocessing import LabelBinarizer
import reader as rn

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_wdw", '/scratch/ceb545/nlp/project/who_did_what/Strict/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", './runs/dump',
                    "Model output directory.")
flags.DEFINE_string("checkpoint_path", './runs/dump',
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
            
class Attention(object):
    def __init__(self,q_output,c_output,batch_size,size):
        #bilinear_weights = tf.ones([2*size,2*size])
        bilinear_weights = tf.get_variable("bilinear_w",[2*size,2*size],dtype=data_type())

        # diag_part returns tensor with bilinear values for each batch (non-diags are uselss)
        bilinear_terms = [tf.diag_part(tf.matmul(\
                tf.matmul(q_output,bilinear_weights),context_term,transpose_b=True))\
                for context_term in c_output]
        # packs bilinear values for each word into a matrix of dim batch_size x num context
        bilinear_matrix = tf.pack(bilinear_terms)

        print("bilinear shape is %s; should be bs x cont_seq"%bilinear_matrix.get_shape())
        self.alphas = tf.nn.softmax(bilinear_matrix)
        print("alpha_s:%s"%self.alphas.get_shape())
        alphas_unpacks = tf.unpack(self.alphas,axis=1)
        c_unpacks = tf.unpack(tf.pack(c_output,axis=1))
        print("c output len: %s"%len(c_output))
        print(alphas_unpacks[0].get_shape())
        print(c_unpacks[0].get_shape())
        alpha_mult  = [tf.matmul(tf.expand_dims(x,1),y,transpose_a=True) \
                for x,y in zip(alphas_unpacks,c_unpacks)]

        self.output = tf.reduce_sum(tf.pack(alpha_mult),
                reduction_indices=1)

class BiLSTM(object):
    """
    Bidirectional LSTM
    """
    def __init__(self, c_x, sequence_lengths,
                  config, num_steps, embedding,name,is_tuple=False):

        self.batch_size = config.batch_size
        self.size = config.hidden_size
        self.num_steps = num_steps
        self.c_x = c_x
        self.sequence_lengths = sequence_lengths
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.size, forget_bias=0.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(
            self.size, forget_bias=0.0, state_is_tuple=True)
        if config.keep_prob < 1:
          lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_fw_cell, output_keep_prob=config.keep_prob)
          lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
              lstm_bw_cell, output_keep_prob=config.keep_prob)
        self._initial_state_fw = lstm_fw_cell.zero_state(
            self.batch_size, data_type())
        self._initial_state_bw = lstm_bw_cell.zero_state(
            self.batch_size, data_type())
        self._initial_state = (self._initial_state_fw, self._initial_state_bw)
        inputs = tf.nn.embedding_lookup(embedding, self.c_x)
        if config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(input_step, [1])
                  for input_step in tf.split(1, self.num_steps, inputs)]
        self.outputs, self.state_fw, self.state_bw \
                 = tf.nn.bidirectional_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                inputs,
                initial_state_fw=self._initial_state[0],
                initial_state_bw=self._initial_state[1],
                sequence_length=self.sequence_lengths,scope="BiRNN_%s"%name)
        self._final_state = (self.state_fw, self.state_bw)


class Model(object):
    """The PTB model."""

    def __init__(self, config, vocab_size, labels_idx,
                 context_steps, question_steps):
        
        labels_size = len(labels_idx)
        self.labels_idx = labels_idx
        self.batch_size = config.batch_size
        self.size = config.hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = config.embedding_size

        self.context_steps = context_steps
        self.question_steps = question_steps

        self.q_x = tf.placeholder(tf.int32,[self.batch_size,self.question_steps])
        self.q_lengths = tf.placeholder(tf.int32, [self.batch_size])
        print("qu shape: %s"%self.q_x.get_shape())

        self.c_x = tf.placeholder(tf.int32, [self.batch_size, self.context_steps])
        self.c_lengths = tf.placeholder(tf.int32, [self.batch_size])

        self.input_y = tf.placeholder(tf.int32, [self.batch_size])
        self.encoded_y = tf.placeholder(
            tf.int32, [self.batch_size, labels_size])
        # the number of choices
        self.choices = tf.placeholder(tf.bool, [self.batch_size,labels_size])

        with tf.device("/cpu:0"):
          embedding = tf.get_variable(
              "embedding", [self.vocab_size, self.embedding_size], dtype=data_type())

        # bidirectional lstm - context
        context_lstm = BiLSTM(self.c_x, self.c_lengths,
                               config, self.context_steps, embedding,
                              name='context')

        # bidirectional lstm - question
        question_lstm = BiLSTM(self.q_x, self.q_lengths,
                config,self.question_steps,embedding, name="question")

        print("state shape: %s"%question_lstm.state_fw.h.get_shape())

        self._initial_state = (context_lstm._initial_state,question_lstm._initial_state)
        self._final_state = (context_lstm._final_state,question_lstm._final_state)

        q_hidden_state = tf.concat(1,(question_lstm.state_fw.h, question_lstm.state_bw.h))

        attn = Attention(q_hidden_state, context_lstm.outputs,self.batch_size,self.size)
        print("attn shape: %s"%attn.output.get_shape())
        self.attn_softmax_W = tf.get_variable("softmax_w",[2*self.size,labels_size],dtype=data_type())
        self._attn_logits = tf.matmul(attn.output,self.attn_softmax_W)

        # Cross-entropy loss over final output.
        self._cost = cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self._attn_logits, self.encoded_y))

        #self._predictions = tf.argmax(tf.select(self.choices,self._attn_logits,\
        #        -1000*tf.ones(self._attn_logits.get_shape())), 1)
        self._predictions = tf.argmax(self._attn_logits,\
                 1)
        correct_preds = tf.equal(tf.to_int32(self._predictions), self.input_y)
        self._acc = tf.reduce_mean(tf.cast(correct_preds, "float"))

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        #optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        self._test = self.attn_softmax_W

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
    init_scale = 0.01
    learning_rate = 0.01
    max_grad_norm = 10
    hidden_size = 150
    embedding_size = 250
    max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 32

def mask_choices(indices,choices):
    choices_mask = [np.zeros(len(indices)) for x in choices]
    for i,x in enumerate(choices_mask):
        curr = x
        for j,choice in enumerate(choices[i].split(" ")):
            if choice in indices:
                curr[indices.index(choice)]=1
        choices_mask[i] = curr
    return np.array(choices_mask)>0


def run_epoch(session, model, input, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    context_steps = model.context_steps
    question_steps = model.question_steps
    all_accs = []
    for j, batch in enumerate(input):

        questions, context, choices, labels, map, context_lens, qs_lens = batch
        choices_mask = mask_choices(model.labels_idx,choices)

        context_lens = [min(x,context_steps) for x in context_lens]
        qs_lens = [min(x,question_steps) for x in qs_lens]
        #print(context_lens)
        #print(qs_lens)

        lb = LabelBinarizer()
        lb.fit(model.labels_idx)

        mapped_labels = lb.transform(labels)
        actual_labels = [int(lab[-1]) for lab in labels]

        state = session.run(model.initial_state)

        fetches = {
          "cost": model.cost,
          "final_state": model.final_state,
          "predictions": model.predictions,
          "logits":model.logits,
          "targets": model.targets,
          "acc": model.acc,
          "test": model.test
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        feed_dict = {}
        feed_dict[model.initial_state] = state
        feed_dict[model.c_x] = context
        feed_dict[model.c_lengths] = context_lens
        feed_dict[model.q_x] = questions
        feed_dict[model.q_lengths] = qs_lens
        feed_dict[model.input_y] = actual_labels
        feed_dict[model.encoded_y] = mapped_labels
        feed_dict[model.choices] = choices_mask 

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        #state = vals["final_state"]
        all_accs.append(vals["acc"])
        if ((verbose) & (j%10==0)):
            print("batch %s; accuracy: %s" % (j, vals["acc"]))
            print(vals['logits'])
            #print(vals['test'])
            print("predictions: %s" %vals["predictions"].T)
    return np.mean(all_accs)


def main(_):
    if (os.path.exists(FLAGS.save_path)):
        shutil.rmtree(FLAGS.save_path)
    os.makedirs(FLAGS.save_path)
    t_log = open(os.path.join(FLAGS.save_path,'train.txt'),'w')
    v_log = open(os.path.join(FLAGS.save_path,'val.txt'),'w')
    te_log = open(os.path.join(FLAGS.save_path,'test.txt'),'w')


    train_path = os.path.join(FLAGS.data_wdw, 'test')
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
    c_steps = 50
    
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        print("Loading model..")
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Model(config=config, vocab_size=train.vocab_size,
                             labels_idx=train.labels_idx, context_steps=c_steps,
                             question_steps = q_steps)
            #tf.scalar_summary("Training Loss", m.cost)
            #tf.scalar_summary("Accuracy",m.acc) 
            #tf.scalar_summary("Learning Rate", m.lr)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            all_st = time.time()
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
                st = time.time()
                train_acc = run_epoch(session, m, train_iter, eval_op=m.train_op,
                          verbose=True)
                print("Epoch time: %s"%(time.time()-st))
                t_log.write("%s,%s,%s\n"%(i,time.time()-st,train_acc))
                print("\nChecking on validation set.")
                st = time.time()
                val_acc = run_epoch(session, m, val_iter, eval_op=None,
                          verbose=False)
                print("\nAvg. Val Accuracy: %s\n"%val_acc)
                v_log.write("%s,%s,%s\n"%(i,time.time()-st,val_acc))
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, os.path.join(FLAGS.save_path,'model'),\
                        global_step=sv.global_step)
                    #saver.save(session,FLAGS.save_path,global_step=sv.global_step)
            test_iter = rn.batch_iter(
                test.contexts, test.questions,
                test.choices, test.labels, test.choices_map, test.context_lens,
                test.qs_lens, batch_size=config.batch_size,
                context_num_steps=c_steps,
                question_num_steps=q_steps)
            print("\nChecking on test set.")
            test_acc = run_epoch(session, m, test_iter, eval_op=None,
                          verbose=False)
            te_log.write("%s,%s\n"%(time.time()-all_st,test_acc))
            print("\nAvg. Test Accuracy: %s\n"%test_acc)
            te_log.close()
            v_log.close()
            t_log.close()

if __name__ == "__main__":
  tf.app.run()

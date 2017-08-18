from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import glob
import hashlib
import os.path
import random
import re
import sys
import tarfile
import pickle
import csv

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import embedding_ops

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summaries_train_dir', 'logs/summaries_train/', """Where to save summary logs for training in TensorBoard.""")
tf.app.flags.DEFINE_string('summaries_finetune_dir', 'logs/summaries_finetune/', """Where to save summary logs for finetuning in TensorBoard.""")
tf.app.flags.DEFINE_string('model_train_dir','logs/models_trained',""" Checkpoints and trained models to finetune them""")
tf.app.flags.DEFINE_string('model_finetune_dir','logs/models_finetuned', """ Checkpoints and finetuned models saved""")

tf.app.flags.DEFINE_integer('iterations',2000, """ Total number of training iterations for a particular word(not epochs) """)
tf.app.flags.DEFINE_integer('saving_iter',100,"""  Iterations after which model file should be updated  """)


class Config(object):
  
  hidden_size_lstm = 16
  num_lstm_layers = 1
  num_class = 28
  learning_rate = 0.001

  #Parameters which can be changed after model instantiation
  batch_size = 1
  step_size = 9  # Maximum frame size of all videos present in UCF-101 dataset

class model(object):

  def __init__(self, is_training, config):
   
    self._batch_size = batch_size = config.batch_size 
    self._step_size = step_size = config.step_size
    self._hidden_size_lstm = hidden_size_lstm = config.hidden_size_lstm
    self._num_lstm_layers = num_lstm_layers = config.num_lstm_layers   
    self._num_class = num_class = config.num_class
    self._learning_rate = tf.Variable(config.learning_rate, trainable=False)

    with tf.name_scope('input'):
      self._init_state = init_state = tf.placeholder(tf.float32,  shape=[batch_size, hidden_size_lstm], name='feature_input')
      self._input_targets = input_targets = tf.placeholder(tf.int32, shape=[batch_size, step_size])
    #print("\033[91m I .. am... here.. 1..  \033[0m")
    layer_name = 'LSTM'
    with tf.name_scope(layer_name):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_lstm, forget_bias=0.0, state_is_tuple=True)
      #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_lstm_layers)
      #self._initial_state = cell.zero_state(batch_size, tf.float32)
      cell = lstm_cell
      
      embedding = variable_scope.get_variable("embedding",[num_class, hidden_size_lstm]) 
      with variable_scope.variable_scope("rnn_decoder"):
        state = (init_state,init_state)
        #state2 = tf.zeros([batch_size, cell.state_size])
        #print(cell.state_size)
        outputs = []
        prev = None
        #inp = np.zeros([batch_size,1])
        inp = tf.zeros([batch_size], tf.int32)
        inp = embedding_ops.embedding_lookup(embedding, inp)
        for i in xrange(step_size):
          #print(i)
          if prev is not None:
            with variable_scope.variable_scope("loop_function", reuse=True):
              inp = prev
          if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
          output, state = cell(inp, state)
          #print(inp,state)
          #output, state = rnn.dynamic_rnn(cell, inp, initial_state=state)
          #print(inp)
          #print(state)
          #print(output)
          outputs.append(output)
          prev = output

      # outputs and state is achieved here....
       
    
      #inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, step_size, input_feature)]
      #outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state, sequence_length=sequence_len)
      self._state = state
      final_output = tf.reshape(tf.concat(1,outputs), [-1, hidden_size_lstm])

    layer_name = 'fully_connected'
    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        self._fc_weights = fc_weights = tf.Variable(tf.truncated_normal([hidden_size_lstm, num_class], stddev=0.001), name='fc_weights')
        #self.variable_summaries(fc_weights, layer_name + '/weights')
      with tf.name_scope('biases'):
        self._fc_biases = fc_biases = tf.Variable(tf.zeros([num_class]), name='fc_biases')
        #self.variable_summaries(fc_biases, layer_name + '/biases')
      with tf.name_scope('Wx_plus_b'):
        self._final_logits = final_logits = tf.matmul(final_output, fc_weights) + fc_biases
        #tf.histogram_summary(layer_name + '/pre_activations_final', final_logits)

    self._final_tensor = final_tensor = tf.nn.softmax(final_logits,name="final_tensor_name")
    #tf.histogram_summary("final_tensor_name" + '/activations', final_tensor)
    #mask_reshape = tf.reshape(tf.sign(tf.reduce_max(tf.abs(input_feature), reduction_indices=2)), [-1])
    self._final_result = final_result = tf.argmax(final_tensor,1, name='final_output')

    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        self._correct_prediction = correct_prediction = tf.equal(final_result, tf.cast(tf.reshape(self._input_targets, [-1]), tf.int64))
      with tf.name_scope('accuracy'):
        temp_step = (tf.cast(correct_prediction, tf.float32))
        self._evaluation_step = evaluation_step = tf.reduce_sum(temp_step)/(batch_size*step_size)
      tf.scalar_summary('accuracy', evaluation_step)

   
    if not is_training:
      return

    # Part exclusively required for training the model 
    with tf.name_scope('loss'):
      loss = tf.nn.seq2seq.sequence_loss_by_example([final_logits], [tf.reshape(self._input_targets, [-1])], [tf.ones([batch_size*step_size])])
      with tf.name_scope('total'):
        self._loss_mean = loss_mean = tf.reduce_sum(loss) / (batch_size*step_size)
      tf.scalar_summary('loss',loss_mean)

    with tf.name_scope('train'):
      optimizer = tf.train.AdamOptimizer(self._learning_rate)
      #self._train_step = train_step = optimizer.minimize(loss_mean)
      self._tvars = tvars = tf.trainable_variables()
      self._grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
      grads = self._grads
      #self._grads_and_vars = grads_and_vars = optimizer.compute_gradients(loss)
      self._train_step = train_step = optimizer.apply_gradients(zip(grads,tvars))
      

  #  self._new_learning_rate = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
  #  self._learning_rate_update = tf.assign(self._learning_rate, self._new_learning_rate)    

  #def assign_learning_rate(self, session, learning_rate_value):
  #  session.run(self._learning_rate_update, feed_dict={self._new_learning_rate: learning_rate_value})

  #def variable_summaries(self, var, name):
  #  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  #  with tf.name_scope('summaries'):
  #    mean = tf.reduce_mean(var)
  #    tf.scalar_summary('mean/' + name, mean)
  #    with tf.name_scope('stddev'):
  #      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
  #    tf.scalar_summary('sttdev/' + name, stddev)
  #    tf.scalar_summary('max/' + name, tf.reduce_max(var))
  #    tf.scalar_summary('min/' + name, tf.reduce_min(var))
  #    tf.histogram_summary(name, var)

  @property
  def input_targets(self):
    return self._input_targets

  @property
  def init_state(self):
    return self._init_state

  @property
  def loss_mean(self):
    return self._loss_mean

  @property
  def train_step(self):
    return self._train_step

  @property
  def evaluation_step(self):
    return self._evaluation_step

  @property
  def state(self):
    return self._state

  @property
  def correct_predictions(self):
    return self._correct_prediction

  @property
  def learning_rate(self):
    return self._learning_rate
  
  @property
  def grads_and_vars(self):
    return self._grads_and_vars

  @property
  def tvars(self):
    return self._tvars

  @property
  def grads(self):
    return self._grads

  @property
  def final_result(self):
    return self._final_result

def main(_):

  summaries_dir = FLAGS.summaries_train_dir
  model_dir = FLAGS.model_train_dir
  
  #CLearing summaries and model directories if exists
  if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
  tf.gfile.MakeDirs(summaries_dir)
  if tf.gfile.Exists(model_dir):
    tf.gfile.DeleteRecursively(model_dir)
  tf.gfile.MakeDirs(model_dir)

  print("\033[95m Preparing the model...\033[0m")
  sess = tf.Session()
  config = Config()
  m = model(is_training=True, config=config)  
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(summaries_dir+'/train', sess.graph)
  test_writer = tf.train.SummaryWriter(summaries_dir+'/test',sess.graph)
  init = tf.initialize_all_variables()
  sess.run(init)

  print("\033[95m Preparing training data...\033[0m")
  #Training Data Preparation
  with open('data/dataset_size16.p') as f:
    inp_words, inp_labels, fe = pickle.load(f)
  
  r_index = random.randint(1,inp_labels.shape[0]-1)
  learn_word1 = np.reshape(fe[inp_words[r_index],:],[1,config.hidden_size_lstm])
  learn_label1 = np.reshape(inp_labels[r_index,:],[1,config.step_size])
  feed_dict1 = {m.init_state: learn_word1, m.input_targets: learn_label1}

  accuracy = []
  t=0
  for i in range(5):
    for j in range(FLAGS.iterations):
      t=t+1
      train_summary, _ = sess.run([merged, m.train_step], feed_dict=feed_dict1)
      train_writer.add_summary(train_summary, t)
      train_accuracy, loss_value, tr_final_result = sess.run([m.evaluation_step, m.loss_mean, m.final_result], feed_dict=feed_dict1)
      test_summary, test_accuracy, te_final_result = sess.run([merged, m.evaluation_step, m.final_result], feed_dict=feed_dict1)
      test_writer.add_summary(test_summary, t)
      accuracy.append(test_accuracy)
    
    for j in range(1500):
      t=t+1
      r_index = random.randint(1,inp_labels.shape[0]-1)
      learn_word2 = np.reshape(fe[inp_words[r_index],:],[1,config.hidden_size_lstm])
      learn_label2 = np.reshape(inp_labels[r_index,:],[1,config.step_size])
      feed_dict2 = {m.init_state: learn_word2, m.input_targets: learn_label2}  
      train_summary, _ = sess.run([merged, m.train_step], feed_dict=feed_dict2)
      train_writer.add_summary(train_summary, t)
      train_accuracy, loss_value, tr_final_result = sess.run([m.evaluation_step, m.loss_mean, m.final_result], feed_dict=feed_dict2)
      test_summary, test_accuracy, te_final_result = sess.run([merged, m.evaluation_step, m.final_result], feed_dict=feed_dict1)
      test_writer.add_summary(test_summary, t)
      accuracy.append(test_accuracy)
  f = open('results/results_exp4.csv', 'a')
  writer = csv.writer(f,quoting=csv.QUOTE_MINIMAL)
  writer.writerow(accuracy)
  f.close()

if __name__ == '__main__':
  tf.app.run()



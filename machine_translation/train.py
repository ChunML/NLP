from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import codecs
import numpy as np

# TODO: Use tf.app.flags
UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
UNK_token = 0
encoder_hidden_size = 512
decoder_hidden_size = 512
encoder_num_layers = 2
decoder_num_layers = 2
batch_size = 128
keep_prob = 0.7
learning_rate = 0.01
source_max_length = 50
target_max_length = 50
max_gradient = 5.0
num_iterations = 12000
print_every = 100
save_every = 1000

# ======================== DATA READING =============================
def load_vocab(vocab_file):
  vocab = []
  with codecs.getreader('utf-8')(tf.gfile.GFile(vocab_file, 'r')) as f:
    vocab_size = 0
    for word in f:
      vocab.append(word.strip())
      vocab_size += 1
  return vocab, vocab_size

def create_input_data(source_data_file, target_data_file,
                      source_vocab_file, target_vocab_file):
  source_dataset = tf.data.TextLineDataset(tf.gfile.Glob(source_data_file))
  target_dataset = tf.data.TextLineDataset(tf.gfile.Glob(target_data_file))
  source_vocab = lookup_ops.index_table_from_file(
    source_vocab_file, default_value=UNK_token)
  target_vocab = lookup_ops.index_table_from_file(
    target_vocab_file, default_value=UNK_token)

  output_buffer_size = batch_size * 1000

  source_eos_id = tf.cast(source_vocab.lookup(tf.constant(EOS)), tf.int32)
  target_sos_id = tf.cast(target_vocab.lookup(tf.constant(SOS)), tf.int32)
  target_eos_id = tf.cast(target_vocab.lookup(tf.constant(EOS)), tf.int32)

  dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
  dataset = dataset.map(
    lambda src, tgt: (tf.string_split([src]).values,
                      tf.string_split([tgt]).values)).prefetch(output_buffer_size)
  dataset = dataset.filter(
    lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
  dataset = dataset.map(
    lambda src, tgt: (src[:source_max_length], tgt[:target_max_length]))
  dataset = dataset.prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt: (tf.cast(source_vocab.lookup(src), tf.int32),
                      tf.cast(target_vocab.lookup(tgt), tf.int32)))
  dataset = dataset.prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt: (tf.reverse(src, axis=[0]),
                      tf.concat(([target_sos_id], tgt), 0),
                      tf.concat((tgt, [target_eos_id]), 0))).prefetch(output_buffer_size)

  dataset = dataset.map(
    lambda src, tgt_in, tgt_out: (
      src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in))).prefetch(output_buffer_size)

  dataset = dataset.shuffle(100).repeat().padded_batch(
    batch_size,
    padded_shapes=(tf.TensorShape([None]),
                   tf.TensorShape([None]),
                   tf.TensorShape([None]),
                   tf.TensorShape([]),
                   tf.TensorShape([])),
    padding_values=(source_eos_id,
                    target_eos_id,
                    target_eos_id,
                    0,
                    0))

  iterator = dataset.make_initializable_iterator()

  return iterator.get_next(), iterator.initializer, source_vocab, target_vocab

# ======================== SEQ2SEQ NETWORK =============================
def create_network(source_sequence,
                   target_sequence_in, target_sequence_out,
                   source_vocab, target_vocab,
                   source_sequence_length,
                   target_sequence_length,
                   source_vocab_size,
                   target_vocab_size):
  with tf.variable_scope('encoder'):
    encoder_embedding = tf.get_variable(
      'encoder_embedding_weights',
      [source_vocab_size, encoder_hidden_size],
      dtype=tf.float32,
      initializer=tf.initializers.random_uniform(-1, 1, dtype=tf.float32))
    # source_sequence = tf.reverse(source_sequence, axis=[1])
    source_sequence = tf.transpose(source_sequence)
    encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, source_sequence)

    # TODO: Update to bidirectional RNN
    def _create_encoder_cell(hidden_size):
      cell =  tf.nn.rnn_cell.LSTMCell(hidden_size)
      return tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    encoder_lstm = tf.nn.rnn_cell.MultiRNNCell(
      [_create_encoder_cell(encoder_hidden_size) for _ in range(encoder_num_layers)])
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      encoder_lstm,
      encoder_embedded,
      dtype=tf.float32,
      time_major=True,
      sequence_length=source_sequence_length)

  with tf.variable_scope('decoder'):
    decoder_embedding = tf.get_variable(
      'decoder_embedding_weights',
      [target_vocab_size, decoder_hidden_size],
      dtype=tf.float32,
      initializer=tf.initializers.random_uniform(-1, 1, dtype=tf.float32))
    target_sequence_in = tf.transpose(target_sequence_in)
    decoder_embedded = tf.nn.embedding_lookup(decoder_embedding, target_sequence_in)

    # TODO: Update to bidirectional RNN
    def _create_decoder_cell(hidden_size):
      cell =  tf.nn.rnn_cell.LSTMCell(hidden_size)
      return tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    decoder_lstm = tf.nn.rnn_cell.MultiRNNCell(
      [_create_decoder_cell(decoder_hidden_size) for _ in range(decoder_num_layers)])
    decoder_output_layer = tf.layers.Dense(target_vocab_size, use_bias=False)

    decoder_initial_state = encoder_state

    helper = tf.contrib.seq2seq.TrainingHelper(
      decoder_embedded,
      target_sequence_length,
      time_major=True)

    my_decoder = tf.contrib.seq2seq.BasicDecoder(
      decoder_lstm,
      helper,
      decoder_initial_state,
      decoder_output_layer)

    decoder_outputs, decoder_final_states, _ = tf.contrib.seq2seq.dynamic_decode(
      my_decoder,
      output_time_major=True,
      swap_memory=True)

    target_sequence_out = tf.transpose(target_sequence_out)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_sequence_out, logits=decoder_outputs.rnn_output)
    loss_weights = tf.sequence_mask(
      target_sequence_length, tf.shape(target_sequence_out)[0],
      dtype=tf.float32)
    loss_weights = tf.transpose(loss_weights)

    loss = tf.reduce_sum(cross_entropy * loss_weights) / tf.to_float(batch_size)

  with tf.variable_scope('decoder', reuse=True):
    target_sos_id = tf.cast(target_vocab.lookup(tf.constant(SOS)), tf.int32)
    target_eos_id = tf.cast(target_vocab.lookup(tf.constant(EOS)), tf.int32)
    infer_sequence_in = tf.fill([batch_size], target_sos_id)

    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
      decoder_embedding,
      infer_sequence_in,
      target_eos_id)
    infer_decoder = tf.contrib.seq2seq.BasicDecoder(
      decoder_lstm,
      infer_helper,
      decoder_initial_state,
      decoder_output_layer)

    maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
    infer_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
      infer_decoder,
      maximum_iterations=maximum_iterations,
      output_time_major=True,
      swap_memory=True)
    preds = infer_decoder_outputs.sample_id
  return loss, source_sequence, target_sequence_in, preds

def create_train_op(loss):
  # global_step = tf.Variable(0, trainable=False)
  global_step = tf.train.get_or_create_global_step()

  params = tf.trainable_variables()
  gradients = tf.gradients(
    loss,
    params)
  clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, max_gradient)

  # TODO: Schedule learning_rate update
  opt = tf.train.AdamOptimizer(learning_rate)
  train_op = opt.apply_gradients(
    zip(clipped_gradients, params), global_step)
  return global_step, train_op

source_data_file = '../data/train.vi'
target_data_file = '../data/train.en'
source_vocab_file = '../data/vocab.vi'
target_vocab_file = '../data/vocab.en'

source_int_to_vocab, source_vocab_size = load_vocab(source_vocab_file)
target_int_to_vocab, target_vocab_size = load_vocab(target_vocab_file)

(source_sequence,
 target_sequence_in, target_sequence_out,
 source_sequence_length, target_sequence_length),\
 iterator_initializer, source_vocab, target_vocab = create_input_data(
  source_data_file, target_data_file,
  source_vocab_file, target_vocab_file)

loss, t_source_sequence, t_target_sequence_in, preds = create_network(
  source_sequence,
  target_sequence_in, target_sequence_out,
  source_vocab, target_vocab,
  source_sequence_length,
  target_sequence_length,
  source_vocab_size,
  target_vocab_size)

global_step, train_op = create_train_op(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
sess.run(iterator_initializer)

saver = tf.train.Saver()
latest_checkpoint = tf.train.latest_checkpoint('checkpoint')
if latest_checkpoint and tf.train.checkpoint_exists(latest_checkpoint):
  saver.restore(sess, latest_checkpoint)

for _ in range(num_iterations):
  i = global_step.eval(sess)
  if i >= num_iterations:
    print('Training complete!')
    break
  src_seq, tar_seq, predictions, loss_value, _ = sess.run(
    [t_source_sequence, t_target_sequence_in, preds, loss, train_op])
  if (i + 1) % print_every == 0:
    random_id = np.random.choice(src_seq.shape[1])
    print('Step {}: loss {:.4f}'.format(i + 1, loss_value))
    src_sent = ' '.join([source_int_to_vocab[ix] for ix in src_seq[::-1, random_id]])
    tar_sent = ' '.join([target_int_to_vocab[ix] for ix in tar_seq[:, random_id]])
    pred_sent = ' '.join([target_int_to_vocab[ix] for ix in predictions[:, random_id]])

    if EOS in src_sent:
      eos_index = src_sent.rindex(EOS)
      src_sent = src_sent[eos_index + len(EOS):]
    if EOS in tar_sent:
      eos_index = tar_sent.index(EOS)
      tar_sent = tar_sent[:eos_index]
    if EOS in pred_sent:
      eos_index = pred_sent.index(EOS)
      pred_sent = pred_sent[:eos_index]
    print('<src>', src_sent.encode('utf-8'))
    print('<dst>', tar_sent.encode('utf-8'))
    print('<pred>', pred_sent.encode('utf-8'))
    print()

  if (i + 1) % save_every == 0:
    print('Saving checkpoint for step {}...\n'.format(i + 1))
    saver.save(sess, 'checkpoint/model-{}.ckpt'.format(i + 1))

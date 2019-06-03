# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, word_to_index, index_to_embedding):
    self._hps = hps
    self._word_to_index = word_to_index
    self._index_to_embedding = index_to_embedding

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._tf_embedding_placeholder = tf.placeholder(tf.float32, shape=self._index_to_embedding.shape)

    # print ("add placeholder hps.batch_size: ", hps.batch_size) #train flag
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size.value], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='enc_padding_mask')
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')
    if hps.use_doc_vec.value:
      self._enc_tag_batch = tf.placeholder(tf.int32, [hps.batch_size.value], name='enc_tag_batch')

    # self._tf_word_ids = tf.placeholder(tf.int32, shape=[hps.batch_size.value])

    # decoder part
    # print ("add placeholder hps.mode: ", hps.mode) #train flag
    max_dec_steps = hps.max_dec_steps # ok
    if hps.mode.value != "decode":
      max_dec_steps = hps.max_dec_steps.value
    
    # print ("add placeholder max_dec_steps: ", max_dec_steps) # int
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size.value, max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size.value, max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, max_dec_steps], name='dec_padding_mask')
    
    # print("add placeholder hps.coverage: ", hps.coverage) #train flag
    if hps.mode.value=="decode" and hps.coverage.value:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='prev_coverage')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    # index_to_embedding
    feed_dict[self._tf_embedding_placeholder] = batch.index_to_embedding

    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
    if self._hps.use_doc_vec.value:
      feed_dict[self._enc_tag_batch] = batch.enc_tag_batch
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      # print ("add encoder self._hps.hidden_dim: ", self._hps.hidden_dim) #train flag
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      # print ("_add_encoder encoder_outputs[0]: ", encoder_outputs[0].shape) # (16, ?, 64)

      for one_lstm_cell in[cell_fw,cell_bw]:
        one_kernel, one_bias = one_lstm_cell.variables
        tf.summary.histogram("Kernel", one_kernel)
        tf.summary.histogram("Bias", one_bias)

      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
      # print ("_add_encoder after concat encoder_outputs: ", encoder_outputs.shape) # (16, ?, 128)
    return encoder_outputs, fw_st, bw_st


  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    # print ("reduce states self._hps.hidden_dim: ", self._hps.hidden_dim) #train flag
    hidden_dim = self._hps.hidden_dim.value
    with tf.variable_scope('reduce_final_st'):
      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state


  def _add_decoder(self, inputs):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    # print ("add decoder hps.hidden_dim: ", hps.hidden_dim) #train flag
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim.value, state_is_tuple=True, initializer=self.rand_unif_init)

#     one_kernel, one_bias = cell.variables
#     tf.summary.histogram("Kernel", one_kernel)
#     tf.summary.histogram("Bias", one_bias)

    # print ("add decoder hps.mode: ", hps.mode) #train flag
    # print ("add decoder hps.coverage: ", hps.coverage) #train flag
    prev_coverage = self.prev_coverage if hps.mode.value=="decode" and hps.coverage.value else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    # print ("add decoder hps.pointer_gen: ", hps.pointer_gen) #train flag
    # print ("_add_decoder self._enc_states: ", self._enc_states.shape)
    outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell, initial_state_attention=(hps.mode.value=="decode"), pointer_gen=hps.pointer_gen.value, use_coverage=hps.coverage.value, prev_coverage=prev_coverage)
    # print ("_add_decoder attn_dists: ", attn_dists[0].shape)
    return outputs, out_state, attn_dists, p_gens, coverage

  def _calc_final_dist(self, vocab_dists, attn_dists):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = len(self._index_to_embedding) + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
      # print ("_calc_final_dist self._hps.batch_size: ", self._hps.batch_size) #train flag
      extra_zeros = tf.zeros((self._hps.batch_size.value, self._max_art_oovs))
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      batch_nums = tf.range(0, limit=self._hps.batch_size.value) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
      indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      shape = [self._hps.batch_size.value, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      # print ("_calc_final_dist vocab_dists_extended: ", vocab_dists_extended[0].shape) # (16, ?)
      # print ("_calc_final_dist attn_dists_projected: ", attn_dists_projected[0].shape) # (16, ?)
      final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

      return final_dists

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    return

    # train_dir = os.path.join(FLAGS.log_root, "train")
    # vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    # self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    # summary_writer = tf.summary.FileWriter(train_dir)
    # config = projector.ProjectorConfig()
    # embedding = config.embeddings.add()
    # embedding.tensor_name = embedding_var.name
    # embedding.metadata_path = vocab_metadata_path
    # projector.visualize_embeddings(summary_writer, config)

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = len(self._index_to_embedding) # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      # print ("_add_seq2seq hps.rand_unif_init_mag: ", hps.rand_unif_init_mag) #train flag
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag.value, hps.rand_unif_init_mag.value, seed=123)
      # print ("_add_seq2seq hps.trunc_norm_init_std: ", hps.trunc_norm_init_std) #train flag
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std.value)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        # print ("_add_seq2seq hps.emb_dim: ", hps.emb_dim) #train flag
        # embedding = tf.get_variable('embedding', [vsize, hps.emb_dim.value], dtype=tf.float32, initializer=self.trunc_norm_init)
        tf_embedding = tf.Variable(
            tf.constant(0.0, shape=self._index_to_embedding.shape),
            trainable=False,
            name="embedding")
        # print ("_add_seq2seq self._index_to_embedding.shape: ", self._index_to_embedding.shape) # (1193516, 25)

        tf_embedding_init = tf_embedding.assign(self._tf_embedding_placeholder)

        # print ("_add_seq2seq hps.mode: ", hps.mode) #train flag
        # print ("_add_seq2seq tf_embedding: ", tf_embedding.shape) # (1193516, 25)
        if hps.mode.value=="train": self._add_emb_vis(tf_embedding) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(tf_embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        # print ("_add_seq2seq emb_enc_inputs: ", emb_enc_inputs.shape) # (16, ?, 25)
        if hps.use_doc_vec.value:
            embedding_doc = tf.get_variable('embedding_doc', [hps.subred_size.value, hps.emb_dim.value], dtype=tf.float32, initializer=self.trunc_norm_init)
            # print ("_add_seq2seq embedding_doc: ", embedding_doc.shape) # (10, 25)
            # print ("_add_seq2seq self._enc_tag_batch: ", self._enc_tag_batch.shape) # (16,)
            tag = tf.expand_dims(self._enc_tag_batch, 1)
            # print ("_add_seq2seq tag: ", tag.shape) # (16,1)
            emb_enc_doc = tf.nn.embedding_lookup(embedding_doc, tag) # self._enc_tag_batch (batch_size, 1)
            # print ("_add_seq2seq emb_enc_doc: ", emb_enc_doc.shape) # (16, 1, 25)
            # print ("emb_enc_inputs: ", tf.shape(emb_enc_inputs))
            cur_enc_steps = tf.shape(emb_enc_inputs)[1]
            emb_enc_doc = tf.broadcast_to(emb_enc_doc, [hps.batch_size.value, cur_enc_steps, hps.emb_dim.value])
            # print ("_add_seq2seq after broadcast_to dims emb_enc_doc: ", emb_enc_doc.shape) # (16, ?, 25)
            emb_enc_inputs_combine = tf.concat([emb_enc_inputs, emb_enc_doc], 2) # (batch_size, 1, emb_size+emb_doc_size)
        else:
            emb_enc_inputs_combine = emb_enc_inputs
        # print ("_add_seq2seq emb_enc_inputs_combine: ", emb_enc_inputs_combine.shape) # (16, ?, 50)
            
        emb_dec_inputs = [tf.nn.embedding_lookup(tf_embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
        # print ("_add_seq2seq emb_dec_inputs: ", len(emb_dec_inputs)) # 5
        # print ("_add_seq2seq emb_dec_inputs: ", emb_dec_inputs[0].shape) # (16, 25)

      # Add the encoder.
      enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs_combine, self._enc_lens)
      self._enc_states = enc_outputs
      # print ("_add_seq2seq enc_outputs: ", enc_outputs.shape) # (16, ?, 128)

      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(fw_st, bw_st)
      # print ("_add_seq2seq self._dec_in_state: ", self._dec_in_state.shape)

      # Add the decoder.
      with tf.variable_scope('decoder'):
        decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs)
        # print ("_add_seq2seq decoder_outputs[0]: ", decoder_outputs[0].shape) # (16, 64)

      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection'):
        # print ("_add_seq2seq hps.hidden_dim: ", hps.hidden_dim) #train flag
        w = tf.get_variable('w', [hps.hidden_dim.value, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        w_t = tf.transpose(w)
        v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        for i,output in enumerate(decoder_outputs):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          # print ("_add_seq2seq output: ", output.shape) # (16, 64)
          # print ("_add_seq2seq w: ", w.shape) # (64, 1193516)
          # print ("_add_seq2seq v: ", v.shape) # (1193516,)
          vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer
          # print ("_add_seq2seq vocab_scores[0]: ", vocab_scores[0].shape) # (16, 1193516) changing

        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
        # print ("_add_seq2seq vocab_dists[0]: ", vocab_dists[0].shape) # (16, 1193516) changing
 
      # print ("_add_seq2seq self.attn_dists[0]: ", self.attn_dists[0].shape) # (16, ?)
      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if FLAGS.pointer_gen:
        final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
        # print ("_add_seq2seq final_dists[0]: ", final_dists[0].shape) # (16, ?)
      else: # final distribution is just vocabulary distribution
        final_dists = vocab_dists

      if hps.mode.value in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
            # print ("_add_seq2seq hps.batch_size: ", hps.batch_size) #train flag
            batch_nums = tf.range(0, limit=hps.batch_size.value) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists):
              targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
              # print ("_add_seq2seq targets: ", targets.shape) # (16,)
              indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
              # print ("_add_seq2seq indices: ", indices.shape) # (16,2)
              gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
              # print ("_add_seq2seq gold_probs: ", gold_probs.shape) # (16,)
              losses = -tf.log(gold_probs)
              # print ("_add_seq2seq losses: ", losses.shape) # (16,)
              loss_per_step.append(losses)

            # Apply dec_padding_mask and get loss
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)
            # print ("_add_seq2seq self._loss: ", self._loss.shape) # ()

          else: # baseline model
            self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          tf.summary.scalar('loss', self._loss)

          # Calculate coverage loss from the attention distributions
          # print("_add seq2seq hps.coverage: ", hps.coverage) #train flag
          if hps.coverage.value:
            with tf.variable_scope('coverage_loss'):
              self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
              tf.summary.scalar('coverage_loss', self._coverage_loss)
            # print(" add seq2seq hps.cov_loss_wt: ", hps.cov_loss_wt) #train flag
            self._total_loss = self._loss + hps.cov_loss_wt.value * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)
    # print ("_add_seq2seq ")
    if hps.mode.value == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      final_dists = final_dists[0]
      topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size.value*2) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs = tf.log(topk_probs)


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    # print("_add_train_op self._hps.coverage: ", self._hps.coverage) #train flag
    loss_to_minimize = self._total_loss if self._hps.coverage.value else self._loss
    tvars_all = tf.trainable_variables()
    # print ("_add_train_op self._hps.fine_tune: ", self._hps.fine_tune) #train flag
    tvars = None
    if self._hps.fine_tune.value:
      print ("freeze encoder and decoder !!!")
      freeze_vars_enc = tf.trainable_variables(scope='seq2seq/encoder')
      freeze_vars_dec = tf.trainable_variables(scope='seq2seq/decoder')
      tvars = [v for v in tvars_all if v not in freeze_vars_enc and v not in freeze_vars_dec]
    else:
      print ("train entire model !!!")
      tvars = tvars_all
    
    # print ("tvars: ", tvars)

    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:0"):
      # print ("_add_train_op self._hps.max_grad_norm: ", self._hps.max_grad_norm) #train flag
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm.value)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    # print ("_add_train_op self._hps.lr: ", self._hps.lr) #train flag
    # print ("_add_train_op self._hps.adagrad_init_acc: ", self._hps.adagrad_init_acc) #train flag
    optimizer = tf.train.AdagradOptimizer(self._hps.lr.value, initial_accumulator_value=self._hps.adagrad_init_acc.value)
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    # print ("build_graph self._hps.mode: ", self._hps.mode) #train flag
    if self._hps.mode.value == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    # print ("run_train_step feed_dict: ", feed_dict)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }

    # print ("tvars_all: ", tf.trainable_variables())
    # cell_0_weights = [v for v in tf.trainable_variables()
                  # if v.name == 'seq2seq/encoder/bidirectional_rnn/fw/lstm_cell/kernel:0'][0]
    # cell_0_weights = tf.Print(cell_0_weights,[cell_0_weights], "cell_0_weights: ")

    # print("run_train_step self._hps.coverage: ", self._hps.coverage) #train flag
    if self._hps.coverage.value:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    # print("run_eval_step self._hps.coverage: ", self._hps.coverage) # eval flag
    if self._hps.coverage.value:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    
    # print ("enc_states: ", enc_states)
    # print ("dec_in_state: ", dec_in_state)

    return enc_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists
    }

    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens
    # print("decode_one_step self._hps.coverage: ", self._hps.coverage) # decode log
    if self._hps.coverage.value:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in range(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in range(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in range(beam_size)]

    return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss
